import numpy as np
import pandas as pd
from scipy import stats as st
import statsmodels.api as sm
from statsmodels.formula.api import ols


def calculate_rsi(series, window=14):
    """
    Calculates the Raw RSI value (0-100).
    Does not generate signals yet.
    """
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)

    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)

    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculate_macd_all(series, fast=12, slow=26, signal=9):
    """
    Calculates and returns all MACD components as a DataFrame.
    """
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist = macd_line - signal_line

    return pd.DataFrame({
        'macd_line': macd_line,
        'macd_signal': signal_line,
        'macd_hist': macd_hist
    }, index=series.index)


def calculate_volatility(series, window=20):
    """
    Calculates the rolling standard deviation of log returns.
    Represents the 'Risk' or 'Uncertainty' pillar.
    """
    return series.rolling(window=window).std()


def calculate_ma_dist(series, window=50):
    """
    Calculates the percentage distance from the Simple Moving Average.
    Formula: (Price - SMA) / SMA
    Represents the 'Mean Reversion' or 'Value' pillar.
    """
    sma = series.rolling(window=window).mean()
    return (series - sma) / sma


def technical_indicator_engine(df):
    """
    Computes all 4 technical pillars and cleans up formatting/NaNs.
    """
    df = df.sort_values(['Ticker', 'Date']).reset_index(drop=True)

    df['rsi_14'] = df.groupby('Ticker')['close_price'].transform(calculate_rsi)

    macd_cols = (
        df.groupby('Ticker')['close_price']
        .apply(calculate_macd_all)
        .reset_index(level=0, drop=True)
    )
    df = pd.concat([df, macd_cols], axis=1)

    df['volatility_20'] = df.groupby('Ticker')['log_return'].transform(calculate_volatility)

    df['ma_50_dist'] = df.groupby('Ticker')['close_price'].transform(calculate_ma_dist)

    technical_cols = ['rsi_14', 'macd_line', 'macd_signal', 'macd_hist', 'volatility_20', 'ma_50_dist']
    df[technical_cols] = df[technical_cols].fillna(0)

    df['target_log_return_t1'] = df.groupby('Ticker')['log_return'].shift(-1)
    df = df.dropna(subset=['target_log_return_t1']).copy()

    return df


def run_individual_hypothesis_tests(df, target, indicators):
    indicators = indicators
    target = target

    print(f"{'Indicator':<15} | {'Correlation':<12} | {'P-Value':<10} | {'Significant?'}")
    print("-" * 60)

    for ind in indicators:
        # Calculate Pearson correlation and two-tailed p-value
        corr, p_value = st.pearsonr(df[ind], df[target])

        is_sig = "YES" if p_value < 0.01 else "NO"
        print(f"{ind:<15} | {corr:>12.6f} | {p_value:>10.4e} | {is_sig}")


def run_multivariate_test(df, target):
    features = ['rsi_14', 'macd_hist', 'volatility_20', 'ma_50_dist']
    X = df[features].copy()
    y = df[target]

    X['RSI_x_Vol'] = X['rsi_14'] * X['volatility_20']
    X['MACD_x_Vol'] = X['macd_hist'] * X['volatility_20']

    X['RSI_x_MACD'] = X['rsi_14'] * X['macd_hist']
    X['RSI_x_MA'] = X['rsi_14'] * X['ma_50_dist']
    X['Vol_x_MA'] = X['volatility_20'] * X['ma_50_dist']

    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    return model


def create_categorical_signals(df):
    """
    Converts raw technical indicators into categorical signals (-1, 0, 1).
    Ensures zero data leakage between tickers via groupby.
    """
    # 1. RSI Fixed Thresholds
    # Capture extreme momentum (Oversold < 30, Overbought > 70)
    df['sig_rsi'] = np.select(
        [df['rsi_14'] < 30, df['rsi_14'] > 70], 
        [1, -1], default=0
    )

    hist = df['macd_hist']
    prev_hist = df.groupby('Ticker')['macd_hist'].shift(1)

    df['sig_macd'] = np.select(
        [(hist > 0) & (prev_hist <= 0), (hist < 0) & (prev_hist >= 0)],
        [1, -1], default=0
    )

    def get_vol_quantiles(x):
        q25, q75 = x.quantile(0.25), x.quantile(0.75)
        return np.select([x < q25, x > q75], [1, -1], default=0)

    df['sig_vol'] = df.groupby('Ticker')['volatility_20'].transform(get_vol_quantiles)

    def get_ma_zscore_signal(x):
        sma = x.rolling(window=50).mean()
        std = x.rolling(window=50).std()
        z_score = (x - sma) / std
        # Buy if price is > 2 StdDevs below SMA, Sell if > 2 StdDevs above
        return np.select([z_score < -2, z_score > 2], [1, -1], default=0)

    df['sig_ma_dist'] = df.groupby('Ticker')['close_price'].transform(get_ma_zscore_signal)

    signal_cols = ['sig_rsi', 'sig_macd', 'sig_vol', 'sig_ma_dist']
    df[signal_cols] = df[signal_cols].fillna(0).astype(int)

    return df


def verify_signal_performance(df, target):
    signals = ['sig_rsi', 'sig_macd', 'sig_vol', 'sig_ma_dist']
    target = target

    performance_metrics = []

    for sig in signals:
        stats_by_sig = df.groupby(sig)[target].agg(['mean', 'std', 'count'])

        buy_returns = df[df[sig] == 1][target]
        neutral_returns = df[df[sig] == 0][target]
        t_stat, p_val = st.ttest_ind(buy_returns, neutral_returns, nan_policy='omit')

        performance_metrics.append({
            'Signal': sig,
            'Buy_Mean_Return': stats_by_sig.loc[1, 'mean'] if 1 in stats_by_sig.index else np.nan,
            'Neutral_Mean_Return': stats_by_sig.loc[0, 'mean'],
            'T_Statistic': t_stat,
            'P_Value': p_val,
            'Significant_01': p_val < 0.01
        })

    return pd.DataFrame(performance_metrics)


def run_multivariate_anova(df, target, signals):
    target = target
    signals = signals

    anova_results = []
    for sig in signals:
        group_neg = df[df[sig] == -1][target]
        group_neu = df[df[sig] == 0][target]
        group_pos = df[df[sig] == 1][target]

        f_stat, p_val = st.f_oneway(group_neg, group_neu, group_pos)

        anova_results.append({
            'Indicator': sig,
            'F_Statistic': f_stat,
            'P_Value': p_val,
            'Reject_H0_01': p_val < 0.01
        })
    return pd.DataFrame(anova_results)


def run_signal_chi2_aligned(df, target):
    """
    Performs a Chi-squared test of independence between signal direction 
    and actual return direction.

    This method aligns:
    +1 (Buy)  -> expects positive return
    -1 (Sell) -> expects negative return

    Neutral signals (0) are excluded from the test.
    """

    indicators = ['sig_rsi', 'sig_macd', 'sig_vol', 'sig_ma_dist']
    results = []

    df = df.copy()
    df['actual_direction'] = np.sign(df[target])

    df = df[df['actual_direction'] != 0]

    for sig in indicators:
        sub_df = df[df[sig] != 0].copy()

        contingency = pd.crosstab(sub_df[sig], sub_df['actual_direction'])

        chi2, p, dof, ex = st.chi2_contingency(contingency)

        results.append({
            'Indicator': sig,
            'Chi2_Stat': chi2,
            'P_Value': p,
            'Significant_01': p < 0.01,
            'Observations': len(sub_df)
        })

    return pd.DataFrame(results)


def add_multi_day_targets(df, horizons=[2, 3, 5]):
    """
    Creates cumulative forward log returns for different time horizons.
    """
    for n in horizons:
        col_name = f'target_log_return_t{n}'

        df[col_name] = df.groupby('Ticker')['log_return'].transform(
            lambda x: sum(x.shift(-i) for i in range(1, n + 1))
        )

    return df.dropna(subset=['target_log_return_t5']).copy()


def normalize_targets_per_ticker(df, target_cols):
    """
    Normalizes (Z-scores) the target returns within each ticker group.
    This aligns the targets with the sentiment logic.
    """
    for col in target_cols:
        df[f'norm_{col}'] = df.groupby('Ticker')[col].transform(
            lambda x: (x - x.mean()) / x.std()
        )
    return df
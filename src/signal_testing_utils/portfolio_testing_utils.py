import numpy as np
import pandas as pd


def calculate_rebalanced_buy_and_hold(df,
                                      weights,
                                      tickers,
                                      initial_capital=1_000_000,
                                      risk_free_rate=0.03,
                                      trading_days=252
                                     ):
    """ Calculate portfolio metrics given a dataframe with chronological returns by dates and tickers """
    returns_df = df[tickers].copy()

    weights = weights[tickers]
    weights = weights / weights.sum()

    portfolio_returns = returns_df.dot(weights)

    portfolio_value = initial_capital * (1 + portfolio_returns).cumprod()
    ending_value = portfolio_value.iloc[-1]

    daily_std = portfolio_returns.std()
    annualized_std = daily_std * np.sqrt(trading_days)

    total_return = ending_value / initial_capital - 1
    years = len(portfolio_returns) / trading_days
    annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else np.nan

    sharpe_ratio = (
        (annualized_return - risk_free_rate) / annualized_std
        if annualized_std > 0 else np.nan
    )

    return (
        portfolio_returns,
        portfolio_value,
        ending_value,
        daily_std,
        annualized_std,
        annualized_return,
        sharpe_ratio
    )


def _calculate_daily_weights(active_buy_count, max_position_weight):
    """ Calculates the weight of the position on the same day """
    if active_buy_count > 0:
        return min(1 / active_buy_count, max_position_weight)
    return 0


def _execute_sell_logic(ticker,
                        current_shares,
                        open_price,
                        sell_cost_multiplier,
                        current_date,
                        prev_signal,
                        positions,
                        trade_log):
    """ Processes the full liduidation of a position. """
    cash_gain = 0
    if current_shares > 0:
        sell_price = open_price * sell_cost_multiplier
        proceeds = current_shares * sell_price
        cash_gain = proceeds

        trade_log.append({
            "Date": current_date,
            "Ticker": ticker,
            "Action": "SELL",
            "SignalUsed": prev_signal,
            "Shares": current_shares,
            "TradePrice": sell_price,
            "TradeValue": proceeds
        })
        positions[ticker] = 0.0
    return cash_gain


def _execute_buy_logic(ticker,
                       current_shares,
                       open_price,
                       buy_cost_multiplier,
                       portfolio_value_before,
                       dynamic_target_weight,
                       cash,
                       min_trade_dollars,
                       current_date,
                       prev_signal,
                       positions,
                       trade_log):
    """ Processes the opening (buy) or top-up of a long position"""
    cash_spent = 0
    current_position_value = current_shares * open_price
    target_position_value = portfolio_value_before * dynamic_target_weight
    gap_to_target = target_position_value - current_position_value

    if gap_to_target > 0:
        buy_price = open_price * buy_cost_multiplier
        amount_to_invest = min(gap_to_target, cash)

        if amount_to_invest >= min_trade_dollars:
            shares_to_buy = amount_to_invest / buy_price
            cash_spent = amount_to_invest
            positions[ticker] = current_shares + shares_to_buy

            trade_log.append({
                "Date": current_date,
                "Ticker": ticker,
                "Action": "BUY",
                "SignalUsed": prev_signal,
                "Shares": shares_to_buy,
                "TradePrice": buy_price,
                "TradeValue": amount_to_invest
            })
    return cash_spent


def _finalize_portfolio_valuation(positions, last_close_prices, cash):
    """Calculates the final position of the portfolio after the end of the period"""
    final_positions_valuation = []
    for ticker, shares in positions.items():
        if shares > 0 and ticker in last_close_prices:
            last_close = last_close_prices[ticker]
            market_value = shares * last_close
            final_positions_valuation.append({
                "Ticker": ticker, "Shares": shares, 
                "LastClose": last_close, "MarketValue": market_value
            })

    final_positions_df = pd.DataFrame(final_positions_valuation)
    cash_row = pd.DataFrame([{"Ticker": "CASH", "Shares": np.nan, "LastClose": 1.0, "MarketValue": cash}])

    final_positions_df = pd.concat([final_positions_df, cash_row], ignore_index=True)
    final_portfolio_value = final_positions_df["MarketValue"].sum()
    final_positions_df["Weight"] = final_positions_df["MarketValue"] / final_portfolio_value

    return final_positions_df.sort_values("MarketValue", ascending=False).reset_index(drop=True), final_portfolio_value


def backtest_signal_strategy(df,
                             initial_capital=1_000_000,
                             annual_cash_rate=0.03,
                             buy_cost_multiplier=1.01,
                             sell_cost_multiplier=0.99,
                             max_position_weight=0.15,
                             min_trade_dollars=100
                            ):

    """ Processes the chronological order of given buy or sell signals, given the defined constraints"""
    cash = initial_capital
    positions = {}
    daily_records = []
    trade_log = []
    daily_rate = annual_cash_rate / 365
    unique_dates = df["Date"].sort_values().unique()

    for current_date in unique_dates:
        day_data = df[df["Date"] == current_date].copy()
        close_prices = dict(zip(day_data["Ticker"], day_data["close_price"]))
        prev_close_prices = dict(zip(day_data["Ticker"], day_data["prev_close"]))

        holdings_value_before = sum(
            shares * prev_close_prices[t] 
            for t, shares in positions.items() 
            if t in prev_close_prices and pd.notna(prev_close_prices[t])
        )
        portfolio_value_before = cash + holdings_value_before

        active_buy_count = (day_data["prev_signal"] == 1).sum()
        dynamic_target_weight = _calculate_daily_weights(active_buy_count, max_position_weight)

        for _, row in day_data.iterrows():
            ticker, prev_signal, open_price = row["Ticker"], row["prev_signal"], row["open_price"]
            if pd.isna(prev_signal): continue

            if prev_signal == -1:
                cash += _execute_sell_logic(
                    ticker, positions.get(ticker, 0.0), open_price, 
                    sell_cost_multiplier, current_date, prev_signal, positions, trade_log
                )
            elif prev_signal == 1 and dynamic_target_weight > 0:
                cash -= _execute_buy_logic(
                    ticker, positions.get(ticker, 0.0), open_price, buy_cost_multiplier, 
                    portfolio_value_before, dynamic_target_weight, cash, min_trade_dollars, 
                    current_date, prev_signal, positions, trade_log
                )

        holdings_value_after = sum(
            shares * close_prices[t] 
            for t, shares in positions.items() 
            if shares > 0 and t in close_prices
        )
        total_portfolio_value = cash + holdings_value_after

        cash_interest = cash * daily_rate
        beginning_cash = cash
        cash += cash_interest

        daily_records.append({
            "Date": current_date, "BeginningCash": beginning_cash, "CashInterest": cash_interest,
            "Cash": cash, "HoldingsValue": holdings_value_after, "TotalPortfolioValue": total_portfolio_value,
            "NumOpenPositions": sum(1 for s in positions.values() if s > 0),
            "ActiveBuySignals": int(active_buy_count), "DynamicTargetWeight": dynamic_target_weight
        })

    daily_df = pd.DataFrame(daily_records)
    daily_df["DailyReturn"] = daily_df["TotalPortfolioValue"].pct_change().fillna(0)

    last_day_data = df.sort_values("Date").groupby("Ticker").tail(1)
    last_close_prices = dict(zip(last_day_data["Ticker"], last_day_data["close_price"]))

    final_positions_df, final_portfolio_value = _finalize_portfolio_valuation(
        positions, last_close_prices, cash
    )

    return daily_df, pd.DataFrame(trade_log), positions, final_positions_df, final_portfolio_value
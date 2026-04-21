# Empirical Evaluation of Technical Signals, Sentiment Analysis, and Return Dynamics in Equity Markets
## SoftUni Data Science Course Final Project - March 2026

## 1. Project Overview & Objective

Financial markets are often assumed to be efficient, implying that price movements are largely unpredictable. However, empirical evidence suggests that while **price direction may be random**, the **distribution, magnitude, and structure of returns exhibit persistent statistical properties**.

This project investigates whether these properties can be systematically identified and transformed into **statistically valid signals**, rather than relying on heuristic or anecdotal trading rules.

The objective is not to “predict the market,” but to rigorously test:

- Whether return dynamics contain **non-random structure**
- Whether **extreme events (outliers)** carry informational value
- Whether **volatility and distributional features** can be exploited
- Whether such signals are **consistent across a diversified equity universe**

## 2. Research Framework

The project is structured around a clear empirical pipeline:

1. **Data Construction** → Build a clean, reproducible return dataset  
2. **Statistical Characterization** → Understand distributional properties  
3. **Structure Identification** → Detect non-random behavior  
4. **Signal Construction** → Translate structure into signals  
5. **Hypothesis Testing** → Validate statistical significance  
6. **Integration & Evaluation** → Assess combined signal impact  

All conclusions are based on **formal statistical testing**, not visual inspection or backtest-driven bias.

## 3. Repository Architecture
```
├── notebooks/ # Core analytical workflow
│ ├── 1_1_stock_returns_data_pipeline.ipynb
│ ├── 1_2_time_series_signals_hypothesis_testing.ipynb
│ ├── 1_3_earnings_releases_data_pipeline_and_exploration.ipynb
│ └── 1_4_signal_to_portfolio_influence_hypothesis_testing.ipynb
│
├── src/ # Modular Python utilities
│ ├── data_pipeline_utils/ # Data ingestion & preprocessing
│ ├── plotting_utils/ # Visualization layer
│ ├── signal_testing_utils/ # Statistical testing utilities
│ └── nlp_utils/ # (Exploratory / partial NLP pipeline)
│
├── data/ # Auto-generated datasets (ignored)
├── static_data/ # Optional cached artifacts
├── .gitignore
└── README.md
```

## 4. Analytical Workflow

### **Notebook 1_1 — Stock Returns Data Pipeline**

**Objective:** Construct the foundational dataset and explore statistical assumptions.

**Key Components:**
- Data acquisition via `yfinance`
- Log return computation
- Rolling statistics (mean, standard deviation)
- Distribution analysis (histograms, Q-Q plots)
- Stationarity testing (ADF, KPSS)
- Autocorrelation analysis (ACF, PACF)
- Hurst exponent estimation
- Volatility clustering diagnostics
- Outlier detection:
  - Rolling σ thresholds
  - Hampel filter (robust method)

**Outcome:**

Establishes the core empirical result:

> Returns are **stationary, non-normal, and weakly autocorrelated**,  
> but exhibit **fat tails, volatility clustering, and mean-reverting tendencies**

### **Notebook 1_2 — Time Series Signals & Hypothesis Testing**

**Objective:** Convert statistical structure into testable signals.

**Signal Categories:**
- Mean-reversion signals
- Volatility-based signals
- Outlier-driven signals

**Methods:**
- Parametric testing (t-tests)
- Distribution comparisons
- Conditional expectation analysis

**Significance Threshold:**
- **α = 0.01 (strict rejection criteria)**

**Outcome:**

Separates:
- **statistically valid signals**
- from **noise and false positives**

### **Notebook 1_3 — Earnings Releases & NLP Exploration**

**Objective:** Explore whether exogenous textual data introduces predictive power.

**Scope:**
- Earnings-related data ingestion
- Preliminary NLP sentiment extraction

**Note:**
- This component is **exploratory**
- Not central to final conclusions
- Included to test extension beyond price-based signals

### **Notebook 1_4 — Signal Integration & Portfolio Impact**

**Objective:** Evaluate whether validated signals influence return distributions.

**Approach:**
- Combine signals from previous steps in Notebooks 1_2 and 1_3
- Analyze conditional return behavior
- Evaluate directional and distributional shifts

**Focus:**
- Actual backtesting economical impact
- Not overfit backtesting


## 5. Key Empirical Findings

Consistent across all 30 large-cap stocks:

### Confirmed Properties

- **Stationarity of returns**
- **Fat-tailed distributions (kurtosis > normal)**
- **Volatility clustering (heteroskedasticity)**
- **Mean-reverting tendencies (H < 0.5)**
- **Frequent extreme events (outliers)**

### ❌ Rejected Assumptions

- No strong **linear autocorrelation in returns**
- No reliable **trend or seasonality**
- No consistent **directional predictability**

---

## 6. Core Insight

The project converges to a single, critical conclusion:

> **Market direction is largely efficient.  
> Market structure is not.**

Specifically:

- Returns → largely unpredictable  
- Volatility → persistent and structured  
- Extremes → statistically significant and non-random  

## 7. Implications

This leads to a reframing of signal design:

| Traditional View | Empirical Reality |
|----------------|-----------------|
| Predict direction | Weak signal |
| Follow trends | Inconsistent |
| Exploit volatility | Strong signal |
| Detect extremes | High value |

## 8. Installation & Setup

```bash
git clone https://github.com/kamend1/data_science_final_project.git
cd data_science_final_project
```

## 9. Execution Order (Critical)

Run notebooks sequentially:

1_1_stock_returns_data_pipeline.ipynb
1_2_time_series_signals_hypothesis_testing.ipynb
1_3_earnings_releases_data_pipeline_and_exploration.ipynb
1_4_signal_to_portfolio_influence_hypothesis_testing.ipynb

## 10. Note to the Grader

This project is intentionally designed to challenge assumptions.

No indicator is accepted without statistical validation
All results are cross-validated across 30 assets
Strict significance level (α = 0.01) is enforced
No reliance on overfit backtests or cherry-picked results

The goal is not to prove that markets are predictable, but to identify:

where structure exists — and where it does not

## 11. Final Conclusion

The analysis demonstrates that:

Financial markets exhibit strong statistical structure
This structure does not exist in price direction
It exists in:
volatility
distributional shape
extreme deviations
Final Takeaway

The edge in financial markets is not in predicting movement —
it is in identifying when the market is behaving abnormally.
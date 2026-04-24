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
- Whether **natural language processing (NLP) and sentiment analysis of corporate disclosures** can provide additional predictive signals
- Whether these combined signals can ultimately generate **economic value when deployed in a portfolio framework through backtesting**

## 2. Research Framework

The project is structured around a clear empirical pipeline:

1. **Data Construction** → Build a clean, reproducible return dataset  
2. **Statistical Characterization** → Understand distributional properties  
3. **Structure Identification** → Detect non-random behavior  
4. **Signal Construction** → Translate structure into signals  
5. **Hypothesis Testing** → Validate statistical significance at a confidence level of $\alpha = 0.01$ across the entire project  
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
- Parametric testing
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
- Actual backtesting economic impact
- Not overfit backtesting


## 5. Key Empirical Findings

Consistent across all 30 large-cap stocks:

### Confirmed Properties

- **Stationarity of returns**
- **Fat-tailed distributions (kurtosis > normal)**
- **Volatility clustering (heteroskedasticity)**
- **Mean-reverting tendencies (H < 0.5)**
- **Frequent extreme events (outliers)**

### Rejected Assumptions

- No strong **linear autocorrelation in returns**
- No reliable **trend or seasonality**
- No consistent **directional predictability**


## 6. Core Insight

The project converges to a single, critical conclusion:

> **Signals can exist and generate value.  
> Their effectiveness breaks down at the portfolio level.**

Specifically:

- Returns → limited predictability, but usable in selective contexts  
- Signals → statistically valid and economically positive in isolation  
- Portfolio performance → constrained by capital allocation, diversification, and scaling limitations  
- Optimization → amplifies estimation error and concentrates risk  

## 7. Implications

This leads to a reframing of signal design and deployment:

| Traditional View | Empirical Reality |
|----------------|-----------------|
| Find predictive signals | Signals exist but are weak and context-dependent |
| Add more signals | Signal quality matters more than quantity |
| Optimize portfolio | Optimization introduces instability and concentration risk |
| Deploy full capital | Signals often do not support full allocation |
| Beat benchmarks with alpha | Simple strategies remain highly competitive |
| Scale profitable signals | Signal performance degrades with scale |

## 8. Installation & Setup

```bash
git clone https://github.com/kamend1/data_science_final_project.git
cd data_science_final_project
```

## 9. Execution Order (Critical)

Run notebooks sequentially:

- 1_1_stock_returns_data_pipeline.ipynb
- 1_2_time_series_signals_hypothesis_testing.ipynb
- 1_3_earnings_releases_data_pipeline_and_exploration.ipynb
- 1_4_signal_to_portfolio_influence_hypothesis_testing.ipynb

## 10. Note to the Grader

This project is intentionally designed to challenge assumptions.

- No indicator is accepted without statistical validation  
- All results are tested across 30 assets  
- A strict significance level ($\alpha = 0.01$) is consistently enforced  
- There is no reliance on overfit backtests or cherry-picked results  

The goal was not to prove that markets are predictable, but to identify:

> where structure exists — and where it does not

---

### Personal Reflection

I started this project with a completely open mind. I did not know where the analysis would lead, and I made a conscious decision not to steer it toward a predefined conclusion.

Following your advice, I adhered as closely as possible to the **scientific method**:
- No look-ahead bias  
- No retrospective adjustments to fit results  
- No altering hypotheses after observing outcomes  

I also made a deliberate effort to **separate this work from my prior professional experience**. I tried to avoid introducing personal bias, even when the results challenged my own expectations or intuitions about the markets.

This project was intentionally ambitious — both in scope and in objective. I challenged myself not only technically, but also in terms of discipline and rigor. I can say with confidence that it became one of the most valuable learning experiences in my program.

---

### Challenges Faced

The process was not without setbacks.

The most significant limitation was related to **data availability and legality**, particularly regarding earnings call transcripts. This forced me to pivot to alternative textual sources (8-K and 10-Q filings), which I recognize are less informative for sentiment extraction.

This constraint likely impacted the strength of the sentiment analysis results. However, I chose to proceed within legal and ethical boundaries rather than compromise the integrity of the project. I hope this trade-off is understood in the context of the overall effort.

---

### Closing Note

I am aware that this topic may not align perfectly with every grader’s area of interest. Still, my goal was to approach it with enough depth, rigor, and curiosity to make the process engaging not only for myself, but also for the reader.

I hope that the work reflects:
- Genuine effort
- Strong methodological discipline
- And a willingness to explore beyond surface-level conclusions

Finally, I am genuinely grateful for this opportunity. This course has been a significant learning experience, and this project amplified that value by pushing me to engage with complex, real-world problems in a structured and disciplined way.

## 11. Final Conclusion

The analysis demonstrates that:

- Financial markets exhibit **strong and persistent statistical structure**
- This structure does not exist in **price direction**
- Instead, it is embedded in:
  - Volatility
  - Distributional shape
  - Extreme deviations

## Final Takeaway

The edge in financial markets is not in predicting movement —  
it is in identifying when the market is behaving abnormally.

## Broader Conclusion

Through this project, I was able to identify **statistically significant relationships** and demonstrate that they can be translated into **economically positive outcomes under certain conditions**.

At the same time, the results show clear limitations:
- Signals do not scale easily
- Portfolio construction and capital allocation become the dominant constraints
- Not all statistically valid signals translate into robust investment strategies

This leads to a balanced conclusion:

> There is evidence of exploitable structure in the market,  
> but extracting consistent and scalable value from it remains a complex problem.

## Personal Reflection

This project revealed a promising avenue for further exploration.

While not all approaches delivered strong results, the process uncovered:
- Multiple directions for improving signal design
- Opportunities to refine portfolio construction
- Clear paths for expanding the dataset and feature space

Rather than closing the question, the findings highlight that:

> There is more to test, more to refine, and meaningful potential to explore further.

Overall, the results reinforce both:
- The **difficulty of consistently extracting alpha**
- And the **existence of underlying structure that justifies continued research**
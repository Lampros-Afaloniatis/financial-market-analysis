# Financial Market Analysis — Risk, Returns & Portfolio Performance

## Overview

This project analyzes the risk and return characteristics of four financial assets and investigates how diversification affects portfolio performance.

The assets analyzed are:

- S&P 500 — SPY
- Nasdaq 100 — QQQ
- Gold — GLD
- US Treasury Bonds — TLT

The analysis uses approximately 10 years of historical market data from Yahoo Finance and the 3-month US Treasury rate (DGS3MO) from FRED.

## Research Question

> How do return and risk differ across financial assets, and how does diversification affect portfolio performance?

## Methodology

The project includes:

1. Historical market data collection
2. Data cleaning and alignment
3. Daily return calculation
4. CAGR and annualized volatility
5. Sharpe and Sortino ratios
6. Maximum drawdown analysis
7. Correlation and covariance analysis
8. Equal-weight portfolio analysis
9. Monte Carlo simulation of 10,000 portfolios
10. Maximum Sharpe portfolio
11. Minimum variance portfolio
12. Approximate efficient frontier
13. Chronological 80/20 train-test split
14. Out-of-sample portfolio evaluation
15. Benchmark comparison

The Monte Carlo simulation generates non-negative portfolio weights that sum to 100%.

## Key Findings

The correlation analysis showed a very high correlation between the S&P 500 and Nasdaq 100 (approximately 0.93), while Gold had a much lower correlation with equities. This suggests that Gold can provide meaningful diversification benefits when combined with equity assets.

The historical maximum-Sharpe portfolio was approximately:

- S&P 500: 1%
- Nasdaq 100: 48%
- Gold: 49%
- US Treasury Bonds: 1%

with a Sharpe ratio of approximately 0.99.

For the out-of-sample test, portfolio weights were selected using training data only and then kept fixed during the test period:

- S&P 500: 2.93%
- Nasdaq 100: 44.01%
- Gold: 52.12%
- US Treasury Bonds: 0.94%

### Out-of-Sample Results

| Metric | Optimized Portfolio |
|---|---:|
| CAGR | 24.76% |
| Volatility | 17.56% |
| Sharpe Ratio | 1.11 |
| Sortino Ratio | 1.58 |
| Maximum Drawdown | -14.01% |
| Initial Investment | €1,000 |
| Final Value | €1,549.79 |

The optimized portfolio did not produce the highest absolute CAGR in the test period. Gold alone achieved a slightly higher CAGR. However, the optimized portfolio achieved a better risk-adjusted performance, with lower volatility and a smaller maximum drawdown.

## Visualizations

The project produces:

- Growth of €1,000 across assets
- Risk vs. return comparison
- Correlation heatmap
- Monte Carlo portfolio simulation
- Approximate efficient frontier
- Sharpe ratio distribution
- Out-of-sample growth comparison
- Out-of-sample drawdown comparison
- Sharpe ratio comparison by asset

## Important Limitation

The results are historical and should not be interpreted as a prediction of future returns.

The portfolio optimization is based on historical data and a specific train-test split. Further robustness analysis could include walk-forward testing, transaction costs, turnover constraints, alternative optimization objectives, and multiple market regimes.

## Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Requests
- yfinance
- python-dotenv
- FRED API
- Yahoo Finance

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Create a `.env` file

Add your own FRED API key:

```text
FRED_API_KEY=YOUR_FRED_API_KEY
```

Do not upload the `.env` file or expose your API key publicly.

### 3. Run the analysis

```bash
python anapysis.py
```

The script prints the main results and displays the visualizations.

## Project Structure

```text
financial-market-analysis/
│
├── anapysis.py
├── README.md
├── requirements.txt
└── .gitignore
```

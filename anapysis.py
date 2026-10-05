# ============================================================
# FINANCIAL MARKET ANALYSIS
# Risk, Returns & Portfolio Performance
# ============================================================

import sys
import os
import numpy as np
import pandas as pd
import requests
import yfinance as yf
import matplotlib.pyplot as plt
from dotenv import load_dotenv


# ============================================================
# 1. SETUP
# ============================================================

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

load_dotenv()

api_key = os.getenv("FRED_API_KEY")

if not api_key:
    raise ValueError(
        "FRED_API_KEY was not found. Check your .env file."
    )

TRADING_DAYS = 252
INITIAL_INVESTMENT = 1000

ASSETS = {
    "S&P 500": "SPY",
    "Nasdaq 100": "QQQ",
    "Gold": "GLD",
    "US Treasury Bonds": "TLT",
}


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def calculate_cagr(returns):
    cumulative = (1 + returns).cumprod()
    years = len(returns) / TRADING_DAYS

    if years <= 0:
        return np.nan

    return cumulative.iloc[-1] ** (1 / years) - 1


def calculate_volatility(returns):
    return returns.std() * np.sqrt(TRADING_DAYS)


def calculate_sharpe(returns, risk_free_daily):
    excess_returns = returns - risk_free_daily
    volatility = returns.std()

    if volatility == 0:
        return np.nan

    return (
        excess_returns.mean()
        / volatility
    ) * np.sqrt(TRADING_DAYS)


def calculate_sortino(returns, risk_free_daily):
    excess_returns = returns - risk_free_daily
    downside = excess_returns.clip(upper=0)

    downside_deviation = np.sqrt(
        (downside ** 2).mean()
    )

    if downside_deviation == 0:
        return np.nan

    return (
        excess_returns.mean()
        / downside_deviation
    ) * np.sqrt(TRADING_DAYS)


def calculate_drawdown_series(returns):
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()

    return (
        cumulative - running_max
    ) / running_max


def calculate_max_drawdown(returns):
    return calculate_drawdown_series(returns).min()


def portfolio_returns(returns_df, weights):
    return returns_df @ weights


# ============================================================
# 3. DATA COLLECTION
# ============================================================

print("\n" + "=" * 70)
print("DATA COLLECTION")
print("=" * 70)

# ------------------------------------------------------------
# 3.1 Risk-free rate from FRED
# ------------------------------------------------------------

fred_url = (
    "https://api.stlouisfed.org/"
    "fred/series/observations"
)

risk_free_params = {
    "series_id": "DGS3MO",
    "api_key": api_key,
    "file_type": "json",
}

response = requests.get(
    fred_url,
    params=risk_free_params,
    timeout=30
)

response.raise_for_status()

risk_free_data = response.json()

risk_free_df = pd.DataFrame(
    risk_free_data["observations"]
)

risk_free_df["value"] = pd.to_numeric(
    risk_free_df["value"],
    errors="coerce"
)

risk_free_df = risk_free_df.dropna(
    subset=["value"]
)

risk_free_df["date"] = pd.to_datetime(
    risk_free_df["date"]
)

risk_free_df = risk_free_df.rename(
    columns={"value": "risk_free_rate"}
)

risk_free_df["risk_free_daily"] = (
    risk_free_df["risk_free_rate"]
    / 100
    / TRADING_DAYS
)

risk_free_df = risk_free_df[
    ["date", "risk_free_rate", "risk_free_daily"]
]


# ------------------------------------------------------------
# 3.2 Market assets from Yahoo Finance
# ------------------------------------------------------------

price_data = {}

for asset_name, ticker in ASSETS.items():

    print(
        f"Downloading {asset_name} ({ticker})..."
    )

    data = yf.download(
        ticker,
        period="10y",
        auto_adjust=True,
        progress=False
    )

    if data.empty:
        raise ValueError(
            f"No data downloaded for {ticker}."
        )

    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"][ticker]
    else:
        close = data["Close"]

    close = close.rename(asset_name)

    close.index = pd.to_datetime(
        close.index
    )

    if close.index.tz is not None:
        close.index = close.index.tz_localize(
            None
        )

    price_data[asset_name] = close


prices = pd.concat(
    price_data.values(),
    axis=1
)

prices = prices.sort_index()
prices = prices.dropna()
prices.index.name = "date"


# ============================================================
# 4. RETURNS & DATA CLEANING
# ============================================================

print("\n" + "=" * 70)
print("RETURNS & DATA CLEANING")
print("=" * 70)

asset_returns = prices.pct_change().dropna()

returns = asset_returns.merge(
    risk_free_df,
    left_index=True,
    right_on="date",
    how="inner"
)

returns = returns.set_index("date")
returns = returns.dropna()

asset_returns = returns[
    list(ASSETS.keys())
]

risk_free_daily = returns[
    "risk_free_daily"
]

print(
    "\nObservations:",
    len(asset_returns)
)

print(
    "\nDate range:",
    asset_returns.index.min().date(),
    "to",
    asset_returns.index.max().date()
)


# ============================================================
# 5. INDIVIDUAL ASSET PERFORMANCE
# ============================================================

print("\n" + "=" * 70)
print("INDIVIDUAL ASSET PERFORMANCE")
print("=" * 70)

performance = []

for asset in asset_returns.columns:

    r = asset_returns[asset]

    performance.append({
        "Asset": asset,
        "CAGR": calculate_cagr(r),
        "Volatility": calculate_volatility(r),
        "Sharpe": calculate_sharpe(
            r,
            risk_free_daily
        ),
        "Sortino": calculate_sortino(
            r,
            risk_free_daily
        ),
        "Max Drawdown": calculate_max_drawdown(r),
    })


performance_df = pd.DataFrame(performance)

print(
    performance_df.to_string(
        index=False,
        formatters={
            "CAGR": "{:.2%}".format,
            "Volatility": "{:.2%}".format,
            "Sharpe": "{:.2f}".format,
            "Sortino": "{:.2f}".format,
            "Max Drawdown": "{:.2%}".format,
        }
    )
)


# ============================================================
# 6. CORRELATION & COVARIANCE
# ============================================================

print("\n" + "=" * 70)
print("CORRELATION & COVARIANCE")
print("=" * 70)

correlation_matrix = asset_returns.corr()
covariance_matrix = asset_returns.cov()

print("\nCorrelation:")
print(correlation_matrix.round(3))

print("\nCovariance:")
print(covariance_matrix.round(6))


# ============================================================
# 7. EQUAL-WEIGHT PORTFOLIO
# ============================================================

print("\n" + "=" * 70)
print("EQUAL-WEIGHT PORTFOLIO")
print("=" * 70)

n_assets = len(ASSETS)

equal_weights = np.ones(n_assets) / n_assets

equal_returns = portfolio_returns(
    asset_returns,
    equal_weights
)

print("\nWeights:")

for asset, weight in zip(
    asset_returns.columns,
    equal_weights
):
    print(f"{asset}: {weight:.2%}")

print(
    f"\nCAGR: "
    f"{calculate_cagr(equal_returns):.2%}"
)

print(
    f"Volatility: "
    f"{calculate_volatility(equal_returns):.2%}"
)

print(
    f"Sharpe: "
    f"{calculate_sharpe(equal_returns, risk_free_daily):.2f}"
)

print(
    f"Max Drawdown: "
    f"{calculate_max_drawdown(equal_returns):.2%}"
)


# ============================================================
# 8. MONTE CARLO PORTFOLIO SIMULATION
# ============================================================

print("\n" + "=" * 70)
print("MONTE CARLO PORTFOLIO SIMULATION")
print("=" * 70)

rng = np.random.default_rng(42)

n_simulations = 10000

random_weights = rng.dirichlet(
    np.ones(n_assets),
    size=n_simulations
)

portfolio_results = []

for weights in random_weights:

    p_returns = portfolio_returns(
        asset_returns,
        weights
    )

    portfolio_results.append({
        "return": p_returns.mean() * TRADING_DAYS,
        "volatility": calculate_volatility(
            p_returns
        ),
        "sharpe": calculate_sharpe(
            p_returns,
            risk_free_daily
        ),
    })


portfolio_results = pd.DataFrame(
    portfolio_results
)

for i, asset in enumerate(
    asset_returns.columns
):
    portfolio_results[asset] = (
        random_weights[:, i]
    )


# Maximum Sharpe
max_sharpe_index = (
    portfolio_results["sharpe"].idxmax()
)

max_sharpe_portfolio = (
    portfolio_results.loc[
        max_sharpe_index
    ]
)


# Minimum variance
min_variance_index = (
    portfolio_results["volatility"].idxmin()
)

min_variance_portfolio = (
    portfolio_results.loc[
        min_variance_index
    ]
)


print("\nMaximum Sharpe Portfolio:")

for asset in asset_returns.columns:
    print(
        f"{asset}: "
        f"{max_sharpe_portfolio[asset]:.2%}"
    )

print(
    f"Return: "
    f"{max_sharpe_portfolio['return']:.2%}"
)

print(
    f"Volatility: "
    f"{max_sharpe_portfolio['volatility']:.2%}"
)

print(
    f"Sharpe: "
    f"{max_sharpe_portfolio['sharpe']:.2f}"
)


print("\nMinimum Variance Portfolio:")

for asset in asset_returns.columns:
    print(
        f"{asset}: "
        f"{min_variance_portfolio[asset]:.2%}"
    )

print(
    f"Return: "
    f"{min_variance_portfolio['return']:.2%}"
)

print(
    f"Volatility: "
    f"{min_variance_portfolio['volatility']:.2%}"
)

print(
    f"Sharpe: "
    f"{min_variance_portfolio['sharpe']:.2f}"
)


# ============================================================
# 9. APPROXIMATE EFFICIENT FRONTIER
# ============================================================

print("\n" + "=" * 70)
print("EFFICIENT FRONTIER")
print("=" * 70)

return_bins = np.linspace(
    portfolio_results["return"].min(),
    portfolio_results["return"].max(),
    80
)

frontier_points = []

for i in range(
    len(return_bins) - 1
):

    lower = return_bins[i]
    upper = return_bins[i + 1]

    subset = portfolio_results[
        (portfolio_results["return"] >= lower)
        & (portfolio_results["return"] < upper)
    ]

    if not subset.empty:

        best = subset.loc[
            subset["volatility"].idxmin()
        ]

        frontier_points.append(best)


efficient_frontier = pd.DataFrame(
    frontier_points
)


# ============================================================
# 10. OUT-OF-SAMPLE TESTING
# ============================================================

print("\n" + "=" * 70)
print("OUT-OF-SAMPLE TESTING")
print("=" * 70)

# Chronological 80/20 split.
split_index = int(
    len(asset_returns) * 0.80
)

train_returns = asset_returns.iloc[
    :split_index
].copy()

test_returns = asset_returns.iloc[
    split_index:
].copy()

train_risk_free = risk_free_daily.loc[
    train_returns.index
]

test_risk_free = risk_free_daily.loc[
    test_returns.index
]

print(
    "\nTraining:",
    train_returns.index.min().date(),
    "to",
    train_returns.index.max().date()
)

print(
    "Testing:",
    test_returns.index.min().date(),
    "to",
    test_returns.index.max().date()
)


# ------------------------------------------------------------
# 10.1 Optimize using training data only
# ------------------------------------------------------------

train_results = []

for weights in random_weights:

    p_returns = portfolio_returns(
        train_returns,
        weights
    )

    train_results.append({
        "sharpe": calculate_sharpe(
            p_returns,
            train_risk_free
        ),
        "volatility": calculate_volatility(
            p_returns
        ),
        "return": p_returns.mean()
        * TRADING_DAYS,
    })


train_results = pd.DataFrame(
    train_results
)

best_train_index = (
    train_results["sharpe"].idxmax()
)

best_train_weights = (
    random_weights[best_train_index]
)

print(
    "\nWeights selected using training data:"
)

for asset, weight in zip(
    asset_returns.columns,
    best_train_weights
):
    print(f"{asset}: {weight:.2%}")


# ------------------------------------------------------------
# 10.2 Test frozen weights on unseen data
# ------------------------------------------------------------

test_portfolio_returns = (
    portfolio_returns(
        test_returns,
        best_train_weights
    )
)

test_cagr = calculate_cagr(
    test_portfolio_returns
)

test_volatility = calculate_volatility(
    test_portfolio_returns
)

test_sharpe = calculate_sharpe(
    test_portfolio_returns,
    test_risk_free
)

test_sortino = calculate_sortino(
    test_portfolio_returns,
    test_risk_free
)

test_drawdown = calculate_max_drawdown(
    test_portfolio_returns
)

test_final_value = (
    INITIAL_INVESTMENT
    * (1 + test_portfolio_returns).prod()
)

print("\nOUT-OF-SAMPLE RESULTS")
print(f"CAGR: {test_cagr:.2%}")
print(f"Volatility: {test_volatility:.2%}")
print(f"Sharpe: {test_sharpe:.2f}")
print(f"Sortino: {test_sortino:.2f}")
print(f"Max Drawdown: {test_drawdown:.2%}")
print(
    f"€1,000 became: "
    f"€{test_final_value:.2f}"
)


# ============================================================
# 11. BENCHMARKS — OUT-OF-SAMPLE
# ============================================================

benchmark_results = []

for asset in asset_returns.columns:

    r = test_returns[asset]

    benchmark_results.append({
        "Asset": asset,
        "CAGR": calculate_cagr(r),
        "Volatility": calculate_volatility(r),
        "Sharpe": calculate_sharpe(
            r,
            test_risk_free
        ),
        "Max Drawdown": calculate_max_drawdown(r),
    })


benchmark_df = pd.DataFrame(
    benchmark_results
)

print("\nBenchmark performance:")
print(
    benchmark_df.to_string(
        index=False,
        formatters={
            "CAGR": "{:.2%}".format,
            "Volatility": "{:.2%}".format,
            "Sharpe": "{:.2f}".format,
            "Max Drawdown": "{:.2%}".format,
        }
    )
)


# ============================================================
# 12. VISUALIZATIONS
# ============================================================

print("\n" + "=" * 70)
print("CREATING GRAPHS")
print("=" * 70)


# ------------------------------------------------------------
# GRAPH 1 — Growth of €1,000
# ------------------------------------------------------------

cumulative_values = (
    INITIAL_INVESTMENT
    * (1 + asset_returns).cumprod()
)

plt.figure(figsize=(12, 7))

for asset in cumulative_values.columns:

    plt.plot(
        cumulative_values.index,
        cumulative_values[asset],
        linewidth=2,
        label=asset
    )

plt.title(
    "Growth of €1,000 — Financial Assets",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Date")
plt.ylabel("Portfolio Value (€)")

plt.legend()
plt.grid(alpha=0.25)

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# GRAPH 2 — Risk vs Return
# ------------------------------------------------------------

plt.figure(figsize=(11, 7))

plt.scatter(
    performance_df["Volatility"],
    performance_df["CAGR"],
    s=140
)

for _, row in performance_df.iterrows():

    plt.annotate(
        row["Asset"],
        (
            row["Volatility"],
            row["CAGR"]
        ),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=10
    )

plt.title(
    "Risk vs Return — Individual Assets",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Annualized Volatility")
plt.ylabel("CAGR")

plt.grid(alpha=0.25)

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# GRAPH 3 — Correlation Heatmap
# ------------------------------------------------------------

plt.figure(figsize=(9, 7))

heatmap = plt.imshow(
    correlation_matrix,
    vmin=-1,
    vmax=1,
    aspect="auto"
)

plt.colorbar(
    heatmap,
    label="Correlation"
)

plt.xticks(
    range(len(correlation_matrix.columns)),
    correlation_matrix.columns,
    rotation=30,
    ha="right"
)

plt.yticks(
    range(len(correlation_matrix.index)),
    correlation_matrix.index
)

for i in range(
    len(correlation_matrix)
):
    for j in range(
        len(correlation_matrix.columns)
    ):

        plt.text(
            j,
            i,
            f"{correlation_matrix.iloc[i, j]:.2f}",
            ha="center",
            va="center"
        )

plt.title(
    "Correlation Matrix — Daily Returns",
    fontsize=16,
    fontweight="bold"
)

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# GRAPH 4 — Monte Carlo + Efficient Frontier
# ------------------------------------------------------------

plt.figure(figsize=(12, 8))

scatter = plt.scatter(
    portfolio_results["volatility"],
    portfolio_results["return"],
    c=portfolio_results["sharpe"],
    s=10,
    alpha=0.35
)

plt.colorbar(
    scatter,
    label="Sharpe Ratio"
)

if not efficient_frontier.empty:

    plt.plot(
        efficient_frontier["volatility"],
        efficient_frontier["return"],
        linewidth=3,
        label="Approx. Efficient Frontier"
    )

plt.scatter(
    max_sharpe_portfolio["volatility"],
    max_sharpe_portfolio["return"],
    s=220,
    marker="*",
    label="Maximum Sharpe"
)

plt.scatter(
    min_variance_portfolio["volatility"],
    min_variance_portfolio["return"],
    s=170,
    marker="X",
    label="Minimum Variance"
)

plt.title(
    "Monte Carlo Portfolio Simulation & Efficient Frontier",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Annualized Volatility")
plt.ylabel("Annualized Return")

plt.legend()
plt.grid(alpha=0.2)

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# GRAPH 5 — Sharpe Ratio Distribution
# ------------------------------------------------------------

plt.figure(figsize=(11, 6))

plt.hist(
    portfolio_results["sharpe"],
    bins=60,
    alpha=0.8
)

plt.axvline(
    max_sharpe_portfolio["sharpe"],
    linestyle="--",
    linewidth=2,
    label=(
        "Maximum Sharpe = "
        f"{max_sharpe_portfolio['sharpe']:.2f}"
    )
)

plt.title(
    "Distribution of Portfolio Sharpe Ratios",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Sharpe Ratio")
plt.ylabel("Number of Portfolios")

plt.legend()
plt.grid(alpha=0.2)

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# GRAPH 6 — Out-of-Sample Growth of €1,000
# ------------------------------------------------------------

test_portfolio_value = (
    INITIAL_INVESTMENT
    * (1 + test_portfolio_returns).cumprod()
)

test_asset_values = (
    INITIAL_INVESTMENT
    * (1 + test_returns).cumprod()
)

plt.figure(figsize=(12, 7))

plt.plot(
    test_portfolio_value.index,
    test_portfolio_value,
    linewidth=3,
    label="Optimized Portfolio"
)

for asset in test_asset_values.columns:

    plt.plot(
        test_asset_values.index,
        test_asset_values[asset],
        linewidth=1.8,
        label=asset
    )

plt.title(
    "Out-of-Sample Performance — Growth of €1,000",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Date")
plt.ylabel("Portfolio Value (€)")

plt.legend()
plt.grid(alpha=0.25)

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# GRAPH 7 — Out-of-Sample Drawdowns
# ------------------------------------------------------------

plt.figure(figsize=(12, 7))

portfolio_dd = calculate_drawdown_series(
    test_portfolio_returns
)

plt.plot(
    portfolio_dd.index,
    portfolio_dd,
    linewidth=3,
    label="Optimized Portfolio"
)

for asset in test_returns.columns:

    asset_dd = calculate_drawdown_series(
        test_returns[asset]
    )

    plt.plot(
        asset_dd.index,
        asset_dd,
        linewidth=1.5,
        label=asset
    )

plt.axhline(
    0,
    linewidth=1
)

plt.title(
    "Out-of-Sample Drawdown Comparison",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Date")
plt.ylabel("Drawdown")

plt.legend()
plt.grid(alpha=0.25)

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# GRAPH 8 — Sharpe Ratio by Asset
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.bar(
    performance_df["Asset"],
    performance_df["Sharpe"]
)

plt.title(
    "Sharpe Ratio by Asset",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Asset")
plt.ylabel("Sharpe Ratio")

plt.xticks(rotation=20)

plt.grid(
    axis="y",
    alpha=0.25
)

plt.tight_layout()
plt.show()


# ============================================================
# 13. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print("\nIndividual assets:")
print(
    performance_df.to_string(
        index=False,
        formatters={
            "CAGR": "{:.2%}".format,
            "Volatility": "{:.2%}".format,
            "Sharpe": "{:.2f}".format,
            "Sortino": "{:.2f}".format,
            "Max Drawdown": "{:.2%}".format,
        }
    )
)

print("\nMaximum Sharpe Portfolio:")

for asset in asset_returns.columns:
    print(
        f"{asset}: "
        f"{max_sharpe_portfolio[asset]:.2%}"
    )

print(
    f"Sharpe: "
    f"{max_sharpe_portfolio['sharpe']:.2f}"
)

print("\nMinimum Variance Portfolio:")

for asset in asset_returns.columns:
    print(
        f"{asset}: "
        f"{min_variance_portfolio[asset]:.2%}"
    )

print(
    f"Volatility: "
    f"{min_variance_portfolio['volatility']:.2%}"
)

print("\nOut-of-Sample Portfolio:")

for asset, weight in zip(
    asset_returns.columns,
    best_train_weights
):
    print(f"{asset}: {weight:.2%}")

print(f"CAGR: {test_cagr:.2%}")
print(f"Volatility: {test_volatility:.2%}")
print(f"Sharpe: {test_sharpe:.2f}")
print(f"Sortino: {test_sortino:.2f}")
print(f"Max Drawdown: {test_drawdown:.2%}")
print(f"€1,000 became: €{test_final_value:.2f}")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)

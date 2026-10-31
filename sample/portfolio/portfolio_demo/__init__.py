import numpy as np
import pandas as pd
from dataclasses import dataclass

__all__ = ["Portfolio", "list_portfolios", "load_portfolio"]


@dataclass
class Portfolio:
    """Hypothetical target weights and synthetic price history, not actual investment results."""

    holdings: pd.DataFrame
    prices: pd.DataFrame
    benchmark: pd.Series
    metadata: dict[str, str]


def list_portfolios() -> list[str]:
    """Return the available demo portfolio names."""
    return ["balanced_equity"]


def load_portfolio(name: str = "balanced_equity") -> Portfolio:
    """Load independent, deterministic demo data; all companies and prices are fictional."""
    if name not in list_portfolios():
        raise ValueError(f"Unknown portfolio {name!r}; choose from {list_portfolios()}")

    holdings = pd.DataFrame(
        [
            ("NOVA", "Nova Systems", "Technology", 0.15),
            ("CLOUD", "Cloud Works", "Technology", 0.15),
            ("BANK", "Union Finance", "Financials", 0.15),
            ("CARE", "Care Health", "Healthcare", 0.15),
            ("SHOP", "Market Retail", "Consumer Discretionary", 0.10),
            ("GRID", "Grid Power", "Utilities", 0.10),
            ("OIL", "Horizon Energy", "Energy", 0.10),
            ("GOODS", "Everyday Goods", "Consumer Staples", 0.10),
        ],
        columns=["symbol", "company", "sector", "weight"],
    )
    dates = pd.bdate_range("2018-01-01", "2025-12-31", name="date")
    rng = np.random.default_rng(42)
    market = rng.normal(0.0003, 0.01, len(dates))
    downturn = (dates >= "2020-02-20") & (dates <= "2020-03-23")
    market[downturn] -= 0.012
    technology = rng.normal(0, 0.008, len(dates))
    beta = np.array([1.2, 1.3, 1.1, 0.8, 1.1, 0.5, 1.0, 0.6])
    volatility = np.array([0.012, 0.016, 0.009, 0.009, 0.012, 0.006, 0.016, 0.006])
    log_returns = market[:, None] * beta + rng.normal(size=(len(dates), len(holdings))) * volatility
    log_returns[:, :2] += technology[:, None]
    log_returns[0] = 0
    prices = pd.DataFrame(100 * np.exp(log_returns.cumsum(axis=0)), index=dates, columns=holdings["symbol"].tolist())
    benchmark_returns = market.copy()
    benchmark_returns[0] = 0
    benchmark = pd.Series(100 * np.exp(benchmark_returns.cumsum()), index=dates, name="Synthetic market benchmark")

    return Portfolio(
        holdings=holdings,
        prices=prices,
        benchmark=benchmark,
        metadata={
            "source": "Seeded synthetic data; fictional companies, prices, and benchmark. Not real market history.",
            "currency": "USD",
            "start_date": "2018-01-01",
            "end_date": "2025-12-31",
            "weights_as_of": "2018-01-01",
            "weights": "Hypothetical starting target weights, not recorded historical holdings.",
            "prices": "Simulated total-return price levels starting at 100; no separate dividends or corporate actions.",
            "calendar": "Monday to Friday, including exchange holidays; not an exchange trading calendar.",
            "backtest": "No rebalancing, fees, taxes, or inflation are applied by this data library.",
        },
    )

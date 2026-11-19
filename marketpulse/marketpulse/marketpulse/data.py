"""Data loading: prices (yfinance, cached to CSV) and recent news headlines."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

CACHE_DIR = Path(".cache")


def load_prices(ticker: str, start: str = "2015-01-01", end: str | None = None,
                use_cache: bool = True) -> pd.DataFrame:
    """Daily OHLCV for a ticker, indexed by date. Cached on disk to avoid refetching."""
    import yfinance as yf

    CACHE_DIR.mkdir(exist_ok=True)
    path = CACHE_DIR / f"{ticker.upper()}_{start}_{end or 'latest'}.csv"
    if use_cache and path.exists():
        return pd.read_csv(path, index_col=0, parse_dates=True)

    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No price data returned for '{ticker}'.")
    if isinstance(df.columns, pd.MultiIndex):  # newer yfinance returns multi-level columns
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.index.name = "Date"
    df.to_csv(path)
    return df


def load_news(ticker: str, limit: int = 20) -> list[str]:
    """Recent headlines from yfinance. Only recent items are available, so this is
    used for a live sentiment panel and NOT as a training feature (no history)."""
    import yfinance as yf

    items = yf.Ticker(ticker).news or []
    titles = []
    for it in items[:limit]:
        title = it.get("title") or (it.get("content") or {}).get("title")
        if title:
            titles.append(title)
    return titles

"""Technical indicators and supervised-learning targets.

Every feature at row t uses data up to and including day t only. The target is the
NEXT day's return, so there is no look-ahead leakage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    c = out["Close"]
    out["ret_1d"] = c.pct_change()
    out["ret_5d"] = c.pct_change(5)
    out["ret_20d"] = c.pct_change(20)
    out["sma_10"] = c.rolling(10).mean()
    out["sma_50"] = c.rolling(50).mean()
    out["ema_12"] = c.ewm(span=12, adjust=False).mean()
    out["ema_26"] = c.ewm(span=26, adjust=False).mean()
    out["macd"] = out["ema_12"] - out["ema_26"]
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()
    out["rsi_14"] = rsi(c)
    mid = c.rolling(20).mean()
    std = c.rolling(20).std()
    out["bb_upper"], out["bb_lower"] = mid + 2 * std, mid - 2 * std
    out["bb_pos"] = (c - out["bb_lower"]) / (out["bb_upper"] - out["bb_lower"])
    out["vol_20d"] = out["ret_1d"].rolling(20).std() * np.sqrt(252)  # annualised volatility
    out["vol_chg"] = out["Volume"].pct_change().replace([np.inf, -np.inf], np.nan)
    out["price_vs_sma50"] = c / out["sma_50"] - 1
    return out


FEATURES = ["ret_1d", "ret_5d", "ret_20d", "macd", "macd_signal", "rsi_14",
            "bb_pos", "vol_20d", "vol_chg", "price_vs_sma50"]


def make_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Indicators plus targets: next-day return and next-day direction (1 = up)."""
    data = add_indicators(df)
    data["target_ret"] = data["Close"].pct_change().shift(-1)
    data["target_up"] = (data["target_ret"] > 0).astype(int)
    data = data.dropna(subset=FEATURES)
    return data


def latest_features(df: pd.DataFrame) -> pd.DataFrame:
    """The most recent row of features, for predicting tomorrow."""
    return add_indicators(df).dropna(subset=FEATURES).iloc[[-1]][FEATURES]

import numpy as np
import pandas as pd

from marketpulse.features import FEATURES, latest_features, make_dataset, rsi
from marketpulse.model import backtest, predict_tomorrow, walk_forward_eval


def synthetic_prices(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    ret = rng.normal(0.0004, 0.012, n)
    close = 100 * np.cumprod(1 + ret)
    idx = pd.bdate_range("2018-01-01", periods=n)
    return pd.DataFrame({"Open": close, "High": close * 1.01, "Low": close * 0.99,
                         "Close": close, "Volume": rng.integers(1e6, 5e6, n)}, index=idx)


def test_rsi_bounds():
    r = rsi(synthetic_prices()["Close"]).dropna()
    assert r.between(0, 100).all()


def test_no_lookahead():
    df = synthetic_prices()
    full = make_dataset(df)
    truncated = make_dataset(df.iloc[:-30])
    common = truncated.index[:-1]
    pd.testing.assert_frame_equal(full.loc[common, FEATURES], truncated.loc[common, FEATURES])


def test_target_is_next_day():
    df = synthetic_prices()
    d = make_dataset(df)
    t = d.index[100]
    nxt = df.index[df.index.get_loc(t) + 1]
    assert abs(d.loc[t, "target_ret"] - (df.loc[nxt, "Close"] / df.loc[t, "Close"] - 1)) < 1e-12


def test_eval_and_backtest_run():
    d = make_dataset(synthetic_prices())
    res = walk_forward_eval(d, n_splits=3)
    assert {"Accuracy", "F1", "Return RMSE"} <= set(res.table.columns)
    assert res.predictions.index.is_monotonic_increasing
    curve = backtest(res.predictions, "pred_Logistic Regression")
    assert curve.notna().all().all()


def test_predict_tomorrow_probabilities():
    df = synthetic_prices()
    p = predict_tomorrow(make_dataset(df), latest_features(df))
    assert all(0 <= v <= 1 for v in p.values())

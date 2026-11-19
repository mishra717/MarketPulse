"""Walk-forward evaluation and prediction.

Models are always compared against naive baselines, so reported numbers are honest:
- Direction: always predict the majority class seen in training.
- Return: always predict a 0% change.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURES


def _classifiers():
    return {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=6,
                                                min_samples_leaf=20, random_state=42, n_jobs=-1),
    }


@dataclass
class EvalResult:
    table: pd.DataFrame          # one row per model/baseline
    predictions: pd.DataFrame    # out-of-sample predictions, indexed by date
    n_train_rows: int
    n_test_rows: int


def walk_forward_eval(data: pd.DataFrame, n_splits: int = 5) -> EvalResult:
    """Expanding-window evaluation: always train on the past, test on the future."""
    X, y_dir, y_ret = data[FEATURES], data["target_up"], data["target_ret"]
    # the last row has no known next-day target
    valid = y_ret.notna()
    X, y_dir, y_ret = X[valid], y_dir[valid], y_ret[valid]

    tscv = TimeSeriesSplit(n_splits=n_splits)
    preds = {name: pd.Series(np.nan, index=X.index) for name in _classifiers()}
    base_dir = pd.Series(np.nan, index=X.index)
    rf_ret = pd.Series(np.nan, index=X.index)

    for train_idx, test_idx in tscv.split(X):
        Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]
        majority = int(y_dir.iloc[train_idx].mean() >= 0.5)
        base_dir.iloc[test_idx] = majority
        for name, clf in _classifiers().items():
            clf.fit(Xtr, y_dir.iloc[train_idx])
            preds[name].iloc[test_idx] = clf.predict(Xte)
        reg = RandomForestRegressor(n_estimators=200, max_depth=5, min_samples_leaf=30,
                                    random_state=42, n_jobs=-1)
        reg.fit(Xtr, y_ret.iloc[train_idx])
        rf_ret.iloc[test_idx] = reg.predict(Xte)

    mask = base_dir.notna()
    y_true = y_dir[mask]
    rows = [{"Model": "Baseline (majority class)",
             "Accuracy": accuracy_score(y_true, base_dir[mask]),
             "F1": f1_score(y_true, base_dir[mask], zero_division=0), "Return RMSE": np.nan}]
    for name, p in preds.items():
        rows.append({"Model": name, "Accuracy": accuracy_score(y_true, p[mask]),
                     "F1": f1_score(y_true, p[mask], zero_division=0), "Return RMSE": np.nan})
    rmse_model = float(np.sqrt(mean_squared_error(y_ret[mask], rf_ret[mask])))
    rmse_zero = float(np.sqrt(mean_squared_error(y_ret[mask], np.zeros(mask.sum()))))
    rows.append({"Model": "Random Forest (return regression)", "Accuracy": np.nan,
                 "F1": np.nan, "Return RMSE": rmse_model})
    rows.append({"Model": "Baseline (predict 0% return)", "Accuracy": np.nan,
                 "F1": np.nan, "Return RMSE": rmse_zero})

    out = pd.DataFrame({"actual_up": y_true, "actual_ret": y_ret[mask],
                        "pred_ret": rf_ret[mask]})
    for name, p in preds.items():
        out[f"pred_{name}"] = p[mask]
    return EvalResult(pd.DataFrame(rows), out, int(len(X) - mask.sum()), int(mask.sum()))


def backtest(preds: pd.DataFrame, model_col: str, cost_bps: float = 5.0) -> pd.DataFrame:
    """Go long when the model predicts up, stay in cash otherwise, versus buy and hold.
    Includes a simple per-trade cost. Past performance is not a forecast."""
    pos = preds[model_col].astype(float)
    trades = pos.diff().abs().fillna(0)
    strat = pos * preds["actual_ret"] - trades * cost_bps / 10_000
    curve = pd.DataFrame({
        "Strategy": (1 + strat).cumprod(),
        "Buy and hold": (1 + preds["actual_ret"]).cumprod(),
    })
    return curve


def predict_tomorrow(data: pd.DataFrame, latest: pd.DataFrame) -> dict[str, float]:
    """Train on all labelled history and return P(up) per model for the next session."""
    labelled = data.dropna(subset=["target_ret"])
    out = {}
    for name, clf in _classifiers().items():
        clf.fit(labelled[FEATURES], labelled["target_up"])
        out[name] = float(clf.predict_proba(latest)[0, 1])
    return out

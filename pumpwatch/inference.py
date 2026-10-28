"""Shared prediction and transparent local sensitivity explanations."""
import numpy as np
import pandas as pd
from .data import validate_candles
from .features import engineer, DESCRIPTIONS
from .train import scores


def predict(frame, bundle, symbol="HISTORICAL"):
    """Score an in-memory historical prefix for replay; no file upload UI is exposed."""
    d, quality = validate_candles(frame)
    if "symbol" not in d:
        d["symbol"] = symbol
    # The trained activity model requires both fields; fail clearly instead of
    # silently claiming optional-feature zeros have the same distribution.
    missing_activity = not {"trades", "taker_buy_volume"}.issubset(d.columns)
    if missing_activity:
        if "core_model" not in bundle:
            raise ValueError("This model needs trades and taker_buy_volume columns.")
        model, names, threshold = bundle['core_model'], bundle['core_features'], bundle['core_threshold']
    else:
        model, names, threshold = bundle['model'], bundle['features'], bundle['threshold']
    f = engineer(d)
    f = f[f.ready].copy()
    if f.empty:
        raise ValueError("At least 61 consecutive one-minute candles are needed. Gaps restart the warm-up.")
    f["risk_score"] = scores(model, f[names])
    f["alert"] = f.risk_score >= threshold
    quality.update(feature_mode="OHLCV only" if missing_activity else "OHLCV + trade activity", threshold=threshold)
    return f, quality


def explain(row, bundle, core=False):
    """One-at-a-time median replacement. Non-additive; not SHAP or causal attribution."""
    names = bundle['core_features'] if core else bundle['features']
    model = bundle['core_model'] if core else bundle['model']
    x = pd.DataFrame([{n: float(row[n]) for n in names}])
    original = float(scores(model, x)[0])
    # Substitute each feature separately so the explanation remains non-additive.
    changed = pd.concat([x] * len(names), ignore_index=True)
    for i, name in enumerate(names):
        changed.loc[i, name] = bundle['reference'][name]
    alternatives = scores(model, changed)
    return pd.DataFrame({"feature": names, "description": [DESCRIPTIONS[n] for n in names],
                         "observed": [row[n] for n in names],
                         "training_median": [bundle['reference'][n] for n in names],
                         "score_change": original - alternatives}).sort_values('score_change', key=abs, ascending=False)

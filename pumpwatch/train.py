"""Chronological model comparison, threshold selection, and untouched test evaluation."""
from __future__ import annotations
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight

from .data import DATA, ROOT
from .features import FEATURES, engineer, label_windows

SEED = 42
ARTIFACTS = ROOT / "artifacts"


def scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    # Monotonic mapping for SVM ranking and thresholding; deliberately NOT calibrated.
    return expit(model.decision_function(X))


def choose_threshold(y, s):
    p, r, t = precision_recall_curve(y, s)
    f = 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-15)
    best = int(np.argmax(f))
    return float(t[best])


def metrics(y, s, threshold):
    pred = np.asarray(s) >= threshold
    return {"precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "average_precision": float(average_precision_score(y, s)),
            "roc_auc": float(roc_auc_score(y, s)),
            "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
            "threshold": float(threshold), "rows": len(y), "positive_rows": int(np.sum(y))}


def split_by_time(d, events):
    days = sorted(pd.to_datetime(events.onset, utc=True).dt.floor("D").unique())
    if len(days) < 10:
        raise ValueError("Need at least ten independent event dates for chronological evaluation.")
    b1, b2 = pd.Timestamp(days[int(len(days)*.6)]), pd.Timestamp(days[int(len(days)*.8)])
    d = d.copy()
    d["split"] = np.where(d.timestamp < b1, "train", np.where(d.timestamp < b2, "validation", "test"))
    # Purge each boundary by the entire feature lookback plus target duration.
    for b in [b1, b2]:
        d.loc[d.timestamp.between(b-pd.Timedelta(minutes=3), b+pd.Timedelta(minutes=60)), "split"] = "purged"
    for event, g in d[d.event_id != ""].groupby("event_id"):
        if g.split.nunique() > 1:
            d.loc[d.event_id == event, "split"] = "purged"
    return d, {"validation_start": str(b1), "test_start": str(b2), "purge_minutes": 60}


def train():
    ARTIFACTS.mkdir(exist_ok=True)
    candles = pd.read_csv(DATA / "candles.csv.gz")
    candles["timestamp"] = pd.to_datetime(candles.timestamp, utc=True)
    events = pd.read_csv(DATA / "events.csv")
    announcements = pd.read_csv(DATA / "known_announcements.csv")
    d = label_windows(engineer(candles), events, announcements)
    d, boundaries = split_by_time(d, events)
    usable = d[d.ready & (d.target >= 0) & (d.split != "purged")].copy()
    parts = {s: usable[usable.split == s] for s in ["train", "validation", "test"]}
    for s, p in parts.items():
        if p.target.nunique() != 2:
            raise ValueError(f"{s} lacks both classes; revise dataset coverage.")
    training = parts["train"]
    positives = training[training.target == 1]
    negatives = training[training.target == 0].sample(min(10000, int((training.target == 0).sum())), random_state=SEED)
    fit = pd.concat([positives, negatives]).sample(frac=1, random_state=SEED)
    X, y = fit[FEATURES], fit.target
    xv, yv = parts["validation"][FEATURES], parts["validation"].target
    xt, yt = parts["test"][FEATURES], parts["test"].target
    candidates = {
        "Logistic Regression": [make_pipeline(StandardScaler(), LogisticRegression(C=c, class_weight="balanced", max_iter=2000, random_state=SEED)) for c in [.1, 1]],
        "Random Forest": [RandomForestClassifier(n_estimators=160, max_depth=depth, min_samples_leaf=3, class_weight="balanced", n_jobs=4, random_state=SEED) for depth in [8, 16]],
        "SVM": [make_pipeline(StandardScaler(), SVC(C=c, kernel="rbf", class_weight="balanced", cache_size=512, random_state=SEED)) for c in [.5, 2]],
        "Gradient Boosting": [HistGradientBoostingClassifier(max_iter=160, max_leaf_nodes=leaves, l2_regularization=1, early_stopping=False, random_state=SEED) for leaves in [7, 15]],
    }
    fitted, results, validation_scores = {}, {}, {}
    for name, models in candidates.items():
        best = None
        for i, model in enumerate(models):
            if name == "Gradient Boosting":
                model.fit(X, y, sample_weight=compute_sample_weight("balanced", y))
            else:
                model.fit(X, y)
            sv = scores(model, xv)
            threshold = choose_threshold(yv, sv)
            m = metrics(yv, sv, threshold)
            print(name, i+1, "validation F1", round(m["f1"], 4), flush=True)
            if best is None or m["f1"] > best[0]["f1"]:
                best = (m, model, sv, i)
        val, fitted[name], validation_scores[name], index = best
        results[name] = {"validation": val, "candidate_index": index,
                         "parameters": str(fitted[name])}
    # Freeze model and thresholds before inspecting any test labels.
    winner = max(results, key=lambda n: results[n]["validation"]["f1"])
    prediction_cols = ["timestamp", "prediction_time", "symbol", "close", "volume", "target", "event_id", "split"]
    predictions = usable[prediction_cols].copy()
    for name, model in fitted.items():
        st = scores(model, xt)
        results[name]["test"] = metrics(yt, st, results[name]["validation"]["threshold"])
        pr, re, _ = precision_recall_curve(yt, st)
        indices = np.unique(np.linspace(0, len(pr)-1, min(500, len(pr))).astype(int))
        results[name]["pr_curve"] = {"precision": pr[indices].tolist(), "recall": re[indices].tolist()}
        joblib.dump({"model": model, "features": FEATURES, "threshold": results[name]["validation"]["threshold"],
                     "name": name, "score_kind": "uncalibrated risk score"}, ARTIFACTS / (name.lower().replace(" ", "_") + ".joblib"))
    best_model = fitted[winner]
    threshold = results[winner]["validation"]["threshold"]
    # Keep ambiguous regions for replay, but exclude them from binary metrics.
    scored = d[d.ready & (d.split != "purged")].copy()
    scored["risk_score"] = scores(best_model, scored[FEATURES])
    scored["alert"] = scored.risk_score >= threshold
    scored.to_csv(ARTIFACTS / "predictions.csv.gz", index=False, compression="gzip")
    bundle = {"model": best_model, "features": FEATURES, "threshold": threshold, "name": winner,
              "score_kind": "uncalibrated risk score", "reference": training[FEATURES].median().to_dict(),
              "feature_schema": "one-minute-v1", "seed": SEED}
    core_features = [n for n in FEATURES if n not in ['trade_z', 'mean_trade_size', 'buy_share', 'activity_available']]
    core_model = clone(best_model)
    if winner == 'Gradient Boosting':
        core_model.fit(X[core_features], y, sample_weight=compute_sample_weight('balanced', y))
    else:
        core_model.fit(X[core_features], y)
    core_threshold = choose_threshold(yv, scores(core_model, xv[core_features]))
    bundle.update(core_model=core_model, core_features=core_features, core_threshold=core_threshold)
    core_test = metrics(yt, scores(core_model, xt[core_features]), core_threshold)
    joblib.dump(bundle, ARTIFACTS / "best_model.joblib")
    # Importance uses validation data only. It is a global ranking, not causation.
    imp_data = pd.concat([parts['validation'].query('target == 1'), parts['validation'].query('target == 0').sample(min(8000, int((yv == 0).sum())), random_state=SEED)])
    importance = permutation_importance(best_model, imp_data[FEATURES], imp_data.target, scoring="average_precision", n_repeats=3, random_state=SEED, n_jobs=1)
    pd.DataFrame({"feature": FEATURES, "importance": importance.importances_mean, "std": importance.importances_std}).sort_values("importance", ascending=False).to_csv(ARTIFACTS / "importance.csv", index=False)
    # Simple rule tuned on validation, evaluated on test.
    rule_choices = []
    for z in [3, 5, 10]:
        for ret in [.01, .03, .05]:
            rule_pred = ((parts['validation'].volume_z >= z) & (parts['validation'].return_1m >= ret)).astype(int)
            rule_choices.append((f1_score(yv, rule_pred), z, ret))
    _, z, ret = max(rule_choices)
    rule_test = ((parts['test'].volume_z >= z) & (parts['test'].return_1m >= ret)).astype(int)
    baselines = {"Always normal": metrics(yt, np.zeros(len(yt)), .5),
                 "Price-volume rule": metrics(yt, rule_test, .5)}
    baselines['Price-volume rule']['rule'] = f"volume_z >= {z} and return_1m >= {ret}"
    event_rows = []
    test_scored = scored[scored.split == "test"]
    for e in events.itertuples():
        g = test_scored[test_scored.event_id == e.event_id]
        if g.empty:
            continue
        alarms = g[g.alert]
        event_rows.append({"event_id": e.event_id, "pair": e.pair, "onset": e.onset, "eligible_minutes": len(g),
                           "detected": bool(len(alarms)), "delay_seconds": float((alarms.prediction_time.min()-pd.Timestamp(e.onset)).total_seconds()) if len(alarms) else None})
    event_report = pd.DataFrame(event_rows)
    event_report.to_csv(ARTIFACTS / "event_results.csv", index=False)
    negative = test_scored[test_scored.target == 0].copy()
    # Alert episodes collapse consecutive alert minutes, with gaps starting new episodes.
    prev_alert = negative.groupby('symbol').alert.shift(fill_value=False)
    gap = negative.groupby('symbol').timestamp.diff().ne(pd.Timedelta(minutes=1))
    episodes = int((negative.alert & (~prev_alert | gap)).sum())
    exposure = len(negative) / 1440
    unseen = parts['test'][~parts['test'].symbol.isin(training.symbol.unique())]
    unseen_report = {"coins": sorted(unseen.symbol.unique().tolist()), "rows": len(unseen), "positives": int(unseen.target.sum())}
    if unseen.target.nunique() == 2:
        unseen_report['metrics'] = metrics(unseen.target, scores(best_model, unseen[FEATURES]), threshold)
    report = {"selected_model": winner, "selection": "Highest validation F1; threshold maximizes validation F1. Test never used for selection.",
              "score_warning": "Uncalibrated scores, not probabilities. Negative sampling and event-conditioned data limit generalization.",
              "boundaries": boundaries, "fit_rows": len(fit), "fit_positive_rows": int(y.sum()),
              "splits": {s: {"rows": len(p), "positives": int(p.target.sum()), "events": p.loc[p.target == 1,'event_id'].nunique(), "start": str(p.timestamp.min()), "end": str(p.timestamp.max())} for s,p in parts.items()},
              "models": results, "baselines": baselines, "unseen_coins": unseen_report, "ohlcv_only_test": core_test,
              "event_evaluation": {"events": len(event_report), "detected": int(event_report.detected.sum()),
                  "recall": float(event_report.detected.mean()), "median_delay_seconds": float(event_report.delay_seconds.median()) if event_report.detected.any() else None,
                  "false_alert_minutes": int(negative.alert.sum()), "false_alert_episodes": episodes,
                  "eligible_negative_coin_days": exposure, "false_alert_episodes_per_coin_day": episodes / exposure,
                  "scope": "Eligible negative minutes on collected event/comparison days; not market-wide deployment rate."}}
    (ARTIFACTS / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"selected_model": winner, "test": results[winner]['test'], "events": report['event_evaluation']}, indent=2), flush=True)


if __name__ == "__main__":
    train()

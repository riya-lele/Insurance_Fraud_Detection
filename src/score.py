"""Scoring, zones and per-claim SHAP for the app. No Streamlit imports, so it is testable."""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap

from src.final_model import (
    calibrated_probs, raw_scores, encode_features, decision_zone, threshold_for,
)

_EXPLAINERS: dict = {}


def to_frame(bundle: dict, row: dict) -> pd.DataFrame:
    """Validate one claim and return a one-row DataFrame in feature order."""
    feats = bundle["feature_list"]
    missing = [f for f in feats
               if f not in row or row[f] is None
               or (isinstance(row[f], float) and np.isnan(row[f]))]
    if missing:
        raise ValueError(f"Missing fields: {', '.join(missing)}")
    for c in bundle["cat_cols"]:
        if row[c] not in bundle["category_maps"][c]:
            raise ValueError(f"Unknown value {row[c]!r} for {c}")
    for c in bundle["num_cols"]:
        try:
            float(row[c])
        except (TypeError, ValueError):
            raise ValueError(f"{c} must be numeric, got {row[c]!r}")
    df = pd.DataFrame([{f: row[f] for f in feats}])
    df[bundle["num_cols"]] = df[bundle["num_cols"]].astype(float)
    return df


def zone_for(prob: float, L: float, c: float) -> str:
    return decision_zone(prob, threshold_for(L=L, c=c))


def score_claim(bundle: dict, row: dict, L: float | None = None, c: float | None = None) -> dict:
    L = bundle["L"] if L is None else L
    c = bundle["c"] if c is None else c
    X = to_frame(bundle, row)
    raw = float(raw_scores(bundle, X)[0])
    prob = float(calibrated_probs(bundle, X)[0])
    t = threshold_for(L=L, c=c)
    return {"raw_score": raw, "prob": prob, "threshold": t, "zone": decision_zone(prob, t)}


def _explainer(bundle: dict):
    key = id(bundle["model"])
    if key not in _EXPLAINERS:
        _EXPLAINERS[key] = shap.TreeExplainer(bundle["model"])
    return _EXPLAINERS[key]


def explain_claim(bundle: dict, row: dict, top: int = 5) -> list[dict]:
    """Top-k features by |SHAP| (log-odds space). Model attribution, not causation."""
    X = to_frame(bundle, row)
    Xe = encode_features(bundle, X)
    sv = _explainer(bundle).shap_values(Xe)
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    vals = sv[0]
    feats = bundle["feature_list"]
    order = np.argsort(-np.abs(vals))[:top]
    return [{"feature": feats[i], "value": X.iloc[0][feats[i]],
             "shap": float(vals[i]),
             "reason": f"{feats[i]} = {X.iloc[0][feats[i]]}"} for i in order]
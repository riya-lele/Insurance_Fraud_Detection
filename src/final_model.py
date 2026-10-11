"""Train and persist the Tier-2 final model (G0 + isotonic calibration)."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split

from src.costs import INVESTIGATION_COST, bayes_threshold
from src.data import load_dataset
from src.models import _fit_g0_raw, transform_ordinal

DEFAULT_L = 10.0
DEFAULT_C = INVESTIGATION_COST
DEFAULT_SEED = 42
DEFAULT_PATH = Path(__file__).resolve().parents[1] / "models" / "final.joblib"


def feature_list(num_cols: list[str], cat_cols: list[str]) -> list[str]:
    return list(num_cols) + list(cat_cols)


def category_maps(enc, cat_cols: list[str]) -> dict[str, list]:
    """Human-readable category maps from a fitted OrdinalEncoder."""
    if not cat_cols:
        return {}
    return {
        col: [None if v is None or (isinstance(v, float) and np.isnan(v)) else v for v in cats]
        for col, cats in zip(cat_cols, enc.categories_)
    }


def decision_zone(prob: float, threshold: float) -> str:
    """Zones at L: >=2x threshold investigate; [t, 2t) review; else approve."""
    p = float(prob)
    t = float(threshold)
    if p >= 2.0 * t:
        return "investigate"
    if p >= t:
        return "review"
    return "approve"


def threshold_for(L: float = DEFAULT_L, c: float = DEFAULT_C) -> float:
    return bayes_threshold(c, L)


def encode_features(bundle: dict, X: pd.DataFrame) -> np.ndarray:
    return transform_ordinal(
        bundle["encoder"], X, bundle["num_cols"], bundle["cat_cols"]
    )


def raw_scores(bundle: dict, X: pd.DataFrame) -> np.ndarray:
    """Raw LightGBM positive-class probabilities (ranking / SHAP model output space via predict_proba)."""
    X_enc = encode_features(bundle, X)
    return bundle["model"].predict_proba(X_enc)[:, 1]


def raw_margins(bundle: dict, X: pd.DataFrame) -> np.ndarray:
    """Raw LightGBM margins (log-odds); TreeExplainer explains these by default."""
    X_enc = encode_features(bundle, X)
    return bundle["model"].predict(X_enc, raw_score=True)


def calibrated_probs(bundle: dict, X: pd.DataFrame) -> np.ndarray:
    raw = raw_scores(bundle, X)
    return np.asarray(bundle["calibrator"].predict(raw), dtype=float)


def train_final(
    X: pd.DataFrame | None = None,
    y: pd.Series | None = None,
    num_cols: list[str] | None = None,
    cat_cols: list[str] | None = None,
    *,
    seed: int = DEFAULT_SEED,
    L: float = DEFAULT_L,
    c: float = DEFAULT_C,
    test_size: float = 0.25,
) -> dict:
    """Fit G0 on 75% stratified; isotonic on held-out 25%. Return saveable bundle."""
    if X is None or y is None or num_cols is None or cat_cols is None:
        X, y, num_cols, cat_cols = load_dataset(verbose=False)

    X_fit, X_cal, y_fit, y_cal = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )

    enc, model, _cat_idx = _fit_g0_raw(
        X_fit, np.asarray(y_fit).astype(int), num_cols, cat_cols, seed
    )
    X_cal_enc = transform_ordinal(enc, X_cal, num_cols, cat_cols)
    raw_cal = model.predict_proba(X_cal_enc)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw_cal, np.asarray(y_cal).astype(int))

    feats = feature_list(num_cols, cat_cols)
    t = threshold_for(L=L, c=c)
    return {
        "model": model,
        "calibrator": calibrator,
        "encoder": enc,
        "category_maps": category_maps(enc, cat_cols),
        "num_cols": list(num_cols),
        "cat_cols": list(cat_cols),
        "feature_list": feats,
        "L": float(L),
        "c": float(c),
        "threshold": float(t),
        "seed": int(seed),
        "cal_index": X_cal.index.to_numpy(),
        "fit_index": X_fit.index.to_numpy(),
    }


def save_final(bundle: dict, path: Path | str = DEFAULT_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return path


def load_final(path: Path | str = DEFAULT_PATH) -> dict:
    return joblib.load(path)


def train_and_save(
    path: Path | str = DEFAULT_PATH,
    *,
    seed: int = DEFAULT_SEED,
    L: float = DEFAULT_L,
    c: float = DEFAULT_C,
) -> dict:
    bundle = train_final(seed=seed, L=L, c=c)
    save_final(bundle, path)
    return bundle

"""Supervised arms L0–G3 with fixed hyperparameters."""

from __future__ import annotations

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

ARMS = ("L0", "G0", "G1", "G2")
ALL_ARMS = ("L0", "G0", "G1", "G2", "G3")

LGBM_PARAMS = dict(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary",
    verbosity=-1,
)


def _lgbm(seed: int, scale_pos_weight: float | None = None) -> LGBMClassifier:
    params = {**LGBM_PARAMS, "random_state": seed}
    if scale_pos_weight is not None:
        params["scale_pos_weight"] = scale_pos_weight
    return LGBMClassifier(**params)


def fit_ordinal(X_fit: pd.DataFrame, cat_cols: list[str]) -> OrdinalEncoder:
    enc = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        encoded_missing_value=-1,
    )
    if cat_cols:
        enc.fit(X_fit[cat_cols])
    return enc


def transform_ordinal(
    enc: OrdinalEncoder,
    X: pd.DataFrame,
    num_cols: list[str],
    cat_cols: list[str],
) -> np.ndarray:
    parts = []
    if num_cols:
        parts.append(X[num_cols].to_numpy(dtype=float))
    if cat_cols:
        parts.append(enc.transform(X[cat_cols]))
    return np.hstack(parts) if parts else np.empty((len(X), 0))


def cat_indices(num_cols: list[str], cat_cols: list[str]) -> list[int]:
    return list(range(len(num_cols), len(num_cols) + len(cat_cols)))


def f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Threshold on scores that maximises F1 (unique score cutpoints)."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    candidates = np.unique(scores)
    if len(candidates) == 0:
        return 0.5
    best_t, best_f1 = 0.5, -1.0
    for t in candidates:
        pred = (scores >= t).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def _fit_g0_raw(
    X_fit: pd.DataFrame,
    y_fit: np.ndarray,
    num_cols: list[str],
    cat_cols: list[str],
    seed: int,
):
    enc = fit_ordinal(X_fit, cat_cols)
    X_fit_enc = transform_ordinal(enc, X_fit, num_cols, cat_cols)
    cat_idx = cat_indices(num_cols, cat_cols)
    model = _lgbm(seed)
    model.fit(X_fit_enc, y_fit, categorical_feature=cat_idx)
    return enc, model, cat_idx


def fit_predict_scores(
    arm: str,
    X_fit: pd.DataFrame,
    y_fit: pd.Series | np.ndarray,
    X_eval: pd.DataFrame,
    num_cols: list[str],
    cat_cols: list[str],
    seed: int = 0,
    X_val: pd.DataFrame | None = None,
    y_val: pd.Series | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Train arm on fit split; return (eval_scores, fit_positions, extras).

    G3 requires X_val/y_val: isotonic calibration and F1 threshold on val only.
    """
    if arm not in ALL_ARMS:
        raise ValueError(f"Unknown arm {arm!r}; expected one of {ALL_ARMS}")

    y_fit = np.asarray(y_fit).astype(int)
    fit_positions = np.arange(len(X_fit))
    extras: dict = {}

    if arm == "L0":
        pre = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), num_cols),
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    cat_cols,
                ),
            ],
            remainder="drop",
        )
        clf = LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=seed,
            solver="lbfgs",
        )
        pipe = Pipeline([("pre", pre), ("clf", clf)])
        pipe.fit(X_fit, y_fit)
        scores = pipe.predict_proba(X_eval)[:, 1]
        return scores, fit_positions, extras

    if arm in ("G0", "G1", "G2"):
        enc = fit_ordinal(X_fit, cat_cols)
        X_fit_enc = transform_ordinal(enc, X_fit, num_cols, cat_cols)
        X_eval_enc = transform_ordinal(enc, X_eval, num_cols, cat_cols)
        cat_idx = cat_indices(num_cols, cat_cols)

        if arm == "G0":
            model = _lgbm(seed)
            model.fit(X_fit_enc, y_fit, categorical_feature=cat_idx)
        elif arm == "G1":
            n_pos = max(int(y_fit.sum()), 1)
            n_neg = max(int((y_fit == 0).sum()), 1)
            model = _lgbm(seed, scale_pos_weight=n_neg / n_pos)
            model.fit(X_fit_enc, y_fit, categorical_feature=cat_idx)
        else:
            if len(cat_idx) == 0:
                raise ValueError("G2/SMOTENC requires at least one categorical feature")
            min_class = int(np.bincount(y_fit).min())
            k_neighbors = max(1, min(5, min_class - 1))
            smote = SMOTENC(
                categorical_features=cat_idx,
                random_state=seed,
                k_neighbors=k_neighbors,
            )
            X_res, y_res = smote.fit_resample(X_fit_enc, y_fit)
            model = _lgbm(seed)
            model.fit(X_res, y_res, categorical_feature=cat_idx)

        scores = model.predict_proba(X_eval_enc)[:, 1]
        return scores, fit_positions, extras

    # G3: G0 + isotonic on val + F1 threshold on calibrated val
    if X_val is None or y_val is None:
        raise ValueError("G3 requires X_val and y_val for calibration/thresholding")
    y_val_arr = np.asarray(y_val).astype(int)
    enc, model, cat_idx = _fit_g0_raw(X_fit, y_fit, num_cols, cat_cols, seed)
    raw_val = model.predict_proba(transform_ordinal(enc, X_val, num_cols, cat_cols))[:, 1]
    raw_eval = model.predict_proba(transform_ordinal(enc, X_eval, num_cols, cat_cols))[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw_val, y_val_arr)
    cal_val = calibrator.predict(raw_val)
    cal_eval = calibrator.predict(raw_eval)
    extras["threshold"] = f1_threshold(y_val_arr, cal_val)
    extras["calibrator"] = calibrator
    return cal_eval, fit_positions, extras

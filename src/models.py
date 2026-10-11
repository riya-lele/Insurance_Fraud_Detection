"""Imbalance arms L0, G0, G1, G2 with fixed hyperparameters."""

from __future__ import annotations

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

ARMS = ("L0", "G0", "G1", "G2")

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


def _fit_ordinal(X_fit: pd.DataFrame, cat_cols: list[str], seed: int) -> OrdinalEncoder:
    enc = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        encoded_missing_value=-1,
    )
    if cat_cols:
        enc.fit(X_fit[cat_cols])
    return enc


def _transform_ordinal(
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


def _cat_indices(num_cols: list[str], cat_cols: list[str]) -> list[int]:
    return list(range(len(num_cols), len(num_cols) + len(cat_cols)))


def fit_predict_scores(
    arm: str,
    X_fit: pd.DataFrame,
    y_fit: pd.Series | np.ndarray,
    X_eval: pd.DataFrame,
    num_cols: list[str],
    cat_cols: list[str],
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Train arm on fit split; return (eval_scores, fit_row_indices_used).

    fit_row_indices_used are positions into X_fit (0..n_fit-1) for real rows
    that entered the model fit set before any synthetic oversampling.
    """
    if arm not in ARMS:
        raise ValueError(f"Unknown arm {arm!r}; expected one of {ARMS}")

    y_fit = np.asarray(y_fit).astype(int)
    fit_positions = np.arange(len(X_fit))

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
        return scores, fit_positions

    enc = _fit_ordinal(X_fit, cat_cols, seed)
    X_fit_enc = _transform_ordinal(enc, X_fit, num_cols, cat_cols)
    X_eval_enc = _transform_ordinal(enc, X_eval, num_cols, cat_cols)
    cat_idx = _cat_indices(num_cols, cat_cols)

    if arm == "G0":
        model = _lgbm(seed)
        model.fit(X_fit_enc, y_fit, categorical_feature=cat_idx)
        scores = model.predict_proba(X_eval_enc)[:, 1]
        return scores, fit_positions

    if arm == "G1":
        n_pos = max(int(y_fit.sum()), 1)
        n_neg = max(int((y_fit == 0).sum()), 1)
        spw = n_neg / n_pos
        model = _lgbm(seed, scale_pos_weight=spw)
        model.fit(X_fit_enc, y_fit, categorical_feature=cat_idx)
        scores = model.predict_proba(X_eval_enc)[:, 1]
        return scores, fit_positions

    # G2: SMOTENC on fit part only
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
    return scores, fit_positions

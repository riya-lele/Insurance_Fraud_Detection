"""Isolation Forest arms and supervised+anomaly combinations."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

from src.models import LGBM_PARAMS, cat_indices, fit_ordinal, transform_ordinal

IF_PARAMS = dict(
    n_estimators=200,
    max_samples="auto",
    contamination="auto",
    random_state=0,
)

RANK_WEIGHT_GRID = (0.3, 0.5, 0.7)


def _iforest(seed: int) -> IsolationForest:
    params = {**IF_PARAMS, "random_state": seed}
    return IsolationForest(**params)


def encode_features(
    X_fit: pd.DataFrame,
    X_others: list[pd.DataFrame],
    num_cols: list[str],
    cat_cols: list[str],
) -> tuple[np.ndarray, list[np.ndarray]]:
    enc = fit_ordinal(X_fit, cat_cols)
    fit_enc = transform_ordinal(enc, X_fit, num_cols, cat_cols)
    others = [transform_ordinal(enc, X, num_cols, cat_cols) for X in X_others]
    return fit_enc, others


def anomaly_score(model: IsolationForest, X_enc: np.ndarray) -> np.ndarray:
    """Higher = more anomalous (negate decision_function)."""
    return -model.decision_function(X_enc)


def fit_if_all(X_enc: np.ndarray, seed: int = 0) -> IsolationForest:
    model = _iforest(seed)
    model.fit(X_enc)
    return model


def fit_if_normal(X_enc: np.ndarray, y: np.ndarray, seed: int = 0) -> IsolationForest:
    y = np.asarray(y).astype(int)
    normal = X_enc[y == 0]
    if len(normal) == 0:
        raise ValueError("IF_normal requires at least one non-fraud row")
    model = _iforest(seed)
    model.fit(normal)
    return model


def scores_if_all(
    X_fit: pd.DataFrame,
    X_eval: pd.DataFrame,
    num_cols: list[str],
    cat_cols: list[str],
    seed: int = 0,
) -> np.ndarray:
    fit_enc, (eval_enc,) = encode_features(X_fit, [X_eval], num_cols, cat_cols)
    model = fit_if_all(fit_enc, seed=seed)
    return anomaly_score(model, eval_enc)


def scores_if_normal(
    X_fit: pd.DataFrame,
    y_fit: pd.Series | np.ndarray,
    X_eval: pd.DataFrame,
    num_cols: list[str],
    cat_cols: list[str],
    seed: int = 0,
) -> np.ndarray:
    fit_enc, (eval_enc,) = encode_features(X_fit, [X_eval], num_cols, cat_cols)
    model = fit_if_normal(fit_enc, np.asarray(y_fit), seed=seed)
    return anomaly_score(model, eval_enc)


def to_rank(scores: np.ndarray) -> np.ndarray:
    """Average ranks scaled to (0, 1]."""
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    order = scores.argsort().argsort().astype(float) + 1.0
    return order / n


def rank_average(sup_scores: np.ndarray, anom_scores: np.ndarray, weight: float = 0.5) -> np.ndarray:
    w = float(weight)
    return w * to_rank(sup_scores) + (1.0 - w) * to_rank(anom_scores)


def choose_rank_weight(
    y_val: np.ndarray,
    sup_val: np.ndarray,
    anom_val: np.ndarray,
    grid: tuple[float, ...] = RANK_WEIGHT_GRID,
) -> float:
    y_val = np.asarray(y_val).astype(int)
    best_w, best_pr = 0.5, -1.0
    for w in grid:
        blended = rank_average(sup_val, anom_val, weight=w)
        pr = average_precision_score(y_val, blended)
        if pr > best_pr:
            best_pr, best_w = pr, float(w)
    return best_w


def oof_anomaly_feature(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    num_cols: list[str],
    cat_cols: list[str],
    mode: str = "IF_normal",
    n_splits: int = 3,
    seed: int = 0,
    leakage_callback: Callable | None = None,
) -> np.ndarray:
    """Out-of-fold anomaly scores for training rows.

    For each row, the IF used to score it is fit only on other folds' rows
    (and for IF_normal, only non-fraud among those).
    """
    if mode not in ("IF_all", "IF_normal"):
        raise ValueError(f"Unknown mode {mode!r}")

    y = np.asarray(y).astype(int)
    n = len(X)
    oof = np.full(n, np.nan)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for fold_i, (tr, te) in enumerate(skf.split(X, y)):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr = y[tr]
        fit_enc, (te_enc,) = encode_features(X_tr, [X_te], num_cols, cat_cols)
        if mode == "IF_all":
            model = fit_if_all(fit_enc, seed=seed + fold_i)
            fit_row_positions = tr
        else:
            model = fit_if_normal(fit_enc, y_tr, seed=seed + fold_i)
            fit_row_positions = tr[y_tr == 0]

        oof[te] = anomaly_score(model, te_enc)

        if leakage_callback is not None:
            leakage_callback(
                fold=fold_i,
                mode=mode,
                scored_idx=np.asarray(te),
                if_fit_idx=np.asarray(fit_row_positions),
            )

    if np.isnan(oof).any():
        raise RuntimeError("OOF anomaly feature has missing values")
    return oof


def full_anomaly_feature(
    X_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    X_eval: pd.DataFrame,
    num_cols: list[str],
    cat_cols: list[str],
    mode: str = "IF_normal",
    seed: int = 0,
) -> np.ndarray:
    """Anomaly feature for eval rows from IF fit on all training rows."""
    train_enc, (eval_enc,) = encode_features(X_train, [X_eval], num_cols, cat_cols)
    y_train = np.asarray(y_train).astype(int)
    if mode == "IF_all":
        model = fit_if_all(train_enc, seed=seed)
    else:
        model = fit_if_normal(train_enc, y_train, seed=seed)
    return anomaly_score(model, eval_enc)


def stack_fit_predict(
    X_fit: pd.DataFrame,
    y_fit: pd.Series | np.ndarray,
    anom_fit: np.ndarray,
    eval_sets: list[tuple[pd.DataFrame, np.ndarray]],
    num_cols: list[str],
    cat_cols: list[str],
    seed: int = 0,
) -> list[np.ndarray]:
    """LightGBM with anomaly score as an extra numeric feature (G0-style).

    eval_sets: list of (X_eval, anom_eval) pairs; returns one score array each.
    """
    y_fit = np.asarray(y_fit).astype(int)
    X_fit_aug = X_fit.copy()
    X_fit_aug["_anom"] = np.asarray(anom_fit, dtype=float)
    num_aug = list(num_cols) + ["_anom"]

    enc = fit_ordinal(X_fit_aug, cat_cols)
    X_fit_enc = transform_ordinal(enc, X_fit_aug, num_aug, cat_cols)
    cat_idx = cat_indices(num_aug, cat_cols)
    model = LGBMClassifier(**{**LGBM_PARAMS, "random_state": seed})
    model.fit(X_fit_enc, y_fit, categorical_feature=cat_idx)

    out = []
    for X_eval, anom_eval in eval_sets:
        X_eval_aug = X_eval.copy()
        X_eval_aug["_anom"] = np.asarray(anom_eval, dtype=float)
        X_eval_enc = transform_ordinal(enc, X_eval_aug, num_aug, cat_cols)
        out.append(model.predict_proba(X_eval_enc)[:, 1])
    return out

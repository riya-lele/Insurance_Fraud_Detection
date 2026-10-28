"""Shared pieces for the Optuna extension: splits, metrics and model factories."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_recall_curve, roc_auc_score)
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

OUTER_SEED = 42
KS = (0.01, 0.02, 0.05, 0.10)
N_JOBS = int(os.environ.get("N_JOBS", "4"))


# ---------------------------------------------------------------- splits
def make_splits(y, n_splits=5, n_repeats=3, seed=OUTER_SEED,
                cal_size=0.25, fitness_size=0.20):
    """Yield positional index arrays for every outer fold.

    outer train -> fit (75%) + cal (25%)       cal: calibration and policy selection only
    fit         -> fit_train (80%) + fitness (20%)   fitness: model/trial selection only
    test: used once, for the final evaluation.
    """
    y = np.asarray(y).astype(int)
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    for fold, (train_idx, test_idx) in enumerate(rskf.split(np.zeros(len(y)), y)):
        fit_idx, cal_idx = train_test_split(
            train_idx, test_size=cal_size, stratify=y[train_idx], random_state=seed + fold)
        fit_train, fitness = train_test_split(
            fit_idx, test_size=fitness_size, stratify=y[fit_idx], random_state=seed + 1000 + fold)
        yield dict(fold=fold, fit=fit_idx, cal=cal_idx, test=test_idx,
                   fit_train=fit_train, fitness=fitness)


# ---------------------------------------------------------------- metrics
def k_count(n: int, frac: float) -> int:
    return max(1, int(round(frac * n)))


def topk_metrics(y, score) -> dict:
    """Precision@K and Recall@K, ranking by the raw score (stable sort)."""
    y = np.asarray(y).astype(int)
    n = len(y)
    order = np.argsort(-np.asarray(score, dtype=float), kind="stable")
    total = int(y.sum())
    out = {}
    for frac in KS:
        k = k_count(n, frac)
        caught = int(y[order[:k]].sum())
        tag = int(round(frac * 100))
        out[f"k_at_{tag}pct"] = k
        out[f"caught_at_{tag}pct"] = caught
        out[f"precision_at_{tag}pct"] = caught / k
        out[f"recall_at_{tag}pct"] = caught / total if total else float("nan")
    return out


def f1_threshold(y_cal, s_cal) -> float:
    """Threshold that maximises F1 on the calibration partition (a permitted use)."""
    p, r, thr = precision_recall_curve(y_cal, s_cal)
    f1 = 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-12)
    return float(thr[int(np.argmax(f1))])


def evaluate_scores(y_test, s_test, y_cal, s_cal) -> dict:
    """Ranking metrics use raw scores. Brier uses isotonic probabilities fitted on the
    calibration partition only, so it is an out-of-sample measurement on the test fold."""
    y_test = np.asarray(y_test).astype(int)
    s_test = np.asarray(s_test, dtype=float)
    y_cal = np.asarray(y_cal).astype(int)
    s_cal = np.asarray(s_cal, dtype=float)

    iso = IsotonicRegression(out_of_bounds="clip").fit(s_cal, y_cal)
    p_test = iso.predict(s_test)
    thr = f1_threshold(y_cal, s_cal)

    m = dict(
        pr_auc=float(average_precision_score(y_test, s_test)),
        roc_auc=float(roc_auc_score(y_test, s_test)),
        brier_calibrated=float(np.mean((p_test - y_test) ** 2)),
        f1_at_cal_threshold=float(f1_score(y_test, (s_test >= thr).astype(int), zero_division=0)),
        f1_threshold=thr,
    )
    m.update(topk_metrics(y_test, s_test))
    return m


# ---------------------------------------------------------------- models
# Each fit_* returns a function X -> fraud score. Everything learned (scalers, encoders,
# category lists) is fitted on the training subset passed in, never on anything else.
def fit_logreg(Xtr, ytr, num_cols, cat_cols, seed):
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    pre = ColumnTransformer([
        ("num", StandardScaler(), list(num_cols)),
        ("cat", OneHotEncoder(handle_unknown="ignore"), list(cat_cols)),
    ])
    pipe = Pipeline([("pre", pre),
                     ("clf", LogisticRegression(class_weight="balanced", C=1.0,
                                                max_iter=2000, random_state=seed))])
    pipe.fit(Xtr, np.asarray(ytr).astype(int))
    return lambda X: pipe.predict_proba(X)[:, 1]


def fit_g0(Xtr, ytr, num_cols, cat_cols, seed):
    """The project's existing LightGBM G0, reused unchanged."""
    from src.models import _fit_g0_raw, transform_ordinal
    enc, model, _ = _fit_g0_raw(Xtr, np.asarray(ytr).astype(int), num_cols, cat_cols, seed)
    return lambda X: model.predict_proba(transform_ordinal(enc, X, num_cols, cat_cols))[:, 1]


def fit_xgb(Xtr, ytr, num_cols, cat_cols, seed):
    from xgboost import XGBClassifier

    cats = {c: pd.CategoricalDtype(sorted(Xtr[c].astype(str).unique())) for c in cat_cols}

    def prep(X):
        out = X.copy()
        for c in cat_cols:
            out[c] = out[c].astype(str).astype(cats[c])   # unseen category -> NaN
        return out

    model = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.8,
                          colsample_bytree=0.8, tree_method="hist", enable_categorical=True,
                          eval_metric="aucpr", random_state=seed, n_jobs=N_JOBS, verbosity=0)
    model.fit(prep(Xtr), np.asarray(ytr).astype(int))
    return lambda X: model.predict_proba(prep(X))[:, 1]


def fit_cat(Xtr, ytr, num_cols, cat_cols, seed):
    from catboost import CatBoostClassifier

    def prep(X):
        out = X.copy()
        for c in cat_cols:
            out[c] = out[c].astype(str)
        return out

    model = CatBoostClassifier(iterations=300, learning_rate=0.05, depth=6, random_seed=seed,
                               verbose=0, thread_count=N_JOBS, allow_writing_files=False)
    model.fit(prep(Xtr), np.asarray(ytr).astype(int), cat_features=list(cat_cols))
    return lambda X: model.predict_proba(prep(X))[:, 1]


MODELS = {"LR": fit_logreg, "G0": fit_g0, "XGB": fit_xgb, "CAT": fit_cat}
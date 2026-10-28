"""Optuna search for LightGBM. tune_fold receives ONLY the fit-train and fitness arrays,
so it cannot see the calibration or test partitions."""
from __future__ import annotations

import numpy as np
import optuna
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score

RANGES = {
    "num_leaves": (8, 64),
    "learning_rate": (0.01, 0.2),
    "n_estimators": (100, 500),
    "min_child_samples": (10, 80),
    "feature_fraction": (0.5, 1.0),
    "reg_lambda": (0.001, 10.0),
}


def suggest_params(trial) -> dict:
    return dict(
        num_leaves=trial.suggest_int("num_leaves", *RANGES["num_leaves"]),
        learning_rate=trial.suggest_float("learning_rate", *RANGES["learning_rate"], log=True),
        n_estimators=trial.suggest_int("n_estimators", *RANGES["n_estimators"]),
        min_child_samples=trial.suggest_int("min_child_samples", *RANGES["min_child_samples"]),
        feature_fraction=trial.suggest_float("feature_fraction", *RANGES["feature_fraction"]),
        reg_lambda=trial.suggest_float("reg_lambda", *RANGES["reg_lambda"], log=True),
    )


def to_lgbm(params: dict) -> dict:
    p = dict(params)
    p["colsample_bytree"] = p.pop("feature_fraction")   # scikit-learn API name for feature_fraction
    return p


def fit_lgbm(Xtr, ytr, base, params, cat_idx, seed, n_jobs=1):
    """Fit LightGBM with G0's own parameters (base) overridden only by the tuned ones."""
    kw = dict(base)
    kw.update(to_lgbm(params))
    kw.update(random_state=int(seed), n_jobs=int(n_jobs))
    kw.setdefault("verbose", -1)
    model = LGBMClassifier(**kw)
    model.fit(Xtr, np.asarray(ytr).astype(int),
              categorical_feature=cat_idx if cat_idx else "auto")
    return model


def tune_fold(Xtr, ytr, Xfv, yfv, base, cat_idx, seed, n_trials, timeout=None, n_jobs=1):
    """TPE search maximising PR-AUC on the internal fitness split.
    Exceptions are not caught: a systematic error stops the run instead of being hidden."""
    sampler = optuna.samplers.TPESampler(seed=int(seed))
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial):
        params = suggest_params(trial)
        model = fit_lgbm(Xtr, ytr, base, params, cat_idx, seed, n_jobs)
        return float(average_precision_score(yfv, model.predict_proba(Xfv)[:, 1]))

    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    return study
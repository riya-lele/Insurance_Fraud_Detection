import numpy as np
import optuna
import pytest

import src.opt_lgbm as m
from src.opt_lgbm import RANGES, fit_lgbm, suggest_params, to_lgbm, tune_fold

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _toy(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    y = (X[:, 0] + 0.5 * rng.normal(size=n) > 1.0).astype(int)
    return X, y


def test_search_space_ranges():
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    seen = []

    def obj(trial):
        seen.append(suggest_params(trial))
        return 0.0

    study.optimize(obj, n_trials=200)
    for p in seen:
        for k, (lo, hi) in RANGES.items():
            assert lo <= p[k] <= hi
        assert isinstance(p["num_leaves"], int) and isinstance(p["n_estimators"], int)


def test_feature_fraction_is_renamed():
    out = to_lgbm({"feature_fraction": 0.7, "num_leaves": 10})
    assert out["colsample_bytree"] == 0.7 and "feature_fraction" not in out


def test_trials_use_only_the_given_fit_and_fitness_rows(monkeypatch):
    X, y = _toy()
    fit_sizes, score_sizes = [], []
    orig_fit = m.LGBMClassifier.fit
    real_ap = m.average_precision_score

    def spy_fit(self, Xa, ya, *a, **k):
        fit_sizes.append(len(ya))
        return orig_fit(self, Xa, ya, *a, **k)

    def spy_ap(yt, s):
        score_sizes.append(len(yt))
        return real_ap(yt, s)

    monkeypatch.setattr(m.LGBMClassifier, "fit", spy_fit)
    monkeypatch.setattr(m, "average_precision_score", spy_ap)
    tune_fold(X[:300], y[:300], X[300:], y[300:], {}, [], seed=0, n_trials=3, n_jobs=1)
    assert fit_sizes == [300] * 3      # trials train only on fit_train
    assert score_sizes == [100] * 3    # and are scored only on the fitness split


def test_tuning_is_reproducible():
    X, y = _toy()
    args = (X[:300], y[:300], X[300:], y[300:], {}, [])
    s1 = tune_fold(*args, seed=1, n_trials=4, n_jobs=1)
    s2 = tune_fold(*args, seed=1, n_trials=4, n_jobs=1)
    assert len(s1.trials) == 4
    assert [t.params for t in s1.trials] == [t.params for t in s2.trials]
    assert [round(t.value, 6) for t in s1.trials] == [round(t.value, 6) for t in s2.trials]


def test_refit_applies_the_selected_configuration():
    X, y = _toy()
    best = dict(num_leaves=12, learning_rate=0.1, n_estimators=120, min_child_samples=15,
                feature_fraction=0.7, reg_lambda=0.5)
    model = fit_lgbm(X, y, {}, best, [], seed=0)
    assert model.get_params()["num_leaves"] == 12
    assert model.get_params()["colsample_bytree"] == 0.7
    assert model.booster_.num_trees() == 120
"""Final model persistence and SHAP additivity checks."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import load_dataset
from src.explain import shap_matrix
from src.final_model import (
    calibrated_probs,
    load_final,
    raw_margins,
    raw_scores,
    save_final,
    train_final,
)


def _small_split():
    X, y, num_cols, cat_cols = load_dataset(verbose=False)
    rng = np.random.default_rng(0)
    pos = np.flatnonzero(y.to_numpy() == 1)
    neg = np.flatnonzero(y.to_numpy() == 0)
    take = np.concatenate(
        [
            rng.choice(pos, size=80, replace=False),
            rng.choice(neg, size=400, replace=False),
        ]
    )
    X_s = X.iloc[take].reset_index(drop=True)
    y_s = y.iloc[take].reset_index(drop=True)
    return X_s, y_s, num_cols, cat_cols


def test_shap_plus_base_equals_raw_margin():
    X, y, num_cols, cat_cols = _small_split()
    bundle = train_final(X, y, num_cols, cat_cols, seed=3)
    # A few held-out (calibration) rows
    idx = bundle["cal_index"][:5]
    X_rows = X.loc[idx]

    values, base, _feats = shap_matrix(bundle, X_rows)
    reconstructed = values.sum(axis=1) + base
    margins = raw_margins(bundle, X_rows)
    np.testing.assert_allclose(reconstructed, margins, rtol=1e-5, atol=1e-5)


def test_saved_model_reloads_identical_scores(tmp_path):
    X, y, num_cols, cat_cols = _small_split()
    bundle = train_final(X, y, num_cols, cat_cols, seed=5)
    path = tmp_path / "final.joblib"
    save_final(bundle, path)

    loaded = load_final(path)
    X_check = X.iloc[:30]
    np.testing.assert_allclose(
        raw_scores(bundle, X_check), raw_scores(loaded, X_check), rtol=0, atol=0
    )
    np.testing.assert_allclose(
        calibrated_probs(bundle, X_check),
        calibrated_probs(loaded, X_check),
        rtol=0,
        atol=0,
    )
    assert loaded["threshold"] == pytest.approx(bundle["threshold"])
    assert loaded["feature_list"] == bundle["feature_list"]

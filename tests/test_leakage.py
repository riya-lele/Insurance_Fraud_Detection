"""Leakage and metric sanity checks for the evaluation harness."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import load_dataset
from src.evaluate import compute_metrics, precision_recall_at_k, run_cv
from src.models import fit_predict_scores


def test_no_test_indices_in_fit_or_resampled_training():
    X, y, num_cols, cat_cols = load_dataset(verbose=False)
    # Subsample for speed while keeping both classes
    rng = np.random.default_rng(0)
    pos = np.flatnonzero(y.to_numpy() == 1)
    neg = np.flatnonzero(y.to_numpy() == 0)
    take = np.concatenate(
        [
            rng.choice(pos, size=80, replace=False),
            rng.choice(neg, size=400, replace=False),
        ]
    )
    X_s, y_s = X.iloc[take].reset_index(drop=True), y.iloc[take].reset_index(drop=True)

    seen: list[tuple] = []

    def callback(**kwargs):
        seen.append(kwargs)

    run_cv(
        X_s,
        y_s,
        num_cols,
        cat_cols,
        arms=("L0", "G0", "G1", "G2"),
        n_splits=3,
        n_repeats=1,
        seed=7,
        leakage_callback=callback,
    )

    assert len(seen) == 3 * 4
    for info in seen:
        test_idx = set(info["test_idx"].tolist())
        fit_idx = set(info["fit_idx"].tolist())
        train_idx = set(info["train_idx"].tolist())
        assert fit_idx.isdisjoint(test_idx), info["arm"]
        assert fit_idx.issubset(train_idx), info["arm"]
        assert test_idx.isdisjoint(train_idx), info["arm"]


def test_metrics_match_hand_computed_toy():
    # scores descending preference: indices 0,1,2,3,4
    y = np.array([1, 0, 1, 0, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.1])
    # K = floor(5 * 0.4) = 2 → top {1, 0} labels → tp=1 → P=0.5, R=0.5
    p, r = precision_recall_at_k(y, scores, 0.4)
    assert p == pytest.approx(0.5)
    assert r == pytest.approx(0.5)

    # Hand AP: sorted by score desc, precision at each positive:
    # pos@1 (idx0): P=1/1; pos@3 (idx2): P=2/3; AP=(1 + 2/3)/2 = 5/6
    metrics = compute_metrics(y, scores)
    assert metrics["pr_auc"] == pytest.approx(5 / 6)
    # pos scores beat 3 and 2 of 3 negatives → (3+2)/(2*3) = 5/6
    assert metrics["roc_auc"] == pytest.approx(5 / 6)


def test_shuffled_labels_pr_auc_near_base_rate():
    X, y, num_cols, cat_cols = load_dataset(verbose=False)
    rng = np.random.default_rng(123)
    pos = np.flatnonzero(y.to_numpy() == 1)
    neg = np.flatnonzero(y.to_numpy() == 0)
    take = np.concatenate(
        [
            rng.choice(pos, size=120, replace=False),
            rng.choice(neg, size=880, replace=False),
        ]
    )
    X_s = X.iloc[take].reset_index(drop=True)
    y_s = y.iloc[take].reset_index(drop=True)
    y_shuf = pd.Series(rng.permutation(y_s.to_numpy()), index=X_s.index)

    # Single stratified holdout
    from sklearn.model_selection import train_test_split

    X_fit, X_te, y_fit, y_te = train_test_split(
        X_s, y_shuf, test_size=0.3, stratify=y_shuf, random_state=0
    )
    scores, _, _ = fit_predict_scores(
        "G0", X_fit, y_fit, X_te, num_cols, cat_cols, seed=0
    )
    pr = average_precision_score(y_te, scores)
    base = float(y_te.mean())
    # With shuffled labels, PR-AUC should sit near prevalence (loose band)
    assert abs(pr - base) < 0.12, f"pr={pr:.3f} base={base:.3f}"

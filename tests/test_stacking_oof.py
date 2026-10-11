"""Stacking anomaly feature on training rows must be out-of-fold."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.anomaly import oof_anomaly_feature
from src.data import load_dataset


def test_stacking_anomaly_feature_is_out_of_fold():
    X, y, num_cols, cat_cols = load_dataset(verbose=False)
    rng = np.random.default_rng(1)
    pos = np.flatnonzero(y.to_numpy() == 1)
    neg = np.flatnonzero(y.to_numpy() == 0)
    take = np.concatenate(
        [
            rng.choice(pos, size=60, replace=False),
            rng.choice(neg, size=300, replace=False),
        ]
    )
    X_s = X.iloc[take].reset_index(drop=True)
    y_s = y.iloc[take].reset_index(drop=True)

    seen: list[dict] = []

    def callback(**kwargs):
        seen.append(kwargs)

    for mode in ("IF_all", "IF_normal"):
        seen.clear()
        oof_anomaly_feature(
            X_s,
            y_s,
            num_cols,
            cat_cols,
            mode=mode,
            n_splits=3,
            seed=11,
            leakage_callback=callback,
        )
        assert len(seen) == 3
        for info in seen:
            scored = set(info["scored_idx"].tolist())
            fitted = set(info["if_fit_idx"].tolist())
            assert scored.isdisjoint(fitted), (mode, info["fold"])

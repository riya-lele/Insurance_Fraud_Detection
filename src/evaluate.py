"""Repeated stratified CV harness, metrics, and paired bootstrap."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

from src.models import ARMS, fit_predict_scores

K_FRACS = (0.01, 0.02, 0.05, 0.10)


def precision_recall_at_k(
    y_true: np.ndarray,
    scores: np.ndarray,
    k_frac: float,
) -> tuple[float, float]:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    n = len(y_true)
    k = max(1, int(np.floor(n * k_frac)))
    top = np.argsort(scores)[::-1][:k]
    tp = float(y_true[top].sum())
    precision = tp / k
    n_pos = float(y_true.sum())
    recall = tp / n_pos if n_pos > 0 else 0.0
    return precision, recall


def compute_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    out: dict[str, float] = {
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
    }
    for frac in K_FRACS:
        pct = int(frac * 100)
        p, r = precision_recall_at_k(y_true, scores, frac)
        out[f"precision_at_{pct}pct"] = p
        out[f"recall_at_{pct}pct"] = r
    return out


def paired_bootstrap_ci(
    diffs: np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Mean paired difference and percentile bootstrap CI over folds.

    Note: CV folds are correlated, so the CI is approximate.
    """
    diffs = np.asarray(diffs, dtype=float)
    if len(diffs) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(diffs, size=len(diffs), replace=True)
        means[i] = sample.mean()
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(diffs.mean()), float(lo), float(hi)


def run_cv(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    num_cols: list[str],
    cat_cols: list[str],
    arms: tuple[str, ...] = ARMS,
    n_splits: int = 5,
    n_repeats: int = 3,
    seed: int = 42,
    inner_val_size: float = 0.25,
    leakage_callback: Callable | None = None,
) -> pd.DataFrame:
    """Outer repeated stratified CV with inner 75/25 fit/validation split.

    Models train on the fit portion only. Validation is used for G3
    calibration/thresholds; unused for L0–G2 ranking metrics.
    """
    y = pd.Series(np.asarray(y).astype(int), index=X.index)
    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=seed,
    )
    rows: list[dict] = []

    for fold_id, (train_idx, test_idx) in enumerate(rskf.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

        X_fit, X_val, y_fit, y_val = train_test_split(
            X_train,
            y_train,
            test_size=inner_val_size,
            stratify=y_train,
            random_state=seed + fold_id,
        )

        for arm in arms:
            scores, fit_pos, extras = fit_predict_scores(
                arm,
                X_fit,
                y_fit,
                X_test,
                num_cols,
                cat_cols,
                seed=seed + fold_id,
                X_val=X_val if arm == "G3" else None,
                y_val=y_val if arm == "G3" else None,
            )
            # Map fit positions back to original row labels / outer indices
            fit_original_labels = X_fit.index.to_numpy()[fit_pos]
            # Also expose positional indices into the full X for leakage tests
            fit_ilocs = X.index.get_indexer(fit_original_labels)

            if leakage_callback is not None:
                leakage_callback(
                    fold_id=fold_id,
                    arm=arm,
                    test_idx=np.asarray(test_idx),
                    fit_idx=fit_ilocs,
                    train_idx=np.asarray(train_idx),
                )

            metrics = compute_metrics(y_test.to_numpy(), scores)
            row = {
                "fold": fold_id,
                "arm": arm,
                "n_test": int(len(y_test)),
                "n_fit": int(len(y_fit)),
                "fraud_rate_test": float(y_test.mean()),
                **metrics,
            }
            if "threshold" in extras:
                row["threshold"] = extras["threshold"]
            rows.append(row)

    return pd.DataFrame(rows)


def paired_diffs_table(
    folds_df: pd.DataFrame,
    metric: str = "pr_auc",
    baseline: str = "G0",
    n_boot: int = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    """Paired mean differences vs baseline across folds with bootstrap CIs."""
    out = []
    base = folds_df.loc[folds_df["arm"] == baseline].set_index("fold")[metric]
    for arm in sorted(folds_df["arm"].unique()):
        if arm == baseline:
            continue
        other = folds_df.loc[folds_df["arm"] == arm].set_index("fold")[metric]
        aligned = pd.concat([other, base], axis=1, keys=["arm", "base"]).dropna()
        diffs = (aligned["arm"] - aligned["base"]).to_numpy()
        mean, lo, hi = paired_bootstrap_ci(diffs, n_boot=n_boot, seed=seed)
        out.append(
            {
                "metric": metric,
                "arm": arm,
                "baseline": baseline,
                "mean_diff": mean,
                "ci_low": lo,
                "ci_high": hi,
                "n_folds": int(len(diffs)),
                "note": "folds are correlated; CI is approximate",
            }
        )
    return pd.DataFrame(out)

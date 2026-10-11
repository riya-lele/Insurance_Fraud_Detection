"""Hypothetical cost layer on calibrated scores (c=1, L in {5,10,20,50})."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.evaluate import K_FRACS, precision_recall_at_k
from src.models import f1_threshold

INVESTIGATION_COST = 1.0
LOSS_RATIOS = (5, 10, 20, 50)
COST_NOTE = "hypothetical unit costs; not real dollars"


def expected_cost_per_claim(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    c: float = INVESTIGATION_COST,
    L: float = 10.0,
) -> float:
    """c per investigation + L per missed fraud; mean over claims."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    investigate = y_pred.astype(float)
    missed = ((y_true == 1) & (y_pred == 0)).astype(float)
    return float(np.mean(c * investigate + L * missed))


def bayes_threshold(c: float, L: float) -> float:
    """Investigate when p > c / (L + c)."""
    return float(c / (L + c))


def min_cost_threshold(
    y_val: np.ndarray,
    scores_val: np.ndarray,
    c: float,
    L: float,
) -> float:
    """Empirical threshold on validation minimising expected cost."""
    y_val = np.asarray(y_val).astype(int)
    scores_val = np.asarray(scores_val, dtype=float)
    candidates = np.unique(scores_val)
    if len(candidates) == 0:
        return bayes_threshold(c, L)
    best_t, best_cost = bayes_threshold(c, L), np.inf
    for t in candidates:
        pred = (scores_val >= t).astype(int)
        cost = expected_cost_per_claim(y_val, pred, c=c, L=L)
        if cost < best_cost:
            best_cost, best_t = cost, float(t)
    return best_t


def apply_rule(
    rule: str,
    scores: np.ndarray,
    *,
    y_val: np.ndarray | None = None,
    scores_val: np.ndarray | None = None,
    c: float = INVESTIGATION_COST,
    L: float = 10.0,
    fixed_threshold: float | None = None,
) -> tuple[np.ndarray, float]:
    """Return (binary predictions on scores, threshold used)."""
    scores = np.asarray(scores, dtype=float)

    if rule == "threshold_0.5":
        t = 0.5
    elif rule == "f1_optimal":
        if y_val is None or scores_val is None:
            raise ValueError("f1_optimal needs validation labels/scores")
        t = f1_threshold(y_val, scores_val)
    elif rule == "cost_bayes":
        t = bayes_threshold(c, L)
    elif rule == "cost_optimal_val":
        if y_val is None or scores_val is None:
            raise ValueError("cost_optimal_val needs validation labels/scores")
        # Bayes cut, verified/replaced by empirical min-cost on validation
        t = min_cost_threshold(y_val, scores_val, c=c, L=L)
    elif rule == "investigate_none":
        return np.zeros(len(scores), dtype=int), 1.0
    elif rule == "investigate_all":
        return np.ones(len(scores), dtype=int), 0.0
    elif rule == "fixed":
        if fixed_threshold is None:
            raise ValueError("fixed rule requires fixed_threshold")
        t = float(fixed_threshold)
    else:
        raise ValueError(f"Unknown rule {rule!r}")

    return (scores >= t).astype(int), float(t)


def cost_table_for_scores(
    y_test: np.ndarray,
    scores_test: np.ndarray,
    y_val: np.ndarray,
    scores_val: np.ndarray,
    *,
    method: str,
    fold: int,
    c: float = INVESTIGATION_COST,
    losses: tuple[int, ...] = LOSS_RATIOS,
) -> pd.DataFrame:
    """Per-fold expected cost under each rule and L (hypothetical)."""
    rows = []
    rules = (
        "threshold_0.5",
        "f1_optimal",
        "cost_bayes",
        "cost_optimal_val",
        "investigate_none",
        "investigate_all",
    )
    for L in losses:
        for rule in rules:
            pred, t = apply_rule(
                rule,
                scores_test,
                y_val=y_val,
                scores_val=scores_val,
                c=c,
                L=float(L),
            )
            rows.append(
                {
                    "fold": fold,
                    "method": method,
                    "rule": rule,
                    "L": L,
                    "c": c,
                    "threshold": t,
                    "expected_cost_per_claim": expected_cost_per_claim(
                        y_test, pred, c=c, L=float(L)
                    ),
                    "f1_test": float(f1_score(y_test, pred, zero_division=0)),
                    "note": COST_NOTE,
                }
            )
    return pd.DataFrame(rows)


def topk_curve(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    method: str,
    fold: int,
    fracs: tuple[float, ...] = K_FRACS,
) -> pd.DataFrame:
    """Precision/Recall at capacity fractions (investigation budget)."""
    rows = []
    y_true = np.asarray(y_true).astype(int)
    for frac in fracs:
        p, r = precision_recall_at_k(y_true, scores, frac)
        rows.append(
            {
                "fold": fold,
                "method": method,
                "k_frac": frac,
                "k_pct": int(frac * 100),
                "precision": p,
                "recall": r,
                "note": COST_NOTE,
            }
        )
    return pd.DataFrame(rows)

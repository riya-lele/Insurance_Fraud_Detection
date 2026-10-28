"""Cost helpers for the Optuna extension. The original src/costs.py is left untouched."""
from __future__ import annotations

import numpy as np

from src.opt_common import k_count


def bayes_threshold_existing(L: float, c: float = 1.0) -> float:
    """The project's original rule: c/(L+c). Optimal only if a correctly flagged fraud costs 0."""
    return c / (L + c)


def bayes_threshold_matched(L: float, c: float = 1.0) -> float:
    """Optimal when every flagged claim costs c and a missed fraud costs L: flag if p*L > c."""
    return c / L


def cost_summary(y, flag, L: float, c: float = 1.0) -> dict:
    y = np.asarray(y).astype(int)
    flag = np.asarray(flag).astype(bool)
    tp = int((flag & (y == 1)).sum())
    fp = int((flag & (y == 0)).sum())
    fn = int((~flag & (y == 1)).sum())
    total = c * (tp + fp) + L * fn
    return dict(
        total_cost=float(total), cost_per_claim=float(total / len(y)), flagged=tp + fp,
        true_positives=tp, false_positives=fp, missed_fraud=fn,
        precision=tp / (tp + fp) if tp + fp else float("nan"),
        recall=tp / (tp + fn) if tp + fn else float("nan"),
    )


def tune_threshold(y, s, L: float, c: float = 1.0) -> float:
    """Raw-score cut-off minimising cost on the calibration partition (flag if s >= threshold).
    Returns inf when flagging nothing is cheapest."""
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    order = np.argsort(-s, kind="stable")
    ss, yy = s[order], y[order]
    n, total = len(y), int(y.sum())
    cum_tp = np.cumsum(yy)
    last = np.r_[ss[1:] != ss[:-1], True]            # cut only between distinct scores
    j = np.arange(1, n + 1)[last]
    cost = (c * j + L * (total - cum_tp[last])) / n
    if (L * total / n) <= cost.min():
        return float("inf")
    return float(ss[last][int(np.argmin(cost))])


def top_k_flags(score, frac: float):
    score = np.asarray(score, dtype=float)
    n = len(score)
    k = k_count(n, frac)
    order = np.argsort(-score, kind="stable")
    flag = np.zeros(n, dtype=bool)
    flag[order[:k]] = True
    return flag
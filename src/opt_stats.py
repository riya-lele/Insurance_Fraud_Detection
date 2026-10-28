from __future__ import annotations

import numpy as np
from scipy import stats


def nadeau_bengio(a, b, n_train: float, n_test: float) -> dict:
    """Corrected resampled paired t-test on aligned fold-level values (a - b)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired arrays must have the same length")
    d = a - b
    k = len(d)
    if k < 3:
        raise ValueError("need at least 3 paired folds")
    mean, sd = float(d.mean()), float(d.std(ddof=1))
    out = dict(n_pairs=k, mean_diff=mean, sd_diff=sd, a_higher_in=int((d > 0).sum()),
               cohens_dz=mean / sd if sd > 0 else float("nan"))
    if sd == 0:                                    # degenerate: no variation, no valid test
        out.update(se_corrected=float("nan"), t=float("nan"), df=k - 1, p_raw=float("nan"),
                   ci_low=float("nan"), ci_high=float("nan"))
        return out
    se = float(np.sqrt((1.0 / k + n_test / n_train) * sd ** 2))
    t = mean / se
    half = float(stats.t.ppf(0.975, k - 1) * se)
    out.update(se_corrected=se, t=float(t), df=k - 1, p_raw=float(2 * stats.t.sf(abs(t), k - 1)),
               ci_low=mean - half, ci_high=mean + half)
    return out


def paired_from_frame(df, metric, a, b, model_col="model", fold_col="fold"):
    """Align two models by fold id and drop folds missing for either."""
    piv = df.pivot(index=fold_col, columns=model_col, values=metric).dropna(subset=[a, b])
    return piv[a].to_numpy(), piv[b].to_numpy(), piv.index.to_numpy()


def holm(pvals) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    adj = np.full(len(p), np.nan)
    valid = np.where(~np.isnan(p))[0]
    order = valid[np.argsort(p[valid])]
    m, run = len(valid), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * p[i]))
        adj[i] = run
    return adj
import json
import sys
import time
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src.data import load_dataset
from src.opt_common import MODELS, OUTER_SEED, evaluate_scores, make_splits

QUICK = "--quick" in sys.argv
n_splits, n_repeats = (3, 1) if QUICK else (5, 3)
res_dir = ROOT / "results"
(res_dir / "figures").mkdir(parents=True, exist_ok=True)

X, y, num_cols, cat_cols = load_dataset(verbose=False)
X = X.reset_index(drop=True)
y = np.asarray(y).astype(int)
print(f"{len(y)} claims, {int(y.sum())} frauds; folds: {n_splits} x {n_repeats}; models: {list(MODELS)}")

rows, sel_rows, saved = [], [], {}
for sp in make_splits(y, n_splits, n_repeats):
    f = sp["fold"]
    for key in ("fit", "cal", "test", "fit_train", "fitness"):
        saved[f"fold{f}_{key}"] = sp[key]
    Xfit, yfit = X.iloc[sp["fit"]], y[sp["fit"]]
    Xcal, ycal = X.iloc[sp["cal"]], y[sp["cal"]]
    Xte, yte = X.iloc[sp["test"]], y[sp["test"]]
    Xft, yft = X.iloc[sp["fit_train"]], y[sp["fit_train"]]
    Xfv, yfv = X.iloc[sp["fitness"]], y[sp["fitness"]]

    for name, fit in MODELS.items():
        seed = OUTER_SEED + f
        # (a) selection score: fit on fit_train, score the internal fitness split only
        score_fn = fit(Xft, yft, num_cols, cat_cols, seed)
        sel_rows.append(dict(fold=f, model=name,
                             fitness_pr_auc=float(average_precision_score(yfv, score_fn(Xfv)))))
        # (b) reported score: fit on the full fit partition, evaluate on calibration and test
        t0 = time.perf_counter()
        score_fn = fit(Xfit, yfit, num_cols, cat_cols, seed)
        fit_s = time.perf_counter() - t0
        s_cal = score_fn(Xcal)
        t0 = time.perf_counter()
        s_te = score_fn(Xte)
        inf_s = time.perf_counter() - t0
        m = evaluate_scores(yte, s_te, ycal, s_cal)
        m.update(fold=f, model=name, n_test=len(yte), fit_seconds=fit_s, inference_seconds=inf_s)
        rows.append(m)
        print(f"fold {f:2d} {name:4s} PR-AUC {m['pr_auc']:.4f}  P@5% {m['precision_at_5pct']:.3f}  "
              f"fit {fit_s:5.1f}s", flush=True)

suffix = "_quick" if QUICK else ""
res = pd.DataFrame(rows)
sel = pd.DataFrame(sel_rows)
res.to_csv(res_dir / f"classifier_fold_metrics{suffix}.csv", index=False)
sel.to_csv(res_dir / f"classifier_selection_folds{suffix}.csv", index=False)
np.savez_compressed(res_dir / f"classifier_fold_splits{suffix}.npz", **saved)

metric_cols = ["pr_auc", "roc_auc", "brier_calibrated", "f1_at_cal_threshold",
               "precision_at_1pct", "precision_at_2pct", "precision_at_5pct", "precision_at_10pct",
               "recall_at_1pct", "recall_at_2pct", "recall_at_5pct", "recall_at_10pct",
               "fit_seconds", "inference_seconds"]
summary = res.groupby("model")[metric_cols].agg(["mean", "std"]).reindex(list(MODELS))
summary.columns = [f"{a}_{b}" for a, b in summary.columns]
summary["fitness_pr_auc_mean"] = sel.groupby("model")["fitness_pr_auc"].mean()
summary.to_csv(res_dir / f"classifier_comparison{suffix}.csv")

chosen = str(sel.groupby("model")["fitness_pr_auc"].mean().idxmax())
info = dict(
    chosen_by_internal_fitness_pr_auc=chosen,
    selection_rule="highest mean PR-AUC on the internal fitness split (fit partition only)",
    outer_seed=OUTER_SEED, n_splits=n_splits, n_repeats=n_repeats, quick=QUICK,
    versions={p: version(p) for p in ["numpy", "pandas", "scikit-learn", "lightgbm", "xgboost", "catboost"]},
)
(res_dir / f"classifier_selection{suffix}.json").write_text(json.dumps(info, indent=2))

fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
for ax, col, title in zip(axes, ["pr_auc", "precision_at_5pct"], ["PR-AUC", "Precision@5%"]):
    ax.bar(summary.index, summary[f"{col}_mean"], yerr=summary[f"{col}_std"], capsize=4, color="#4C78A8")
    ax.set_title(title + " (mean \u00b1 SD over outer folds)")
fig.tight_layout()
fig.savefig(res_dir / "figures" / f"classifier_comparison{suffix}.png", dpi=200)

print("\n" + summary[["pr_auc_mean", "pr_auc_std", "precision_at_5pct_mean",
                      "fitness_pr_auc_mean", "fit_seconds_mean"]].round(4).to_string())
print("\nchosen candidate (selection split):", chosen)
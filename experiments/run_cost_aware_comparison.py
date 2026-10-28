import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from src.data import load_dataset
from src.opt_common import KS, make_splits
from src.opt_costs import (bayes_threshold_existing, bayes_threshold_matched, cost_summary,
                           top_k_flags, tune_threshold)

QUICK = "--quick" in sys.argv
n_splits, n_repeats = (3, 1) if QUICK else (5, 3)
suffix = "_quick" if QUICK else ""
C, LOSSES = 1.0, (5, 10, 20, 50)
res = ROOT / "results"

_, y, _, _ = load_dataset(verbose=False)
y = np.asarray(y).astype(int)
Z = np.load(res / f"optuna_fold_scores{suffix}.npz")

cost_rows, cap_rows = [], []
for sp in make_splits(y, n_splits, n_repeats):
    f = sp["fold"]
    assert np.array_equal(Z[f"f{f}_cal_idx"], sp["cal"]) and np.array_equal(Z[f"f{f}_test_idx"], sp["test"]), \
        "saved scores do not match the regenerated folds"
    ycal, yte = y[sp["cal"]], y[sp["test"]]
    n = len(yte)
    for model in ("G0", "OPT"):
        s_cal, s_te = Z[f"f{f}_{model}_cal"], Z[f"f{f}_{model}_test"]
        # calibrator and thresholds use the calibration partition only
        p_te = IsotonicRegression(out_of_bounds="clip").fit(s_cal, ycal).predict(s_te)
        for L in LOSSES:
            t_raw = tune_threshold(ycal, s_cal, L, C)
            t_ex, t_ma = bayes_threshold_existing(L, C), bayes_threshold_matched(L, C)
            policies = {
                "investigate_none": (np.zeros(n, bool), np.nan),
                "investigate_all": (np.ones(n, bool), np.nan),
                "threshold_0.5": (p_te >= 0.5, 0.5),
                "bayes_existing": (p_te >= t_ex, t_ex),          # c/(L+c), calibrated probability
                "bayes_matched": (p_te >= t_ma, t_ma),           # c/L, calibrated probability
                "val_tuned_raw": (s_te >= t_raw, t_raw),         # raw-score cut-off from calibration data
            }
            for pol, (flag, thr) in policies.items():
                cost_rows.append(dict(fold=f, model=model, L=L, policy=pol, threshold=thr,
                                      **cost_summary(yte, flag, L, C)))
        for frac in KS:
            flag = top_k_flags(s_te, frac)                       # ranked by the raw score
            for L in LOSSES:
                cap_rows.append(dict(fold=f, model=model, capacity_pct=int(round(frac * 100)), L=L,
                                     k=int(flag.sum()), **cost_summary(yte, flag, L, C)))

cost = pd.DataFrame(cost_rows)
cap = pd.DataFrame(cap_rows)
cost.to_csv(res / f"cost_fold_results{suffix}.csv", index=False)
cap.to_csv(res / f"fixed_capacity_fold_results{suffix}.csv", index=False)

agg = cost.groupby(["model", "L", "policy"]).agg(
    cost_per_claim_mean=("cost_per_claim", "mean"), cost_per_claim_std=("cost_per_claim", "std"),
    total_cost_mean=("total_cost", "mean"), flagged_mean=("flagged", "mean"),
    false_positives_mean=("false_positives", "mean"), missed_fraud_mean=("missed_fraud", "mean"),
    precision_mean=("precision", "mean"), recall_mean=("recall", "mean"),
    threshold_median=("threshold", "median")).reset_index()
agg.to_csv(res / f"cost_comparison{suffix}.csv", index=False)

capagg = cap.groupby(["model", "capacity_pct", "L"]).agg(
    k_mean=("k", "mean"), caught_mean=("true_positives", "mean"),
    precision_mean=("precision", "mean"), precision_std=("precision", "std"),
    recall_mean=("recall", "mean"), recall_std=("recall", "std"),
    cost_per_claim_mean=("cost_per_claim", "mean"), cost_per_claim_std=("cost_per_claim", "std")).reset_index()
capagg.to_csv(res / f"fixed_capacity_comparison{suffix}.csv", index=False)

# ---- figures ----
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
for ax, pol in zip(axes, ["bayes_existing", "bayes_matched", "val_tuned_raw"]):
    for model, col in (("G0", "#4C78A8"), ("OPT", "#E45756")):
        d = agg[(agg.model == model) & (agg.policy == pol)].sort_values("L")
        ax.errorbar(d["L"], d["cost_per_claim_mean"], yerr=d["cost_per_claim_std"],
                    marker="o", capsize=3, color=col, label=model)
    for pol2, ls in (("investigate_none", "--"), ("investigate_all", ":")):
        d = agg[(agg.model == "G0") & (agg.policy == pol2)].sort_values("L")
        ax.plot(d["L"], d["cost_per_claim_mean"], ls, color="grey", label=pol2.replace("_", " "))
    ax.set_title(pol); ax.set_xlabel("L (assumed cost of a missed fraud)")
axes[0].set_ylabel("Expected cost per claim (hypothetical)")
axes[0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(res / "figures" / f"cost_vs_loss{suffix}.png", dpi=200)

cg = cap[cap.L == LOSSES[0]].groupby(["model", "capacity_pct"]).agg(
    p=("precision", "mean"), p_sd=("precision", "std"), r=("recall", "mean"), r_sd=("recall", "std")).reset_index()
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
for ax, (m, sd, title) in zip(axes, [("p", "p_sd", "Precision@K"), ("r", "r_sd", "Recall@K")]):
    for i, (model, col) in enumerate((("G0", "#4C78A8"), ("OPT", "#E45756"))):
        d = cg[cg.model == model].sort_values("capacity_pct")
        ax.bar(np.arange(len(d)) + i * 0.38, d[m], 0.38, yerr=d[sd], capsize=3, color=col, label=model)
    ax.set_xticks(np.arange(len(d)) + 0.19)
    ax.set_xticklabels([f"{k}%" for k in sorted(cg.capacity_pct.unique())])
    ax.set_xlabel("Investigation capacity"); ax.set_title(title)
axes[0].legend()
fig.tight_layout(); fig.savefig(res / "figures" / f"fixed_capacity_comparison{suffix}.png", dpi=200)

print(agg[agg.L == 10][["model", "policy", "cost_per_claim_mean", "flagged_mean", "false_positives_mean",
                        "missed_fraud_mean", "threshold_median"]].round(4).to_string(index=False))
print("\nCosts are hypothetical: investigating a claim costs 1, a missed fraud costs L.")
print("Compare bayes_existing (c/(L+c)) with bayes_matched (c/L) before changing any threshold in the report.")
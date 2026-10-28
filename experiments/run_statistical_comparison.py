import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.data import load_dataset
from src.opt_common import make_splits
from src.opt_stats import holm, nadeau_bengio, paired_from_frame

QUICK = "--quick" in sys.argv
n_splits, n_repeats = (3, 1) if QUICK else (5, 3)
suffix = "_quick" if QUICK else ""
res = ROOT / "results"

_, y, _, _ = load_dataset(verbose=False)
splits = list(make_splits(np.asarray(y).astype(int), n_splits, n_repeats))
n_test = float(np.mean([len(s["test"]) for s in splits]))
n_fit = float(np.mean([len(s["fit"]) for s in splits]))
print(f"{len(splits)} folds; n_test={n_test:.0f}, n_train (fit partition)={n_fit:.0f}; "
      f"correction factor 1/k + n_test/n_train = {1 / len(splits) + n_test / n_fit:.3f}")

opt = pd.read_csv(res / f"optimization_fold_metrics{suffix}.csv")
clf = pd.read_csv(res / f"classifier_fold_metrics{suffix}.csv")
cost = pd.read_csv(res / f"cost_fold_results{suffix}.csv")

jobs = []   # (family, frame, metric, label, model_a, model_b, higher_is_better)
for m in ["pr_auc", "roc_auc", "precision_at_1pct", "precision_at_2pct", "precision_at_5pct",
          "precision_at_10pct", "recall_at_5pct", "recall_at_10pct"]:
    jobs.append(("OPT_vs_G0_ranking", opt, m, m, "OPT", "G0", True))
jobs.append(("OPT_vs_G0_ranking", opt, "brier_calibrated", "brier_calibrated", "OPT", "G0", False))
for L in (5, 10, 20, 50):
    for pol in ("bayes_existing", "bayes_matched", "val_tuned_raw"):
        sub = cost[(cost["L"] == L) & (cost["policy"] == pol)]
        jobs.append(("OPT_vs_G0_cost", sub, "cost_per_claim", f"cost_per_claim|L={L}|{pol}", "OPT", "G0", False))
for other in ("LR", "XGB", "CAT"):
    for m in ("pr_auc", "precision_at_5pct"):
        jobs.append(("CLASSIFIERS_vs_G0", clf, m, m, other, "G0", True))

rows, obs = [], []
for fam, frame, metric, label, a_name, b_name, hib in jobs:
    a, b, folds = paired_from_frame(frame, metric, a_name, b_name)
    r = nadeau_bengio(a, b, n_fit, n_test)
    better = ("neither" if r["mean_diff"] == 0 else
              a_name if (r["mean_diff"] > 0) == hib else b_name)
    rows.append(dict(family=fam, metric=label, model_a=a_name, model_b=b_name,
                     higher_is_better=hib, direction_favours=better, **r))
    obs += [dict(family=fam, metric=label, fold=int(f), model_a=a_name, a_value=float(x),
                 model_b=b_name, b_value=float(z)) for f, x, z in zip(folds, a, b)]

out = pd.DataFrame(rows)
out["p_holm"] = out.groupby("family")["p_raw"].transform(lambda s: holm(s.to_numpy()))
out["significant_holm_0.05"] = out["p_holm"] < 0.05
out.to_csv(res / f"statistical_comparisons{suffix}.csv", index=False)
pd.DataFrame(obs).to_csv(res / f"paired_observations{suffix}.csv", index=False)

show = ["family", "metric", "model_a", "model_b", "mean_diff", "ci_low", "ci_high", "cohens_dz",
        "p_raw", "p_holm", "direction_favours"]
print(out[show].round(4).to_string(index=False))
print("\nHolm adjusts within each family. The correction is approximate for repeated k-fold, "
      "and 15 correlated folds give limited power: a large p-value does not prove two models are equal.")
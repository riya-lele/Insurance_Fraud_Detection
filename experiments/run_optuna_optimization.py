import argparse
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
import optuna
import pandas as pd
from sklearn.metrics import average_precision_score

from src.data import load_dataset
from src.models import _fit_g0_raw, transform_ordinal
from src.opt_common import N_JOBS, OUTER_SEED, evaluate_scores, make_splits
from src.opt_lgbm import fit_lgbm, tune_fold

optuna.logging.set_verbosity(optuna.logging.WARNING)

ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true", help="3 folds x 1 repeat, 10 trials")
ap.add_argument("--trials", type=int, default=None, help="Optuna trials per fold (default 30, quick 10)")
ap.add_argument("--timeout", type=float, default=None, help="max seconds of tuning per fold")
ap.add_argument("--yes", action="store_true", help="do not ask for confirmation")
args = ap.parse_args()

QUICK = args.quick
n_splits, n_repeats = (3, 1) if QUICK else (5, 3)
n_trials = args.trials or (10 if QUICK else 30)
suffix = "_quick" if QUICK else ""
res = ROOT / "results"
(res / "figures").mkdir(parents=True, exist_ok=True)

sel_file = res / f"classifier_selection{suffix}.json"
if sel_file.exists():
    chosen = json.loads(sel_file.read_text())["chosen_by_internal_fitness_pr_auc"]
    print(f"Experiment 1 selected: {chosen}")
    if chosen != "G0":
        print("NOTE: the selected candidate is not LightGBM. This script still tunes LightGBM "
              "(the brief's first-implementation scope). State that choice in the report.")

X, y, num_cols, cat_cols = load_dataset(verbose=False)
X = X.reset_index(drop=True)
y = np.asarray(y).astype(int)
splits = list(make_splits(y, n_splits, n_repeats))


def prep(Xa, ya, seed):
    """Existing G0 pipeline: returns (fitted ordinal encoder, G0 model, categorical indices)."""
    return _fit_g0_raw(Xa, ya, num_cols, cat_cols, seed)


def enc(e, Xa):
    return transform_ordinal(e, Xa, num_cols, cat_cols)


def audit(sp):
    s = {k: set(sp[k].tolist()) for k in ("fit", "cal", "test", "fit_train", "fitness")}
    assert not (s["fit"] & s["cal"]) and not (s["fit"] & s["test"]) and not (s["cal"] & s["test"])
    assert not (s["fit_train"] & s["fitness"]) and (s["fit_train"] | s["fitness"]) == s["fit"]
    assert not (s["fitness"] & (s["cal"] | s["test"]))
    assert not (s["fit_train"] & (s["cal"] | s["test"]))
    return dict(fold=sp["fold"], n_fit_train=len(sp["fit_train"]), n_fitness=len(sp["fitness"]),
                n_cal=len(sp["cal"]), n_test=len(sp["test"]), overlaps=0)


# ---- pilot: time one fit to estimate the total runtime ----
sp0 = splits[0]
Xft0, yft0 = X.iloc[sp0["fit_train"]], y[sp0["fit_train"]]
e0, g00, cat0 = prep(Xft0, yft0, OUTER_SEED)
A0 = enc(e0, Xft0)
t0 = time.perf_counter()
fit_lgbm(A0, yft0, g00.get_params(),
         dict(num_leaves=31, learning_rate=0.05, n_estimators=300, min_child_samples=20,
              feature_fraction=0.8, reg_lambda=0.001), cat0, OUTER_SEED, N_JOBS)
pilot = time.perf_counter() - t0
est = pilot * (n_trials + 3) * len(splits)
print(f"Pilot fit {pilot:.1f}s. {n_trials} trials x {len(splits)} folds -> roughly {est / 60:.0f} min "
      f"(rough: trials with more trees or leaves can be slower; N_JOBS={N_JOBS}).")
if not args.yes:
    if input("Press Enter to continue, or type n to stop: ").strip().lower().startswith("n"):
        sys.exit("stopped by user")

trial_frames, best_rows, metric_rows, audit_rows, imps = [], [], [], [], []
scores = {}
t_all = time.perf_counter()

for sp in splits:
    f, seed = sp["fold"], OUTER_SEED + sp["fold"]
    audit_rows.append(audit(sp))

    # ---- tuning: fit_train for fitting, fitness split for scoring. Nothing else. ----
    Xft, yft = X.iloc[sp["fit_train"]], y[sp["fit_train"]]
    Xfv, yfv = X.iloc[sp["fitness"]], y[sp["fitness"]]
    e_t, g0_t, cat_t = prep(Xft, yft, seed)
    A, V = enc(e_t, Xft), enc(e_t, Xfv)
    g0_fitness = float(average_precision_score(yfv, g0_t.predict_proba(V)[:, 1]))

    t0 = time.perf_counter()
    study = tune_fold(A, yft, V, yfv, g0_t.get_params(), cat_t, seed, n_trials, args.timeout, N_JOBS)
    tune_s = time.perf_counter() - t0

    states = pd.Series([t.state.name for t in study.trials]).value_counts().to_dict()
    df = study.trials_dataframe()
    df.insert(0, "fold", f)
    trial_frames.append(df)
    best_rows.append(dict(fold=f, best_fitness_pr_auc=study.best_value, g0_fitness_pr_auc=g0_fitness,
                          n_trials_run=len(study.trials), completed=states.get("COMPLETE", 0),
                          failed=states.get("FAIL", 0), pruned=states.get("PRUNED", 0),
                          tune_seconds=tune_s, **{f"param_{k}": v for k, v in study.best_params.items()}))
    try:
        import optuna.importance as oi
        imps.append(oi.get_param_importances(study))
    except Exception as e:                                 # report it, do not hide it
        print(f"  (parameter importance skipped for fold {f}: {e})")

    # ---- refit on the full fit partition; calibration and test are used only from here ----
    Xfit, yfit = X.iloc[sp["fit"]], y[sp["fit"]]
    Xcal, ycal = X.iloc[sp["cal"]], y[sp["cal"]]
    Xte, yte = X.iloc[sp["test"]], y[sp["test"]]
    t0 = time.perf_counter()
    e_f, g0_f, cat_f = prep(Xfit, yfit, seed)
    g0_fit_s = time.perf_counter() - t0
    Bfit, Bcal, Bte = enc(e_f, Xfit), enc(e_f, Xcal), enc(e_f, Xte)
    t0 = time.perf_counter()
    opt = fit_lgbm(Bfit, yfit, g0_f.get_params(), study.best_params, cat_f, seed, N_JOBS)
    opt_fit_s = time.perf_counter() - t0

    for name, model, fit_s in (("G0", g0_f, g0_fit_s), ("OPT", opt, opt_fit_s)):
        s_cal = model.predict_proba(Bcal)[:, 1]
        t0 = time.perf_counter()
        s_te = model.predict_proba(Bte)[:, 1]
        inf_s = time.perf_counter() - t0
        mt = evaluate_scores(yte, s_te, ycal, s_cal)
        mt.update(fold=f, model=name, fit_seconds=fit_s, inference_seconds=inf_s, n_test=len(yte))
        metric_rows.append(mt)
        scores[f"f{f}_{name}_cal"] = s_cal
        scores[f"f{f}_{name}_test"] = s_te
    scores[f"f{f}_cal_idx"] = sp["cal"]
    scores[f"f{f}_test_idx"] = sp["test"]

    g, o = metric_rows[-2], metric_rows[-1]
    print(f"fold {f:2d}  fitness PR-AUC G0 {g0_fitness:.4f} -> best {study.best_value:.4f} | "
          f"outer-test PR-AUC G0 {g['pr_auc']:.4f} vs OPT {o['pr_auc']:.4f} | tuning {tune_s:.0f}s", flush=True)

total_s = time.perf_counter() - t_all

# ---- save ----
trials = pd.concat(trial_frames, ignore_index=True)
best = pd.DataFrame(best_rows)
mdf = pd.DataFrame(metric_rows)
trials.to_csv(res / f"optuna_trials{suffix}.csv", index=False)
best.to_csv(res / f"optuna_best_params{suffix}.csv", index=False)
mdf.to_csv(res / f"optimization_fold_metrics{suffix}.csv", index=False)
pd.DataFrame(audit_rows).to_csv(res / f"optimization_split_audit{suffix}.csv", index=False)
np.savez_compressed(res / f"optuna_fold_scores{suffix}.npz", **scores)

cols = ["pr_auc", "roc_auc", "brier_calibrated", "f1_at_cal_threshold",
        "precision_at_1pct", "precision_at_2pct", "precision_at_5pct", "precision_at_10pct",
        "recall_at_1pct", "recall_at_2pct", "recall_at_5pct", "recall_at_10pct",
        "fit_seconds", "inference_seconds"]
summ = mdf.groupby("model")[cols].agg(["mean", "std"])
summ.columns = [f"{a}_{b}" for a, b in summ.columns]
piv = mdf.pivot(index="fold", columns="model", values=cols)
summ.loc["OPT_minus_G0", :] = np.nan
for c in cols:
    summ.loc["OPT_minus_G0", f"{c}_mean"] = float((piv[(c, "OPT")] - piv[(c, "G0")]).mean())
summ.to_csv(res / f"optimization_comparison{suffix}.csv")

(res / f"optuna_run_info{suffix}.json").write_text(json.dumps(dict(
    trials_per_fold=n_trials, timeout_per_fold=args.timeout, sampler="TPESampler",
    sampler_seed_base=OUTER_SEED, objective="average_precision on the internal fitness split",
    pruning="none", n_splits=n_splits, n_repeats=n_repeats, n_jobs_lightgbm=N_JOBS, optuna_n_jobs=1,
    total_seconds=total_s, quick=QUICK,
    versions={p: version(p) for p in ["numpy", "pandas", "scikit-learn", "lightgbm", "optuna"]}), indent=2))

# ---- figures ----
fig, ax = plt.subplots(figsize=(6, 4))
curves = []
for _, d in trials.groupby("fold"):
    v = d.sort_values("number")["value"].cummax().to_numpy()
    ax.plot(range(1, len(v) + 1), v, alpha=0.3, color="#4C78A8")
    curves.append(v)
L_ = min(len(c) for c in curves)
ax.plot(range(1, L_ + 1), np.mean([c[:L_] for c in curves], axis=0), color="#E45756", lw=2, label="mean over folds")
ax.set_xlabel("Trial"); ax.set_ylabel("Best internal-fitness PR-AUC so far"); ax.legend()
fig.tight_layout(); fig.savefig(res / "figures" / f"optuna_optimization_history{suffix}.png", dpi=200)

if imps:
    imp = pd.DataFrame(imps).mean().sort_values()
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.barh(imp.index, imp.values, color="#4C78A8")
    ax.set_xlabel("Mean importance over folds (fANOVA)")
    fig.tight_layout(); fig.savefig(res / "figures" / f"optuna_parameter_importance{suffix}.png", dpi=200)

fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
for ax, col, title in zip(axes, ["pr_auc", "precision_at_5pct"], ["PR-AUC", "Precision@5%"]):
    p = mdf.pivot(index="fold", columns="model", values=col)
    for _, r in p.iterrows():
        ax.plot([0, 1], [r["G0"], r["OPT"]], color="grey", alpha=0.4)
    ax.plot([0, 1], [p["G0"].mean(), p["OPT"].mean()], "o-", color="#E45756", lw=2.5)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["G0", "Optuna LightGBM"])
    ax.set_title(f"{title} (each grey line = one outer fold)")
fig.tight_layout(); fig.savefig(res / "figures" / f"optimization_comparison{suffix}.png", dpi=200)

print("\n" + summ[["pr_auc_mean", "pr_auc_std", "precision_at_5pct_mean", "recall_at_5pct_mean"]].round(4).to_string())
print(f"\nTotal {total_s / 60:.1f} min. Mean tuning gain on the internal fitness split: "
      f"{(best['best_fitness_pr_auc'] - best['g0_fitness_pr_auc']).mean():+.4f} "
      "(optimistic: it is the winner of many trials on the same split).")
print("Outer-test differences are tested properly in Step 7. Do not call OPT better from this table alone.")
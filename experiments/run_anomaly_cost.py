"""G3 + anomaly combinations and hypothetical cost layer (--quick supported)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.anomaly import (
    choose_rank_weight,
    full_anomaly_feature,
    oof_anomaly_feature,
    rank_average,
    scores_if_all,
    scores_if_normal,
    stack_fit_predict,
)
from src.costs import cost_table_for_scores, topk_curve
from src.data import load_dataset
from src.evaluate import compute_metrics, paired_diffs_table
from src.models import fit_predict_scores


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick", action="store_true", help="3 folds x 1 repeat")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path, default=ROOT / "results")
    return p.parse_args(argv)


def _calibrate(
    raw_val: np.ndarray,
    y_val: np.ndarray,
    raw_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    cal = IsotonicRegression(out_of_bounds="clip")
    cal.fit(raw_val, y_val)
    return cal.predict(raw_val), cal.predict(raw_test)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    n_splits, n_repeats = (3, 1) if args.quick else (5, 3)
    config = {
        "n_splits": n_splits,
        "n_repeats": n_repeats,
        "seed": args.seed,
        "quick": args.quick,
        "supervised_base": "G0",
        "anomaly_modes": ("IF_all", "IF_normal"),
    }
    print("config:", config)

    X, y, num_cols, cat_cols = load_dataset(verbose=True)
    y = pd.Series(np.asarray(y).astype(int), index=X.index)

    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=args.seed
    )
    metric_rows: list[dict] = []
    cost_frames: list[pd.DataFrame] = []
    topk_frames: list[pd.DataFrame] = []

    for fold_id, (train_idx, test_idx) in enumerate(rskf.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
        X_fit, X_val, y_fit, y_val = train_test_split(
            X_train,
            y_train,
            test_size=0.25,
            stratify=y_train,
            random_state=args.seed + fold_id,
        )
        y_test_arr = y_test.to_numpy()
        y_val_arr = y_val.to_numpy()
        seed = args.seed + fold_id

        # --- G3 (calibrated G0 + threshold on val) ---
        g3_test, _, g3_extras = fit_predict_scores(
            "G3",
            X_fit,
            y_fit,
            X_test,
            num_cols,
            cat_cols,
            seed=seed,
            X_val=X_val,
            y_val=y_val,
        )
        g3_val, _, _ = fit_predict_scores(
            "G3",
            X_fit,
            y_fit,
            X_val,
            num_cols,
            cat_cols,
            seed=seed,
            X_val=X_val,
            y_val=y_val,
        )

        # --- G0 raw (supervised base for combinations) ---
        g0_test, _, _ = fit_predict_scores(
            "G0", X_fit, y_fit, X_test, num_cols, cat_cols, seed=seed
        )
        g0_val, _, _ = fit_predict_scores(
            "G0", X_fit, y_fit, X_val, num_cols, cat_cols, seed=seed
        )

        # Ranking metrics may use raw scores; cost layer uses calibrated scores.
        rank_methods: dict[str, np.ndarray] = {
            "G3": g3_test,
            "G0": g0_test,
        }
        cost_methods: dict[str, tuple[np.ndarray, np.ndarray]] = {
            "G3": (g3_test, g3_val),
        }
        g0_val_c, g0_test_c = _calibrate(g0_val, y_val_arr, g0_test)
        cost_methods["G0_cal"] = (g0_test_c, g0_val_c)

        for mode in config["anomaly_modes"]:
            if mode == "IF_all":
                a_test = scores_if_all(X_fit, X_test, num_cols, cat_cols, seed=seed)
                a_val = scores_if_all(X_fit, X_val, num_cols, cat_cols, seed=seed)
            else:
                a_test = scores_if_normal(
                    X_fit, y_fit, X_test, num_cols, cat_cols, seed=seed
                )
                a_val = scores_if_normal(
                    X_fit, y_fit, X_val, num_cols, cat_cols, seed=seed
                )
            rank_methods[mode] = a_test

            # Rank-average: weight from validation grid (incl. 0.5)
            w = choose_rank_weight(y_val_arr, g0_val, a_val)
            ra_test = rank_average(g0_test, a_test, weight=w)
            ra_val = rank_average(g0_val, a_val, weight=w)
            ra_val_c, ra_test_c = _calibrate(ra_val, y_val_arr, ra_test)
            name_ra = f"rankavg_{mode}"
            rank_methods[name_ra] = ra_test_c
            cost_methods[name_ra] = (ra_test_c, ra_val_c)

            # Stacking: OOF anomaly on fit rows; full-fit anomaly on val/test
            oof_fit = oof_anomaly_feature(
                X_fit,
                y_fit,
                num_cols,
                cat_cols,
                mode=mode,
                n_splits=3,
                seed=seed,
            )
            anom_val = full_anomaly_feature(
                X_fit, y_fit, X_val, num_cols, cat_cols, mode=mode, seed=seed
            )
            anom_test = full_anomaly_feature(
                X_fit, y_fit, X_test, num_cols, cat_cols, mode=mode, seed=seed
            )
            stack_test, stack_val = stack_fit_predict(
                X_fit,
                y_fit,
                oof_fit,
                [(X_test, anom_test), (X_val, anom_val)],
                num_cols,
                cat_cols,
                seed=seed,
            )
            st_val_c, st_test_c = _calibrate(stack_val, y_val_arr, stack_test)
            name_st = f"stack_{mode}"
            rank_methods[name_st] = st_test_c
            cost_methods[name_st] = (st_test_c, st_val_c)

        for method, scores_te in rank_methods.items():
            metrics = compute_metrics(y_test_arr, scores_te)
            row = {
                "fold": fold_id,
                "arm": method,
                "n_test": int(len(y_test_arr)),
                "fraud_rate_test": float(y_test_arr.mean()),
                **metrics,
            }
            if method == "G3":
                row["threshold"] = g3_extras["threshold"]
            metric_rows.append(row)

        for method, (scores_te, scores_va) in cost_methods.items():
            cost_frames.append(
                cost_table_for_scores(
                    y_test_arr,
                    scores_te,
                    y_val_arr,
                    scores_va,
                    method=method,
                    fold=fold_id,
                )
            )
            topk_frames.append(
                topk_curve(y_test_arr, scores_te, method=method, fold=fold_id)
            )

    folds = pd.DataFrame(metric_rows)
    costs = pd.concat(cost_frames, ignore_index=True)
    topk = pd.concat(topk_frames, ignore_index=True)
    diffs = paired_diffs_table(folds, metric="pr_auc", baseline="G0", seed=args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    folds_path = args.out_dir / "folds_metrics_anomaly.csv"
    diffs_path = args.out_dir / "paired_diffs_anomaly.csv"
    cost_path = args.out_dir / "cost_table.csv"
    topk_path = args.out_dir / "topk_curve.csv"
    folds.to_csv(folds_path, index=False)
    diffs.to_csv(diffs_path, index=False)
    costs.to_csv(cost_path, index=False)
    topk.to_csv(topk_path, index=False)

    print(f"wrote {folds_path}")
    print(f"wrote {diffs_path}")
    print(f"wrote {cost_path}")
    print(f"wrote {topk_path}")
    print(folds.groupby("arm")[["pr_auc", "roc_auc"]].mean().round(4))
    print(
        costs.groupby(["method", "rule", "L"])["expected_cost_per_claim"]
        .mean()
        .unstack("L")
        .round(4)
    )


if __name__ == "__main__":
    main()

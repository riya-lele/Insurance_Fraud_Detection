"""Train final model if needed; save global SHAP figure and reason examples."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import load_dataset
from src.explain import explain_claims, global_mean_abs_shap, plot_global_shap
from src.final_model import (
    DEFAULT_PATH,
    calibrated_probs,
    decision_zone,
    load_final,
    raw_scores,
    train_and_save,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model-path", type=Path, default=DEFAULT_PATH)
    p.add_argument("--out-dir", type=Path, default=ROOT / "results")
    p.add_argument("--fig-dir", type=Path, default=ROOT / "results" / "figures")
    p.add_argument("--n-examples", type=int, default=10)
    p.add_argument("--retrain", action="store_true", help="Always retrain final model")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = {
        "seed": args.seed,
        "model_path": str(args.model_path),
        "n_examples": args.n_examples,
        "retrain": args.retrain,
    }
    print("config:", config)

    X, y, num_cols, cat_cols = load_dataset(verbose=True)

    if args.retrain or not args.model_path.exists():
        bundle = train_and_save(args.model_path, seed=args.seed)
        print(f"wrote {args.model_path}")
    else:
        bundle = load_final(args.model_path)
        print(f"loaded {args.model_path}")

    # Held-out calibration rows for global SHAP (never used to fit the booster)
    cal_idx = bundle["cal_index"]
    X_hold = X.loc[cal_idx]

    mean_abs = global_mean_abs_shap(bundle, X_hold)
    fig_path = args.fig_dir / "global_shap.png"
    plot_global_shap(mean_abs, fig_path)
    print(f"wrote {fig_path}")

    # Flagged claims: calibrated p >= threshold; rank by raw score
    probs = calibrated_probs(bundle, X_hold)
    raw = raw_scores(bundle, X_hold)
    t = float(bundle["threshold"])
    flagged = X_hold.loc[probs >= t].copy()
    flagged["_raw"] = raw[probs >= t]
    flagged["_prob"] = probs[probs >= t]
    flagged = flagged.sort_values("_raw", ascending=False).head(args.n_examples)

    explanations = explain_claims(bundle, flagged.drop(columns=["_raw", "_prob"]))
    rows = []
    for i, (idx, row) in enumerate(flagged.iterrows()):
        zone = decision_zone(row["_prob"], t)
        for rank, item in enumerate(explanations[i], start=1):
            rows.append(
                {
                    "claim_index": int(idx),
                    "raw_score": float(row["_raw"]),
                    "calibrated_prob": float(row["_prob"]),
                    "threshold": t,
                    "zone": zone,
                    "rank": rank,
                    "feature": item["feature"],
                    "value": item["value"],
                    "shap": item["shap"],
                    "reason": item["reason"],
                    "label": int(y.loc[idx]),
                }
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.out_dir / "reason_examples.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()

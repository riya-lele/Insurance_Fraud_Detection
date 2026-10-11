"""Run imbalance arms L0, G0, G1, G2 and write fold / paired-diff CSVs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import load_dataset
from src.evaluate import paired_diffs_table, run_cv
from src.models import ARMS


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--quick",
        action="store_true",
        help="3 folds x 1 repeat (default: 5 folds x 3 repeats)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "results",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    n_splits, n_repeats = (3, 1) if args.quick else (5, 3)

    config = {
        "arms": ARMS,
        "n_splits": n_splits,
        "n_repeats": n_repeats,
        "seed": args.seed,
        "quick": args.quick,
    }
    print("config:", config)

    X, y, num_cols, cat_cols = load_dataset(verbose=True)
    folds = run_cv(
        X,
        y,
        num_cols,
        cat_cols,
        arms=ARMS,
        n_splits=n_splits,
        n_repeats=n_repeats,
        seed=args.seed,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    folds_path = args.out_dir / "folds_metrics.csv"
    diffs_path = args.out_dir / "paired_diffs.csv"
    folds.to_csv(folds_path, index=False)

    diffs = paired_diffs_table(folds, metric="pr_auc", baseline="G0", seed=args.seed)
    diffs.to_csv(diffs_path, index=False)

    print(f"wrote {folds_path}")
    print(f"wrote {diffs_path}")
    print(folds.groupby("arm")[["pr_auc", "roc_auc"]].mean().round(4))


if __name__ == "__main__":
    main()

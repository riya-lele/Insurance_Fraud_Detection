"""Read results/*.csv; write three figures and results/summary.md (tables only)."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGS = RESULTS / "figures"


def load_csvs(results_dir: Path = RESULTS) -> dict[str, pd.DataFrame]:
    paths = sorted(results_dir.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No CSV files in {results_dir}")
    return {p.stem: pd.read_csv(p) for p in paths}


def fig_pr_curves(data: dict[str, pd.DataFrame], out: Path) -> None:
    """Sparse PR operating curves from mean Precision/Recall @ K points."""
    frames = []
    if "folds_metrics" in data:
        df = data["folds_metrics"].copy()
        df["source"] = "imbalance"
        frames.append(df)
    if "folds_metrics_anomaly" in data:
        df = data["folds_metrics_anomaly"].copy()
        # Prefer anomaly file for overlapping arms (G0/G3); keep IF/rank/stack
        df["source"] = "anomaly"
        frames.append(df)
    if not frames:
        raise FileNotFoundError("Need folds_metrics*.csv for PR curves")

    combined = pd.concat(frames, ignore_index=True)
    # Deduplicate arm names: imbalance arms from folds_metrics; extra from anomaly
    imb_arms = set(data.get("folds_metrics", pd.DataFrame(columns=["arm"]))["arm"].unique())
    anom = data.get("folds_metrics_anomaly")
    if anom is not None:
        extra = anom.loc[~anom["arm"].isin(imb_arms) | (anom["arm"] == "G3")]
        base = data["folds_metrics"] if "folds_metrics" in data else anom.iloc[0:0]
        plot_df = pd.concat([base, extra], ignore_index=True)
    else:
        plot_df = combined

    ks = [1, 2, 5, 10]
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for arm, sub in plot_df.groupby("arm"):
        recalls, precisions = [], []
        for k in ks:
            rcol, pcol = f"recall_at_{k}pct", f"precision_at_{k}pct"
            if rcol not in sub.columns:
                continue
            recalls.append(sub[rcol].mean())
            precisions.append(sub[pcol].mean())
        if not recalls:
            continue
        order = np.argsort(recalls)
        ax.plot(
            np.asarray(recalls)[order],
            np.asarray(precisions)[order],
            marker="o",
            label=arm,
        )
    ax.set_xlabel("Recall @ K")
    ax.set_ylabel("Precision @ K")
    ax.set_title("PR operating points (mean over folds, K ∈ {1,2,5,10}%)")
    ax.legend(fontsize=8, loc="best", ncol=2)
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_precision_at_k(data: dict[str, pd.DataFrame], out: Path) -> None:
    if "folds_metrics" not in data and "folds_metrics_anomaly" not in data:
        raise FileNotFoundError("Need folds_metrics*.csv for Precision@K")

    frames = []
    if "folds_metrics" in data:
        frames.append(data["folds_metrics"])
    if "folds_metrics_anomaly" in data:
        anom = data["folds_metrics_anomaly"]
        if "folds_metrics" in data:
            keep = ~anom["arm"].isin(data["folds_metrics"]["arm"].unique()) | (
                anom["arm"] == "G3"
            )
            frames.append(anom.loc[keep])
        else:
            frames.append(anom)
    df = pd.concat(frames, ignore_index=True)

    ks = [1, 2, 5, 10]
    arms = sorted(df["arm"].unique())
    x = np.arange(len(ks))
    width = 0.8 / max(len(arms), 1)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, arm in enumerate(arms):
        means = [df.loc[df["arm"] == arm, f"precision_at_{k}pct"].mean() for k in ks]
        ax.bar(x + i * width, means, width=width, label=arm)
    ax.set_xticks(x + width * (len(arms) - 1) / 2)
    ax.set_xticklabels([f"{k}%" for k in ks])
    ax.set_xlabel("Investigation capacity K (% of claims)")
    ax.set_ylabel("Precision @ K")
    ax.set_title("Precision @ K (mean over folds)")
    ax.legend(fontsize=8, ncol=2, loc="best")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_cost_vs_ratio(data: dict[str, pd.DataFrame], out: Path) -> None:
    if "cost_table" not in data:
        raise FileNotFoundError("Need cost_table.csv for cost-vs-ratio plot")
    cost = data["cost_table"]
    # Focus on decision rules (exclude investigate-all/none clutter optional — keep all rules)
    rules = [
        "threshold_0.5",
        "f1_optimal",
        "cost_bayes",
        "cost_optimal_val",
        "investigate_none",
        "investigate_all",
    ]
    methods = sorted(cost["method"].unique())
    # One panel per method, lines = rules
    n = len(methods)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.4 * nrows), sharey=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, method in zip(axes, methods):
        sub = cost.loc[cost["method"] == method]
        for rule in rules:
            r = sub.loc[sub["rule"] == rule]
            if r.empty:
                continue
            g = r.groupby("L")["expected_cost_per_claim"].mean().sort_index()
            ax.plot(g.index, g.values, marker="o", label=rule)
        ax.set_title(method)
        ax.set_xlabel("L (missed-fraud loss)")
        ax.set_ylabel("Expected cost / claim")
        ax.grid(True, alpha=0.3)

    for ax in axes[len(methods) :]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=8)
    fig.suptitle("Hypothetical cost vs loss ratio (c = 1)", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _md_table(df: pd.DataFrame, float_cols: list[str] | None = None) -> str:
    show = df.copy()
    if float_cols:
        for c in float_cols:
            if c in show.columns:
                show[c] = show[c].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")
    cols = list(show.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = [
        "| " + " | ".join(str(show.iloc[i][c]) for c in cols) + " |"
        for i in range(len(show))
    ]
    return "\n".join([header, sep, *body])


def write_summary(data: dict[str, pd.DataFrame], out: Path) -> None:
    lines: list[str] = ["# Results summary", ""]
    lines.append("Tables only. Costs are hypothetical (c = 1).")
    lines.append("")

    # Imbalance fold means
    if "folds_metrics" in data:
        df = data["folds_metrics"]
        rows = []
        metrics = [
            "pr_auc",
            "roc_auc",
            "precision_at_1pct",
            "precision_at_2pct",
            "precision_at_5pct",
            "precision_at_10pct",
            "recall_at_1pct",
            "recall_at_2pct",
            "recall_at_5pct",
            "recall_at_10pct",
        ]
        for arm, sub in df.groupby("arm"):
            row = {"arm": arm, "n_folds": len(sub)}
            for m in metrics:
                row[f"{m}_mean"] = sub[m].mean()
            rows.append(row)
        tab = pd.DataFrame(rows).sort_values("arm")
        lines.append("## Imbalance arms (mean over folds)")
        lines.append("")
        lines.append(_md_table(tab, [c for c in tab.columns if c.endswith("_mean")]))
        lines.append("")

    if "paired_diffs" in data:
        lines.append("## Paired differences vs G0 (imbalance)")
        lines.append("")
        lines.append(
            _md_table(
                data["paired_diffs"],
                ["mean_diff", "ci_low", "ci_high"],
            )
        )
        lines.append("")
        lines.append("_Note from CSV: folds are correlated; CI is approximate._")
        lines.append("")

    if "folds_metrics_anomaly" in data:
        df = data["folds_metrics_anomaly"]
        rows = []
        for arm, sub in df.groupby("arm"):
            rows.append(
                {
                    "arm": arm,
                    "n_folds": len(sub),
                    "pr_auc_mean": sub["pr_auc"].mean(),
                    "roc_auc_mean": sub["roc_auc"].mean(),
                    "precision_at_5pct_mean": sub["precision_at_5pct"].mean(),
                    "recall_at_5pct_mean": sub["recall_at_5pct"].mean(),
                }
            )
        tab = pd.DataFrame(rows).sort_values("arm")
        lines.append("## Anomaly / calibrated methods (mean over folds)")
        lines.append("")
        lines.append(
            _md_table(
                tab,
                [
                    "pr_auc_mean",
                    "roc_auc_mean",
                    "precision_at_5pct_mean",
                    "recall_at_5pct_mean",
                ],
            )
        )
        lines.append("")

    if "paired_diffs_anomaly" in data:
        lines.append("## Paired differences vs G0 (anomaly)")
        lines.append("")
        lines.append(
            _md_table(
                data["paired_diffs_anomaly"],
                ["mean_diff", "ci_low", "ci_high"],
            )
        )
        lines.append("")

    if "cost_table" in data:
        cost = data["cost_table"]
        g = (
            cost.groupby(["method", "rule", "L"], as_index=False)[
                "expected_cost_per_claim"
            ]
            .mean()
            .rename(columns={"expected_cost_per_claim": "mean_expected_cost_per_claim"})
        )
        wide = g.pivot_table(
            index=["method", "rule"],
            columns="L",
            values="mean_expected_cost_per_claim",
        ).reset_index()
        wide.columns = [
            str(c) if c in ("method", "rule") else f"L={c}" for c in wide.columns
        ]
        float_cols = [c for c in wide.columns if c.startswith("L=")]
        lines.append("## Hypothetical expected cost per claim (mean over folds)")
        lines.append("")
        lines.append(_md_table(wide, float_cols))
        lines.append("")

    if "topk_curve" in data:
        topk = data["topk_curve"]
        g = (
            topk.groupby(["method", "k_pct"], as_index=False)[["precision", "recall"]]
            .mean()
            .rename(columns={"precision": "precision_mean", "recall": "recall_mean"})
        )
        lines.append("## Top-K capacity curve (mean over folds)")
        lines.append("")
        lines.append(_md_table(g, ["precision_mean", "recall_mean"]))
        lines.append("")

    lines.append("## Figures")
    lines.append("")
    lines.append("- `results/figures/pr_curves.png`")
    lines.append("- `results/figures/precision_at_k.png`")
    lines.append("- `results/figures/cost_vs_ratio.png`")
    lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    data = load_csvs(RESULTS)
    FIGS.mkdir(parents=True, exist_ok=True)
    fig_pr_curves(data, FIGS / "pr_curves.png")
    fig_precision_at_k(data, FIGS / "precision_at_k.png")
    fig_cost_vs_ratio(data, FIGS / "cost_vs_ratio.png")
    write_summary(data, RESULTS / "summary.md")
    print(f"wrote {FIGS / 'pr_curves.png'}")
    print(f"wrote {FIGS / 'precision_at_k.png'}")
    print(f"wrote {FIGS / 'cost_vs_ratio.png'}")
    print(f"wrote {RESULTS / 'summary.md'}")


if __name__ == "__main__":
    main()

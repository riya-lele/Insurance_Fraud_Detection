
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the project's dataset
df = pd.read_csv("data/raw/fraud_oracle.csv")

# Create output directory
out = Path("results/figures")
out.mkdir(parents=True, exist_ok=True)

target = "FraudFound_P"

# 1. Class distribution
counts = df[target].value_counts().reindex([0, 1], fill_value=0)

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(
    ["Non-fraud (0)", "Fraud (1)"],
    counts.values
)
ax.set_title("Distribution of Insurance Claims")
ax.set_ylabel("Number of claims")
ax.bar_label(bars, fmt="%d")
fig.tight_layout()
fig.savefig(out / "class_distribution.png", dpi=300)
plt.close(fig)

# 2. Fraud rate by important categorical features
categorical_features = [
    "Fault",
    "AccidentArea",
    "Sex",
    "VehicleCategory",
    "AgentType",
]

available = [c for c in categorical_features if c in df.columns]

fig, axes = plt.subplots(
    len(available), 1,
    figsize=(9, 4 * len(available)),
    squeeze=False
)

for ax, col in zip(axes.flatten(), available):
    summary = df.groupby(
        col, observed=True
    )[target].agg(["mean", "count"])

    summary = summary.sort_values("mean", ascending=False)

    bars = ax.bar(
        summary.index.astype(str),
        summary["mean"] * 100
    )

    ax.set_title(f"Fraud Rate by {col}")
    ax.set_ylabel("Fraud rate (%)")
    ax.set_xlabel(col)
    ax.tick_params(axis="x", rotation=30)
    ax.bar_label(bars, fmt="%.1f%%", padding=2)

fig.tight_layout()
fig.savefig(out / "fraud_rate_by_category.png", dpi=300)
plt.close(fig)

# 3. Correlation heatmap for numeric features
numeric = df.select_dtypes(include="number").copy()

# Exclude row/index identifiers from the heatmap
numeric = numeric.drop(
    columns=["PolicyNumber", "Year"],
    errors="ignore"
)

corr = numeric.corr()

fig, ax = plt.subplots(
    figsize=(max(9, len(corr.columns) * 0.65),
             max(7, len(corr.columns) * 0.55))
)

im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr.columns)))
ax.set_yticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
ax.set_yticklabels(corr.columns, fontsize=8)
ax.set_title("Correlation Heatmap of Numeric Features")
fig.colorbar(im, ax=ax, label="Pearson correlation")
fig.tight_layout()
fig.savefig(out / "correlation_heatmap.png", dpi=300)
plt.close(fig)

# Print actual fraud rates and sample sizes for the paper
print("\nFraud rates by categorical feature:")
for col in available:
    print(f"\n{col}")
    summary = df.groupby(col, observed=True)[target].agg(
        fraud_rate="mean",
        count="count"
    )
    summary["fraud_rate_percent"] = summary["fraud_rate"] * 100
    print(summary[["fraud_rate_percent", "count"]].round(2))

print(f"\nFigures saved in: {out}")

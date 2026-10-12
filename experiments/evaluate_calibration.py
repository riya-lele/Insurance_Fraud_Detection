
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve

from src.final_model import load_final, calibrated_probs
from src.data import load_dataset

# Load the saved model and dataset
bundle = load_final()
X, y, _, _ = load_dataset(verbose=False)

# Use the calibration subset recorded in the saved model
idx = [i for i in bundle["cal_index"] if i in X.index]

if not idx:
    raise ValueError(
        "No matching calibration indices found. "
        "Check whether the dataset has changed since the model was saved."
    )

X_cal = X.loc[idx, bundle["feature_list"]]
y_cal = y.loc[idx].astype(int)

# Get calibrated fraud probabilities
probs = calibrated_probs(bundle, X_cal)

# Calculate Brier score
brier = brier_score_loss(y_cal, probs)

# Calculate reliability curve
observed_rate, mean_predicted = calibration_curve(
    y_cal,
    probs,
    n_bins=10,
    strategy="uniform"
)

# Create output folder
out_dir = ROOT / "results" / "figures"
out_dir.mkdir(parents=True, exist_ok=True)

# Plot reliability curve
fig, ax = plt.subplots(figsize=(6, 5))

ax.plot(
    mean_predicted,
    observed_rate,
    marker="o",
    label="Final model"
)
ax.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Perfect calibration"
)

ax.set_xlabel("Mean predicted fraud probability")
ax.set_ylabel("Observed fraud rate")
ax.set_title("Reliability Curve — Final Model")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()

fig.savefig(out_dir / "reliability_curve.png", dpi=200)
plt.close(fig)

# Save metrics and evaluation details
report = (
    "Calibration evaluation\n"
    "----------------------\n"
    f"Claims evaluated: {len(y_cal)}\n"
    f"Fraud cases: {int(y_cal.sum())}\n"
    f"Brier score: {brier:.6f}\n"
    "Evaluation subset: model calibration subset\n"
    "Caution: calibrator was fitted on these same labels; "
    "results are diagnostic, not independent test performance.\n"
)

report_path = ROOT / "results" / "calibration_metrics.txt"
report_path.parent.mkdir(parents=True, exist_ok=True)
report_path.write_text(report, encoding="utf-8")

print(report)
print(f"Reliability curve saved to: {out_dir / 'reliability_curve.png'}")
print(f"Metrics saved to: {report_path}")

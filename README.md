# Insurance Fraud Detection (Tier 1)

Fixed-hyperparameter comparison of imbalance strategies, anomaly scores, and hypothetical cost-based thresholds on vehicle insurance claims. Design source of truth: [`spec.md`](spec.md).

## Setup

Python 3.10+ recommended.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

## Dataset

Place the raw file at:

```text
data/raw/fraud_oracle.csv
```

as specified in [`spec.md`](spec.md) (target `FraudFound_P`; ~15.4k rows).  
Download source (same `fraud_oracle.csv`): [Vehicle Insurance Claim Fraud Detection on Kaggle](https://www.kaggle.com/datasets/shivamb/vehicle-claim-fraud-detection).

`data/` is gitignored and must not be committed. Treat `data/raw` as read-only after download.

## Reproduce

From the repo root (with the venv active):

```bash
# Tests (leakage / metrics / OOF stacking)
python -m pytest tests/ -v

# Quick mode (3 folds × 1 repeat) — used for smoke checks
python experiments/run_imbalance.py --quick
python experiments/run_anomaly_cost.py --quick

# Full Tier-1 CV (5 folds × 3 repeats) — slower; run locally when needed
python experiments/run_imbalance.py
python experiments/run_anomaly_cost.py

# Figures + tables-only summary from results/*.csv
python experiments/make_report.py
```

Outputs land in `results/` (`folds_metrics.csv`, `paired_diffs.csv`, `cost_table.csv`, `topk_curve.csv`, anomaly variants, `figures/`, `summary.md`).

## Limitations

- **Hypothetical costs.** Investigation cost `c = 1` and missed-fraud losses `L ∈ {5, 10, 20, 50}` are unitless assumptions, not dollars or insurer loss data.
- **Single dataset.** Results are for this one public claims table only; no external validation set.
- **No tuning.** LightGBM and Isolation Forest hyperparameters are fixed and shared across arms (no hyperparameter search in Tier 1).

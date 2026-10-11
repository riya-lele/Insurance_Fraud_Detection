# Spec (Tier 1)
Question: at fixed investigation capacity, which imbalance strategy and whether an anomaly
score improves fraud recovery; how do cost-optimal thresholds shift with assumed cost ratios.

## Data
data/raw/fraud_oracle.csv. Target: FraudFound_P (1 = fraud). ~15.4k rows, ~923 fraud (~6%).
Columns: Month, WeekOfMonth, DayOfWeek, Make, AccidentArea, DayOfWeekClaimed, MonthClaimed,
WeekOfMonthClaimed, Sex, MaritalStatus, Age, Fault, PolicyType, VehicleCategory, VehiclePrice,
PolicyNumber, RepNumber, Deductible, DriverRating, Days_Policy_Accident, Days_Policy_Claim,
PastNumberOfClaims, AgeOfVehicle, AgeOfPolicyHolder, PoliceReportFiled, WitnessPresent,
AgentType, NumberOfSuppliments, AddressChange_Claim, NumberOfCars, Year, BasePolicy.
Many are pre-binned strings. Drop PolicyNumber (row index, not an ID). Drop Year from features
(kept for an optional temporal split later). Treat every other non-numeric column as categorical.
Drop exact duplicate rows; print counts before and after.

## Evaluation harness
Repeated stratified CV, 5 folds x 3 repeats (quick mode: 3 folds x 1). Seeds fixed.
Per outer fold, split the training part 75/25 (stratified) into fit/validation. Calibration
(isotonic) and thresholds are fit on the validation part only (never resampled).
Metrics on outer test fold: PR-AUC, ROC-AUC (supplementary), Precision@K and Recall@K for
K = 1, 2, 5, 10 percent of test claims. Save per-fold rows. Report paired differences between
arms across folds with a 95 percent bootstrap CI over folds (note: folds are correlated; say so).

## Models (fixed params, same for all arms)
LightGBM: n_estimators=300, learning_rate=0.05, num_leaves=31, subsample=0.8, colsample_bytree=0.8.
Categoricals: ordinal-encode; pass as categorical to LightGBM.
Arms:
 L0  logistic regression (one-hot, class_weight=balanced)
 G0  LightGBM, no imbalance handling
 G1  LightGBM, class weights (scale_pos_weight from training fold)
 G2  LightGBM, SMOTENC on training fit part only
 G3  = G0 + isotonic calibration + threshold moving (no resampling)
Anomaly:
 IF_all     Isolation Forest fit on all training rows
 IF_normal  Isolation Forest fit on non-fraud training rows only
Combinations with the best supervised arm:
 rank-average of supervised score and anomaly score (weight 0.5 and a small grid on validation)
 stacking: anomaly score as an extra feature. The feature on TRAINING rows must come from
 inner cross-fitting (out-of-fold), never a model that saw those rows, especially for IF_normal.

## Cost layer (on calibrated scores)
Unit costs: investigation cost c = 1, missed fraud loss L in {5, 10, 20, 50}.
Rules compared: threshold 0.5; F1-optimal on validation; cost-optimal on validation
(p > c/(L+c), then verified on validation); plus investigate-none and investigate-all baselines.
Report expected cost per claim on the outer test fold. Also top-K capacity curve.
All costs are hypothetical; label outputs as such.

## Outputs
results/folds_metrics.csv, results/paired_diffs.csv, results/cost_table.csv,
results/topk_curve.csv, figures: PR curves, Precision@K bar chart, cost-vs-ratio plot.

## Tier 2 (decisions fixed from Tier 1 results)
Final model: G0 (LightGBM, no resampling, fixed params from Tier 1). Train on a 75/25
stratified split of ALL data: fit on 75 percent, fit isotonic calibrator on the 25 percent.
Save models/final.joblib containing: model, calibrator, ordinal encoders / category maps,
feature list, default L=10 and c=1, threshold = c/(L+c) applied to the calibrated probability.
Rank claims by raw model score. Use the calibrated probability for thresholds and display only.

Decision zones at L=10: probability >= 2x threshold -> "investigate";
threshold <= probability < 2x threshold -> "review"; below threshold -> "approve".
Make L adjustable in the app; recompute thresholds and zones from c/(L+c).

SHAP: TreeExplainer on the raw LightGBM model. Global: mean |SHAP| bar plot on a held-out sample.
Per-claim: top 5 features by |SHAP| for flagged claims, converted to reason codes using a
small hand-written dictionary in src/reasons.py (feature, value -> phrase); fall back to
"feature = value". Features are pre-binned, so reasons are coarse. State this in the README.
Describe SHAP as model attribution, not causation.

Stability: arms G0, G1, G2 (reuse src/models.py unchanged), 5 seeds, same folds, fixed held-out
set. For each model compute global mean |SHAP| per feature. Report Spearman rank correlation of
feature rankings (within each arm across seeds, and across arms) and top-10 overlap.
Output results/shap_stability.csv. Report whatever is found.

App: Streamlit, loads models/final.joblib only; no training in the app. Tab 1: form with claim
fields, shows calibrated risk, decision zone, top-5 reason codes and a SHAP bar. Reject
missing or unknown category values with a clear message. Tab 2: replay held-out claims in
Year then Month order, with a running cost tally for the model policy versus investigate-none
and investigate-all, and per-claim latency. Costs hypothetical; label them.
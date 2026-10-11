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
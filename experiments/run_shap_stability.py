import sys, itertools
import numpy as np, pandas as pd, lightgbm as lgb, shap
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTENC

QUICK = "--quick" in sys.argv
SEEDS = [0, 1] if QUICK else [0, 1, 2, 3, 4]

df = pd.read_csv("data/raw/fraud_oracle.csv").drop_duplicates()
y = df["FraudFound_P"].values
X = df.drop(columns=["FraudFound_P", "PolicyNumber", "Year"])
cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
for c in cat_cols:
    X[c] = X[c].astype("category").cat.codes
feats = list(X.columns)
cat_idx = [feats.index(c) for c in cat_cols]

# fixed held-out set for SHAP, never used for training
X_pool, X_held, y_pool, y_held = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=0)
if QUICK:
    X_held = X_held.iloc[:500]

PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=31,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)

def fit_arm(arm, Xtr, ytr, seed):
    p = dict(PARAMS, random_state=seed)
    if arm == "G1":
        p["scale_pos_weight"] = (ytr == 0).sum() / (ytr == 1).sum()
    if arm == "G2":
        sm = SMOTENC(categorical_features=cat_idx, random_state=seed)
        Xr, ytr = sm.fit_resample(Xtr, ytr)
        Xtr = pd.DataFrame(Xr, columns=feats)
        Xtr[cat_cols] = Xtr[cat_cols].astype(int)
    m = lgb.LGBMClassifier(**p)
    m.fit(Xtr, ytr, categorical_feature=cat_cols)
    return m

def mean_abs_shap(m, Xh):
    sv = shap.TreeExplainer(m).shap_values(Xh)
    if isinstance(sv, list):
        sv = sv[1]
    elif getattr(sv, "ndim", 2) == 3:
        sv = sv[:, :, 1]
    return np.abs(sv).mean(axis=0)

rows = []
for seed in SEEDS:
    Xtr, _, ytr, _ = train_test_split(
        X_pool, y_pool, train_size=0.75, stratify=y_pool, random_state=seed)
    for arm in ["G0", "G1", "G2"]:
        imp = mean_abs_shap(fit_arm(arm, Xtr, ytr, seed), X_held)
        rows += [dict(arm=arm, seed=seed, feature=f, mean_abs_shap=v)
                 for f, v in zip(feats, imp)]
        print("done", arm, seed)

long = pd.DataFrame(rows)
long.to_csv("results/shap_importances.csv", index=False)
W = long.pivot_table(index="feature", columns=["arm", "seed"], values="mean_abs_shap")

def top10(a, b):
    ta = set(a.sort_values(ascending=False).index[:10])
    tb = set(b.sort_values(ascending=False).index[:10])
    return len(ta & tb)

out = []
for arm in ["G0", "G1", "G2"]:          # within arm, across seeds
    pairs = list(itertools.combinations(SEEDS, 2))
    out.append(dict(comparison=f"{arm} across seeds",
                    spearman_mean=np.mean([spearmanr(W[(arm, a)], W[(arm, b)])[0] for a, b in pairs]),
                    top10_overlap_mean=np.mean([top10(W[(arm, a)], W[(arm, b)]) for a, b in pairs])))
for a, b in itertools.combinations(["G0", "G1", "G2"], 2):  # across arms, same seed
    out.append(dict(comparison=f"{a} vs {b} (same seed)",
                    spearman_mean=np.mean([spearmanr(W[(a, s)], W[(b, s)])[0] for s in SEEDS]),
                    top10_overlap_mean=np.mean([top10(W[(a, s)], W[(b, s)]) for s in SEEDS])))
pd.DataFrame(out).to_csv("results/shap_stability.csv", index=False)
print(pd.DataFrame(out).round(3).to_string(index=False))
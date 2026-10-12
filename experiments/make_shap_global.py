import numpy as np, shap, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.final_model import load_final, encode_features
from src.data import load_dataset

b = load_final()
X, y, _, _ = load_dataset(verbose=False)
idx = [i for i in b["cal_index"] if i in X.index][:1500]
Xe = encode_features(b, X.loc[idx, b["feature_list"]])
sv = shap.TreeExplainer(b["model"]).shap_values(Xe)
if isinstance(sv, list): sv = sv[1]
sv = np.asarray(sv)
if sv.ndim == 3: sv = sv[:, :, 1]
imp = np.abs(sv).mean(axis=0)
order = np.argsort(imp)[-15:]
fig, ax = plt.subplots(figsize=(6, 5))
ax.barh([b["feature_list"][i] for i in order], imp[order], color="#4C78A8")
ax.set_xlabel("Mean |SHAP| (log-odds)"); ax.set_title("Global importance, final model")
fig.tight_layout(); fig.savefig("results/figures/shap_global.png", dpi=200)
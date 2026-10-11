"""SHAP explanations for the raw LightGBM model (attribution, not causation)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.final_model import encode_features, feature_list
from src.reasons import reason_for


def _positive_shap(shap_values) -> np.ndarray:
    """Return (n_samples, n_features) SHAP for the positive class."""
    if isinstance(shap_values, list):
        return np.asarray(shap_values[1])
    arr = np.asarray(shap_values)
    if arr.ndim == 3:
        # (n_samples, n_features, n_classes)
        return arr[:, :, 1]
    return arr


def _base_value(explainer, n_rows: int) -> np.ndarray:
    ev = explainer.expected_value
    if isinstance(ev, (list, np.ndarray)):
        ev = np.asarray(ev)
        base = float(ev[1]) if ev.ndim > 0 and len(ev) > 1 else float(ev.ravel()[0])
    else:
        base = float(ev)
    return np.full(n_rows, base, dtype=float)


def make_explainer(bundle: dict) -> shap.TreeExplainer:
    return shap.TreeExplainer(bundle["model"])


def shap_matrix(
    bundle: dict,
    X: pd.DataFrame,
    explainer: shap.TreeExplainer | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (shap_values, base_values, feature_names) for rows in X."""
    if explainer is None:
        explainer = make_explainer(bundle)
    X_enc = encode_features(bundle, X)
    feats = bundle.get("feature_list") or feature_list(
        bundle["num_cols"], bundle["cat_cols"]
    )
    values = _positive_shap(explainer.shap_values(X_enc))
    base = _base_value(explainer, len(X))
    return values, base, feats


def global_mean_abs_shap(
    bundle: dict,
    X: pd.DataFrame,
    explainer: shap.TreeExplainer | None = None,
) -> pd.Series:
    values, _, feats = shap_matrix(bundle, X, explainer=explainer)
    return pd.Series(np.mean(np.abs(values), axis=0), index=feats).sort_values(
        ascending=False
    )


def plot_global_shap(
    mean_abs: pd.Series,
    path: Path | str,
    *,
    top_n: int = 20,
    title: str = "Global mean |SHAP| (model attribution)",
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plot_s = mean_abs.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(plot_s))))
    ax.barh(plot_s.index.astype(str), plot_s.values, color="#4C72B0")
    ax.set_xlabel("mean |SHAP|")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def top_k_features(
    shap_row: np.ndarray,
    feature_names: list[str],
    X_row: pd.Series,
    k: int = 5,
) -> list[dict]:
    """Top-k features by |SHAP| with reason codes."""
    order = np.argsort(-np.abs(shap_row))[:k]
    out = []
    for i in order:
        feat = feature_names[i]
        val = X_row[feat]
        out.append(
            {
                "feature": feat,
                "value": val,
                "shap": float(shap_row[i]),
                "abs_shap": float(abs(shap_row[i])),
                "reason": reason_for(feat, val),
            }
        )
    return out


def explain_claims(
    bundle: dict,
    X: pd.DataFrame,
    *,
    k: int = 5,
    explainer: shap.TreeExplainer | None = None,
) -> list[list[dict]]:
    values, _, feats = shap_matrix(bundle, X, explainer=explainer)
    return [
        top_k_features(values[i], feats, X.iloc[i], k=k) for i in range(len(X))
    ]

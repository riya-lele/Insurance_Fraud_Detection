"""Load and audit the fraud_oracle dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

TARGET = "FraudFound_P"
DROP_COLS = ("PolicyNumber", "Year")
DEFAULT_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "fraud_oracle.csv"


def load_raw(path: Path | str = DEFAULT_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def audit_and_prepare(df: pd.DataFrame, verbose: bool = True) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Drop ID/leak columns and duplicates; return X-ready frame, numeric cols, categorical cols."""
    n_before = len(df)
    if TARGET not in df.columns:
        raise ValueError(f"Missing target column {TARGET!r}")

    work = df.drop(columns=[c for c in DROP_COLS if c in df.columns]).copy()
    work = work.drop_duplicates().reset_index(drop=True)
    n_after = len(work)

    y = work[TARGET].astype(int)
    feature_df = work.drop(columns=[TARGET])

    num_cols = [c for c in feature_df.columns if pd.api.types.is_numeric_dtype(feature_df[c])]
    cat_cols = [c for c in feature_df.columns if c not in num_cols]

    if verbose:
        print(f"rows before drop_duplicates: {n_before}")
        print(f"rows after drop_duplicates:  {n_after}")
        print(f"dropped exact duplicates:    {n_before - n_after}")
        print(f"fraud count / rate:          {int(y.sum())} / {y.mean():.4f}")
        print(f"n_features:                  {feature_df.shape[1]}")
        print(f"numeric ({len(num_cols)}):   {num_cols}")
        print(f"categorical ({len(cat_cols)}): {cat_cols}")
        assert "PolicyNumber" not in feature_df.columns
        assert "Year" not in feature_df.columns

    out = feature_df.copy()
    out[TARGET] = y
    return out, num_cols, cat_cols


def load_dataset(path: Path | str = DEFAULT_PATH, verbose: bool = True):
    """Return X (DataFrame), y (Series), num_cols, cat_cols."""
    raw = load_raw(path)
    prepared, num_cols, cat_cols = audit_and_prepare(raw, verbose=verbose)
    y = prepared[TARGET]
    X = prepared.drop(columns=[TARGET])
    return X, y, num_cols, cat_cols

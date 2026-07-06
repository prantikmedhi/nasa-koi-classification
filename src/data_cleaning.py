"""Data loading and cleaning for the KOI exoplanet classification project.

Handles: loading the raw Kepler Objects of Interest (KOI) cumulative table,
validating its shape, dropping fully-null / identifier / leakage-risk columns,
imputing missing values, encoding categoricals, and producing a stratified
train/test split.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42

EXPECTED_SHAPE = (9564, 140)
TARGET = "koi_disposition"

# Columns that are 100% null in this extract (verified in EDA) — carry no signal.
FULLY_NULL_COLS = [
    "koi_eccen_err1", "koi_eccen_err2",
    "koi_longp", "koi_longp_err1", "koi_longp_err2",
    "koi_ingress", "koi_ingress_err1", "koi_ingress_err2",
    "koi_sma_err1", "koi_sma_err2",
    "koi_incl_err1", "koi_incl_err2",
    "koi_teq_err1", "koi_teq_err2",
    "koi_model_dof", "koi_model_chisq",
    "koi_sage", "koi_sage_err1", "koi_sage_err2",
]

# Identifiers, free text, provenance metadata, and columns that would leak the
# label. `koi_pdisposition` is a near-duplicate of the target and must never be
# used as a feature. `kepler_name` only exists for confirmed planets — using it
# (or its missingness) would trivially leak the answer.
ID_LEAKAGE_COLS = [
    "rowid", "kepid", "kepoi_name", "kepler_name",
    "koi_vet_stat", "koi_vet_date", "koi_disp_prov", "koi_comment",
    "koi_pdisposition",
    "koi_datalink_dvr", "koi_datalink_dvs",
]

# Kepler pipeline false-positive vetting flags. These are the OUTPUT of NASA's
# automated vetting pipeline — they encode the very thing we are trying to
# predict and using them as features is data leakage. Removing them is the
# honest approach; the model must learn to classify planets from the transit
# physics and stellar parameters alone.
FPFLAG_COLS = ["koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_co", "koi_fpflag_ec"]

# Non-informative object columns: koi_quarters is a per-quarter observation
# bitstring (not a tabular feature), koi_limbdark_mod and koi_trans_mod are
# (near-)constant model-name strings. FPFLAG_COLS are data-leakage columns
# (NASA's pre-evaluated false-positive vetting pipeline output).
NON_INFORMATIVE_COLS = ["koi_quarters", "koi_limbdark_mod", "koi_trans_mod"] + FPFLAG_COLS

# Low-cardinality categorical columns kept and one-hot encoded.
CATEGORICAL_COLS = ["koi_fittype", "koi_parm_prov", "koi_sparprov", "koi_tce_delivname"]


def load_raw(path: str | Path) -> pd.DataFrame:
    """Load the raw KOI CSV and validate it against the expected shape."""
    df = pd.read_csv(path)
    if df.shape != EXPECTED_SHAPE:
        raise ValueError(f"Unexpected shape {df.shape}, expected {EXPECTED_SHAPE}")
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' missing")
    return df


def drop_unusable_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop fully-null, identifier/leakage, and non-informative columns."""
    to_drop = [
        c for c in FULLY_NULL_COLS + ID_LEAKAGE_COLS + NON_INFORMATIVE_COLS
        if c in df.columns
    ]
    return df.drop(columns=to_drop)


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode the remaining low-cardinality categorical columns.

    Missing categories become an explicit level rather than being imputed —
    'no value recorded' is itself informative for provenance columns.
    """
    present = [c for c in CATEGORICAL_COLS if c in df.columns]
    df = df.copy()
    for c in present:
        df[c] = df[c].fillna("missing")
    return pd.get_dummies(df, columns=present, dtype=int)


def impute_numeric(df: pd.DataFrame, medians: pd.Series | None = None):
    """Median-impute numeric columns.

    Median chosen over mean because most KOI physical quantities (period,
    depth, insolation, ...) are heavily right-skewed, so the mean would be
    dominated by outliers. If `medians` is given (fit on train), it is applied
    as-is so the test set never influences imputation.

    Returns (imputed_df, medians_used).
    """
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns
    if medians is None:
        medians = df[num_cols].median()
    df[num_cols] = df[num_cols].fillna(medians)
    return df, medians


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Column-level cleaning only (drop + encode). Imputation is done after the
    train/test split to avoid test-set information leaking into the medians."""
    df = drop_unusable_columns(df)
    df = encode_categoricals(df)
    return df


def split_features_target(df: pd.DataFrame):
    """Separate the feature matrix from the target labels."""
    y = df[TARGET]
    X = df.drop(columns=[TARGET])
    return X, y


def stratified_split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
    """80/20 split stratified on the class label to preserve class ratios."""
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )

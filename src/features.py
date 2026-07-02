"""Feature engineering for the KOI exoplanet classification project.

Adds domain-derived features on top of the cleaned KOI table and provides a
redundancy-reduction pass over the +/- uncertainty (`_err1`/`_err2`) columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Engineered feature names, for downstream reporting.
ENGINEERED_FEATURES = [
    "feat_depth_per_hour",
    "feat_prad_srad_ratio",
    "feat_mes_over_ses",
    "feat_duration_ratio",
    "feat_log_period",
    "feat_log_depth",
    "feat_log_insol",
    "feat_snr_per_transit",
]


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add domain-motivated derived features.

    Rationale per feature:
    - depth_per_hour: transit depth divided by duration. Genuine planetary
      transits are shallow and long; eclipsing binaries (the dominant false
      positive) tend to be deep and sharp, so this ratio separates them.
    - prad_srad_ratio: planet radius over stellar radius (unit-consistent
      re-derivation alongside koi_ror). Physically impossible values (ratio
      near/above 1 implies a stellar companion, not a planet) flag false
      positives.
    - mes_over_ses: multiple-event statistic over max single-event statistic.
      A real periodic transit accumulates evidence across events (high MES
      relative to any single event); a one-off systematic spike does not.
    - duration_ratio: observed duration vs. the duration expected for a
      central transit given the orbital period (Kepler's third law scaling,
      duration ~ period^{1/3}). Large deviations suggest a grazing eclipse or
      a blend.
    - log transforms of period/depth/insol: these span several orders of
      magnitude; logs make them usable by linear models and stabilize trees'
      split behavior in the tails.
    - snr_per_transit: total signal-to-noise normalized by the number of
      observed transits — separates "strong because deep" from "strong
      because seen many times".
    """
    df = df.copy()
    eps = 1e-10

    df["feat_depth_per_hour"] = df["koi_depth"] / (df["koi_duration"] + eps)

    # koi_prad is in Earth radii, koi_srad in Solar radii; 1 R_sun ≈ 109.2 R_earth.
    df["feat_prad_srad_ratio"] = df["koi_prad"] / (df["koi_srad"] * 109.2 + eps)

    df["feat_mes_over_ses"] = df["koi_max_mult_ev"] / (df["koi_max_sngle_ev"] + eps)

    expected_duration = df["koi_period"].clip(lower=0) ** (1.0 / 3.0)
    df["feat_duration_ratio"] = df["koi_duration"] / (expected_duration + eps)

    df["feat_log_period"] = np.log10(df["koi_period"].clip(lower=eps))
    df["feat_log_depth"] = np.log10(df["koi_depth"].clip(lower=eps))
    df["feat_log_insol"] = np.log10(df["koi_insol"].clip(lower=eps))

    df["feat_snr_per_transit"] = df["koi_model_snr"] / (
        df["koi_num_transits"].clip(lower=1)
    )

    return df


def find_low_variance_columns(df: pd.DataFrame, threshold: float = 1e-12) -> list[str]:
    """Numeric columns whose variance is (near) zero — they cannot inform any model."""
    num = df.select_dtypes(include=[np.number])
    variances = num.var()
    return variances[variances <= threshold].index.tolist()


def find_redundant_err_columns(df: pd.DataFrame, corr_threshold: float = 0.98) -> list[str]:
    """Flag `_err2` columns that are ~perfectly anti-correlated with their
    `_err1` twin.

    KOI uncertainty columns come in +/- pairs; for symmetric error bars
    err2 = -err1 exactly, so keeping both doubles the column count with zero
    added information. We keep err1 (the + side) and drop the mirrored err2.
    """
    redundant = []
    for col in df.columns:
        if not col.endswith("_err2"):
            continue
        twin = col[:-1] + "1"  # _err2 -> _err1
        if twin not in df.columns:
            continue
        pair = df[[col, twin]].dropna()
        if len(pair) < 10:
            continue
        corr = pair[col].corr(pair[twin])
        if pd.notna(corr) and abs(corr) >= corr_threshold:
            redundant.append(col)
    return redundant


def build_features(df: pd.DataFrame, drop_redundant: bool = True) -> pd.DataFrame:
    """Full feature pass: engineered features, then variance/redundancy pruning."""
    df = add_engineered_features(df)
    if drop_redundant:
        to_drop = set(find_low_variance_columns(df)) | set(find_redundant_err_columns(df))
        df = df.drop(columns=sorted(to_drop))
    return df

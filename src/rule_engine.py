"""
Rule-Based Detection Engine.

Applies configurable business rules to flag suspicious transactions.
Each rule is independent and additive — a transaction can trigger
multiple flags simultaneously.

Rules:
- Amount threshold: transactions above the 95th percentile
- Velocity check: more than N transactions in a time window per customer
- Time anomaly: transactions outside normal business hours
- Repeat pattern: same amount to same merchant in short succession
- Probe pattern: uses preceded_by_probe from feature_engine if present
"""

import logging

import numpy as np
import pandas as pd
from typing import Dict

logger = logging.getLogger("fraud_dashboard")

# Default thresholds — configurable via dashboard sidebar
DEFAULT_THRESHOLDS = {
    "amount_percentile": 0.95,
    "velocity_max_txns": 5,
    "velocity_window_hours": 1,
    "night_start_hour": 22,
    "night_end_hour": 6,
    "repeat_window_minutes": 30,
}


def apply_all_rules(df: pd.DataFrame, thresholds: Dict = None) -> pd.DataFrame:
    """Apply all rule-based flags to the dataset.

    Runs each rule function, adds individual flag columns, then sums them
    into a rule_score that counts how many rules each transaction triggered.
    Also uses preceded_by_probe from feature_engine output when present.

    Args:
        df: Feature-engineered DataFrame.
        thresholds: Optional custom thresholds overriding defaults.

    Returns:
        DataFrame with boolean flag columns and a combined rule_score.
    """
    t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    df = df.copy()

    df["flag_high_amount"] = flag_amount_threshold(df, t["amount_percentile"])
    df["flag_velocity"] = flag_velocity(df, t["velocity_max_txns"], t["velocity_window_hours"])
    df["flag_time_anomaly"] = flag_time_anomaly(df, t["night_start_hour"], t["night_end_hour"])
    df["flag_repeat_pattern"] = flag_repeat_pattern(df, t["repeat_window_minutes"])

    # Probe flag: reuse the feature already computed by feature_engine rather
    # than reimplementing the logic here.
    if "preceded_by_probe" in df.columns:
        df["flag_probe"] = df["preceded_by_probe"].fillna(0).astype(bool)

    flag_cols = [c for c in df.columns if c.startswith("flag_")]
    df["rule_score"] = df[flag_cols].sum(axis=1).astype(int)

    return df


def flag_amount_threshold(df: pd.DataFrame, percentile: float = 0.95) -> pd.Series:
    """Flag transactions with amounts above the given percentile.

    Args:
        df: DataFrame with 'amount' column.
        percentile: Percentile threshold (0-1).

    Returns:
        Boolean Series — True for flagged transactions.
    """
    threshold = df["amount"].quantile(percentile)
    return df["amount"] > threshold


def flag_velocity(
    df: pd.DataFrame, max_txns: int = 5, window_hours: int = 1
) -> pd.Series:
    """Flag customers with too many transactions in a rolling window.

    Uses the pre-computed txn_count_{window_hours}h column from feature_engine
    when available; recomputes it on the fly otherwise.

    Args:
        df: DataFrame with 'customer_id' and 'timestamp' columns.
        max_txns: Maximum allowed transactions per window before flagging.
        window_hours: Size of rolling window in hours.

    Returns:
        Boolean Series — True for flagged transactions.
    """
    col = f"txn_count_{window_hours}h"
    if col not in df.columns:
        from src.feature_engine import add_velocity_features
        df = add_velocity_features(df, windows=[window_hours])
    # NaN (no customer_id) treated as 0 — rule does not fire
    return df[col].fillna(0) > max_txns


def flag_time_anomaly(
    df: pd.DataFrame, night_start: int = 22, night_end: int = 6
) -> pd.Series:
    """Flag transactions occurring during unusual hours.

    Uses the pre-computed hour_of_day column from feature_engine when
    available; falls back to extracting directly from timestamp.

    Args:
        df: DataFrame with 'hour_of_day' or 'timestamp' column.
        night_start: Start of night window (24h format, inclusive).
        night_end: End of night window (24h format, exclusive).

    Returns:
        Boolean Series — True for flagged transactions.
    """
    hour = (
        df["hour_of_day"] if "hour_of_day" in df.columns
        else df["timestamp"].dt.hour
    )
    return (hour >= night_start) | (hour < night_end)


def flag_repeat_pattern(df: pd.DataFrame, window_minutes: int = 30) -> pd.Series:
    """Flag repeated same-amount transactions to the same merchant.

    A transaction is flagged when at least one other transaction shares the
    same (merchant, amount) pair within window_minutes in either direction.
    Both the original and the repeat are flagged.

    Args:
        df: DataFrame with 'amount', 'merchant', and 'timestamp' columns.
        window_minutes: Time window to check for repeats.

    Returns:
        Boolean Series — True for flagged transactions.
        Returns all-False when the 'merchant' column is absent.
    """
    if "merchant" not in df.columns:
        logger.warning("'merchant' column not found; repeat pattern flag will be False")
        return pd.Series(False, index=df.index)

    sort_order = np.argsort(df["timestamp"].values, kind="stable")
    restore_order = np.argsort(sort_order, kind="stable")

    df_sorted = df.iloc[sort_order].copy()
    df_sorted.index = range(len(df_sorted))

    ts_seconds = df_sorted["timestamp"].values.astype("datetime64[s]").astype(np.int64)
    window_s = window_minutes * 60
    flagged = np.zeros(len(df_sorted), dtype=bool)

    for (_, _), group in df_sorted.groupby(["merchant", "amount"]):
        if len(group) < 2:
            continue
        idx = group.index.values
        ts_grp = ts_seconds[idx]

        # has_prior: this row has a predecessor with same (merchant, amount) in window
        left_idxs = np.searchsorted(ts_grp, ts_grp - window_s, side="left")
        has_prior = np.arange(len(idx)) > left_idxs

        # has_next: this row has a successor with same (merchant, amount) in window
        right_idxs = np.searchsorted(ts_grp, ts_grp + window_s, side="right")
        has_next = right_idxs > np.arange(1, len(idx) + 1)

        flagged[idx] = has_prior | has_next

    return pd.Series(flagged[restore_order], index=df.index)

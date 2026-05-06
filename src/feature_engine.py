"""
Feature Engineering Module.

Creates derived features from raw transaction data to improve
both rule-based detection and ML model performance.

Features include:
- Rolling transaction averages per customer
- Transaction velocity (count per time window)
- Time-based features (hour of day, day of week, is_weekend)
- Amount deviation from customer historical mean
- Round-amount flag (whole-dollar amounts common in fraud)
- Card-probe pattern (small test transaction followed by large cash-out)
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("fraud_dashboard")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Main entry point: apply all feature engineering steps.

    Args:
        df: Validated DataFrame with standardized columns.

    Returns:
        DataFrame with original columns plus all derived features.
    """
    df = add_time_features(df)
    df = add_velocity_features(df)
    df = add_amount_features(df)
    df = add_probe_features(df)
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract time-based features from the timestamp column.

    Creates: hour_of_day, day_of_week, is_weekend, is_night (10pm-6am).

    Args:
        df: DataFrame with a 'timestamp' column.

    Returns:
        DataFrame with added time feature columns.
    """
    df = df.copy()
    ts = df["timestamp"]

    df["hour_of_day"] = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek  # 0 = Monday, 6 = Sunday
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_night"] = ((df["hour_of_day"] >= 22) | (df["hour_of_day"] < 6)).astype(int)

    return df


def add_velocity_features(df: pd.DataFrame, windows: list = [1, 6, 24]) -> pd.DataFrame:
    """Calculate transaction velocity per customer over rolling time windows.

    For each window, counts how many transactions the same customer made
    (including the current one) within the preceding N hours.

    Args:
        df: DataFrame with 'customer_id' and 'timestamp' columns.
        windows: List of time windows in hours.

    Returns:
        DataFrame with velocity columns (e.g., txn_count_1h, txn_count_6h).
        Columns are filled with NaN when customer_id is absent.
    """
    df = df.copy()

    if "customer_id" not in df.columns:
        logger.warning("'customer_id' not found; velocity features will be NaN")
        for window in windows:
            df[f"txn_count_{window}h"] = np.nan
        return df

    # Sort by timestamp so that within each customer group timestamps are monotonic.
    # argsort of argsort gives the inverse permutation to restore original row order.
    sort_order = np.argsort(df["timestamp"].values, kind="stable")
    restore_order = np.argsort(sort_order, kind="stable")

    df_sorted = df.iloc[sort_order].copy()
    df_sorted.index = range(len(df_sorted))  # clean integer positional index

    # Represent timestamps as integer seconds for vectorised window arithmetic.
    # datetime64[s] is stable across pandas versions (avoids ns vs us ambiguity).
    ts_seconds = df_sorted["timestamp"].values.astype("datetime64[s]").astype(np.int64)

    for window in windows:
        col = f"txn_count_{window}h"
        window_s = int(window * 3600)
        counts = np.zeros(len(df_sorted), dtype=float)

        for _, group in df_sorted.groupby("customer_id"):
            idx = group.index.values  # positions in df_sorted (sorted by timestamp)
            ts_grp = ts_seconds[idx]  # already sorted within each group
            # For each row, find the leftmost row in this group whose timestamp
            # falls within [ts - window, ts] (closed="both").
            left_idxs = np.searchsorted(ts_grp, ts_grp - window_s, side="left")
            counts[idx] = np.arange(len(ts_grp)) - left_idxs + 1

        df[col] = counts[restore_order]

    return df


def add_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create amount-based derived features.

    Creates: amount_zscore (deviation from customer mean),
    amount_percentile (within full dataset), log_amount.

    Uses per-customer statistics when customer_id is present; falls back
    to global statistics otherwise.

    Args:
        df: DataFrame with 'amount' and optionally 'customer_id' columns.

    Returns:
        DataFrame with added amount feature columns.
    """
    df = df.copy()

    df["log_amount"] = np.log1p(df["amount"])
    df["amount_percentile"] = df["amount"].rank(pct=True)
    df["is_round_amount"] = (df["amount"] % 1 == 0).astype(int)

    if "customer_id" in df.columns:
        grp = df.groupby("customer_id")["amount"]
        customer_mean = grp.transform("mean")
        # Customers with a single transaction have std=NaN; use 1 to avoid div-by-zero
        customer_std = grp.transform("std").fillna(1)
        df["amount_zscore"] = (df["amount"] - customer_mean) / customer_std
    else:
        global_mean = df["amount"].mean()
        global_std = df["amount"].std()
        denom = global_std if global_std > 0 else 1.0
        df["amount_zscore"] = (df["amount"] - global_mean) / denom

    return df


def add_probe_features(
    df: pd.DataFrame,
    probe_threshold: float = 10.0,
    window_hours: float = 24.0,
) -> pd.DataFrame:
    """Flag card-testing probe transactions and transactions preceded by one.

    Fraudsters confirm a stolen card is live by making a small test transaction
    (the 'probe') then follow up with a large cash-out. This function creates
    two signals:

    - is_small_probe: 1 when amount < probe_threshold. Marks the test transaction.
    - preceded_by_probe: 1 when the same customer made a small transaction within
      the preceding window_hours. Marks the follow-up cash-out.

    Args:
        df: DataFrame with 'amount', 'timestamp', and optionally 'customer_id'.
        probe_threshold: Amounts strictly below this value are classified as probes.
        window_hours: Look-back window in hours when searching for prior probes.

    Returns:
        DataFrame with added is_small_probe and preceded_by_probe columns.
    """
    df = df.copy()
    df["is_small_probe"] = (df["amount"] < probe_threshold).astype(int)

    if "customer_id" not in df.columns:
        logger.warning("'customer_id' not found; preceded_by_probe will be NaN")
        df["preceded_by_probe"] = np.nan
        return df

    sort_order = np.argsort(df["timestamp"].values, kind="stable")
    restore_order = np.argsort(sort_order, kind="stable")

    df_sorted = df.iloc[sort_order].copy()
    df_sorted.index = range(len(df_sorted))

    ts_seconds = df_sorted["timestamp"].values.astype("datetime64[s]").astype(np.int64)
    amounts = df_sorted["amount"].values
    window_s = int(window_hours * 3600)
    preceded = np.zeros(len(df_sorted), dtype=int)

    for _, group in df_sorted.groupby("customer_id"):
        idx = group.index.values
        ts_grp = ts_seconds[idx]
        amt_grp = amounts[idx]

        # Prefix sum over probe flags allows O(1) range-count queries.
        # cum_probes[i] = number of probes in amt_grp[0:i] (excludes index i).
        cum_probes = np.concatenate([[0], np.cumsum(amt_grp < probe_threshold)])
        left_idxs = np.searchsorted(ts_grp, ts_grp - window_s, side="left")

        # probe_counts[i] = probes in [left_idxs[i], i) — strictly before current row
        probe_counts = cum_probes[np.arange(len(idx))] - cum_probes[left_idxs]
        preceded[idx] = (probe_counts > 0).astype(int)

    df["preceded_by_probe"] = preceded[restore_order]
    return df

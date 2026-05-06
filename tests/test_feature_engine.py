"""Tests for feature_engine module."""

import numpy as np
import pandas as pd
import pytest
from src.feature_engine import (
    add_amount_features,
    add_probe_features,
    add_time_features,
    add_velocity_features,
    engineer_features,
)


def _make_df(**overrides) -> pd.DataFrame:
    """Minimal valid DataFrame for feature engine tests."""
    data = {
        "transaction_id": [0, 1, 2, 3],
        "amount": [100.0, 9.0, 500.0, 50.00],
        "timestamp": pd.to_datetime([
            "2023-01-02 14:00",  # Monday 2pm
            "2023-01-02 14:30",  # Monday 2:30pm
            "2023-01-07 23:00",  # Saturday 11pm
            "2023-01-08 03:00",  # Sunday 3am
        ]),
        "label": [0, 0, 1, 0],
        "customer_id": ["A", "A", "B", "B"],
    }
    data.update(overrides)
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# add_time_features
# ---------------------------------------------------------------------------

class TestAddTimeFeatures:
    def test_creates_hour_of_day(self):
        """hour_of_day should equal the hour component of each timestamp."""
        df = _make_df()
        result = add_time_features(df)
        assert list(result["hour_of_day"]) == [14, 14, 23, 3]

    def test_creates_day_of_week(self):
        """day_of_week should be 0=Monday through 6=Sunday."""
        df = _make_df()
        result = add_time_features(df)
        # Jan 2 2023 = Monday (0), Jan 7 = Saturday (5), Jan 8 = Sunday (6)
        assert result["day_of_week"].iloc[0] == 0
        assert result["day_of_week"].iloc[2] == 5
        assert result["day_of_week"].iloc[3] == 6

    def test_is_weekend_true_for_saturday_sunday(self):
        """is_weekend should be 1 for Saturday and Sunday, 0 for weekdays."""
        df = _make_df()
        result = add_time_features(df)
        assert result["is_weekend"].iloc[0] == 0  # Monday
        assert result["is_weekend"].iloc[1] == 0  # Monday
        assert result["is_weekend"].iloc[2] == 1  # Saturday
        assert result["is_weekend"].iloc[3] == 1  # Sunday

    def test_is_night_flags_after_22_and_before_6(self):
        """is_night should be 1 for hours >= 22 or < 6."""
        df = _make_df()
        result = add_time_features(df)
        assert result["is_night"].iloc[0] == 0  # 2pm
        assert result["is_night"].iloc[1] == 0  # 2:30pm
        assert result["is_night"].iloc[2] == 1  # 11pm
        assert result["is_night"].iloc[3] == 1  # 3am

    def test_does_not_mutate_input(self):
        """add_time_features should return a copy, not modify the input."""
        df = _make_df()
        original_cols = list(df.columns)
        add_time_features(df)
        assert list(df.columns) == original_cols


# ---------------------------------------------------------------------------
# add_velocity_features
# ---------------------------------------------------------------------------

class TestAddVelocityFeatures:
    def test_single_transaction_count_is_one(self):
        """A customer with one transaction should have a velocity count of 1."""
        df = pd.DataFrame({
            "transaction_id": [0],
            "amount": [100.0],
            "timestamp": pd.to_datetime(["2023-01-01 12:00"]),
            "label": [0],
            "customer_id": ["X"],
        })
        result = add_velocity_features(df, windows=[1])
        assert result["txn_count_1h"].iloc[0] == 1.0

    def test_counts_accumulate_within_window(self):
        """Three transactions within 1h for the same customer should give counts 1, 2, 3."""
        df = pd.DataFrame({
            "transaction_id": [0, 1, 2],
            "amount": [10.0, 20.0, 30.0],
            "timestamp": pd.to_datetime([
                "2023-01-01 10:00",
                "2023-01-01 10:20",
                "2023-01-01 10:40",
            ]),
            "label": [0, 0, 0],
            "customer_id": ["A", "A", "A"],
        })
        result = add_velocity_features(df, windows=[1])
        assert list(result["txn_count_1h"]) == [1.0, 2.0, 3.0]

    def test_transaction_outside_window_not_counted(self):
        """A transaction more than 1h before the current one should not be counted."""
        df = pd.DataFrame({
            "transaction_id": [0, 1],
            "amount": [10.0, 20.0],
            "timestamp": pd.to_datetime([
                "2023-01-01 08:00",
                "2023-01-01 10:01",  # > 1h after row 0
            ]),
            "label": [0, 0],
            "customer_id": ["A", "A"],
        })
        result = add_velocity_features(df, windows=[1])
        assert result["txn_count_1h"].iloc[1] == 1.0  # only counts itself

    def test_different_customers_counted_independently(self):
        """Velocity counts must not bleed across customer boundaries."""
        df = pd.DataFrame({
            "transaction_id": [0, 1, 2, 3],
            "amount": [10.0] * 4,
            "timestamp": pd.to_datetime([
                "2023-01-01 10:00",
                "2023-01-01 10:10",
                "2023-01-01 10:20",
                "2023-01-01 10:30",
            ]),
            "label": [0] * 4,
            "customer_id": ["A", "A", "B", "B"],
        })
        result = add_velocity_features(df, windows=[1])
        # Each customer's counts reset independently
        assert result["txn_count_1h"].iloc[0] == 1.0  # A: 1st
        assert result["txn_count_1h"].iloc[1] == 2.0  # A: 2nd
        assert result["txn_count_1h"].iloc[2] == 1.0  # B: 1st
        assert result["txn_count_1h"].iloc[3] == 2.0  # B: 2nd

    def test_missing_customer_id_fills_nan(self):
        """When customer_id column is absent, velocity columns should be NaN."""
        df = _make_df().drop(columns=["customer_id"])
        result = add_velocity_features(df, windows=[1])
        assert result["txn_count_1h"].isna().all()

    def test_multiple_windows_all_created(self):
        """All requested window columns should be present in the result."""
        df = _make_df()
        result = add_velocity_features(df, windows=[1, 6, 24])
        for w in [1, 6, 24]:
            assert f"txn_count_{w}h" in result.columns


# ---------------------------------------------------------------------------
# add_amount_features
# ---------------------------------------------------------------------------

class TestAddAmountFeatures:
    def test_log_amount_equals_log1p(self):
        """log_amount should equal numpy log1p of the amount column."""
        df = _make_df()
        result = add_amount_features(df)
        expected = np.log1p(df["amount"])
        pd.testing.assert_series_equal(
            result["log_amount"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_amount_percentile_in_unit_interval(self):
        """amount_percentile should be in [0, 1] for all rows."""
        df = _make_df()
        result = add_amount_features(df)
        assert result["amount_percentile"].between(0, 1).all()

    def test_is_round_amount_flags_whole_dollars(self):
        """Amounts with no decimal part should be flagged as round."""
        df = pd.DataFrame({
            "transaction_id": [0, 1, 2],
            "amount": [100.0, 99.99, 500.0],
            "timestamp": pd.to_datetime(["2023-01-01"] * 3),
            "label": [0, 0, 0],
        })
        result = add_amount_features(df)
        assert result["is_round_amount"].iloc[0] == 1   # 100.00 — round
        assert result["is_round_amount"].iloc[1] == 0   # 99.99  — not round
        assert result["is_round_amount"].iloc[2] == 1   # 500.00 — round

    def test_amount_zscore_per_customer_when_customer_id_present(self):
        """amount_zscore should be computed per customer when customer_id exists."""
        df = pd.DataFrame({
            "transaction_id": [0, 1, 2, 3],
            "amount": [10.0, 20.0, 1000.0, 1010.0],
            "timestamp": pd.to_datetime(["2023-01-01"] * 4),
            "label": [0] * 4,
            "customer_id": ["A", "A", "B", "B"],
        })
        result = add_amount_features(df)
        # Customer A: mean=15, std=~7.07; row 0 should be negative z-score
        assert result["amount_zscore"].iloc[0] < 0
        assert result["amount_zscore"].iloc[1] > 0
        # Customer B: mean=1005; row 2 negative, row 3 positive
        assert result["amount_zscore"].iloc[2] < 0
        assert result["amount_zscore"].iloc[3] > 0

    def test_amount_zscore_falls_back_to_global_without_customer_id(self):
        """Without customer_id, zscore should still be computed globally."""
        df = _make_df().drop(columns=["customer_id"])
        result = add_amount_features(df)
        assert "amount_zscore" in result.columns
        assert result["amount_zscore"].notna().any()


# ---------------------------------------------------------------------------
# add_probe_features
# ---------------------------------------------------------------------------

class TestAddProbeFeatures:
    def test_small_probe_flag_below_threshold(self):
        """Amounts strictly below probe_threshold should be flagged as probes."""
        df = _make_df()
        result = add_probe_features(df, probe_threshold=10.0)
        # amount 9.0 (row 1) is below 10 → is_small_probe = 1
        assert result["is_small_probe"].iloc[1] == 1
        # amounts 100.0, 500.0, 50.0 are above → 0
        assert result["is_small_probe"].iloc[0] == 0
        assert result["is_small_probe"].iloc[2] == 0
        assert result["is_small_probe"].iloc[3] == 0

    def test_preceded_by_probe_detects_card_testing_pattern(self):
        """A large transaction within the window after a small probe should be flagged."""
        df = pd.DataFrame({
            "transaction_id": [0, 1],
            "amount": [5.0, 500.0],   # probe then cash-out
            "timestamp": pd.to_datetime([
                "2023-01-01 10:00",
                "2023-01-01 11:00",   # 1h after probe — within 24h window
            ]),
            "label": [0, 0],
            "customer_id": ["A", "A"],
        })
        result = add_probe_features(df, probe_threshold=10.0, window_hours=24.0)
        assert result["is_small_probe"].iloc[0] == 1
        assert result["preceded_by_probe"].iloc[1] == 1

    def test_probe_not_flagged_outside_window(self):
        """A transaction more than window_hours after the probe should not be flagged."""
        df = pd.DataFrame({
            "transaction_id": [0, 1],
            "amount": [5.0, 500.0],
            "timestamp": pd.to_datetime([
                "2023-01-01 10:00",
                "2023-01-02 11:00",  # 25h later — outside 24h window
            ]),
            "label": [0, 0],
            "customer_id": ["A", "A"],
        })
        result = add_probe_features(df, probe_threshold=10.0, window_hours=24.0)
        assert result["preceded_by_probe"].iloc[1] == 0

    def test_probe_does_not_cross_customer_boundary(self):
        """A probe from customer A should not flag a transaction from customer B."""
        df = pd.DataFrame({
            "transaction_id": [0, 1],
            "amount": [5.0, 500.0],
            "timestamp": pd.to_datetime([
                "2023-01-01 10:00",
                "2023-01-01 11:00",
            ]),
            "label": [0, 0],
            "customer_id": ["A", "B"],
        })
        result = add_probe_features(df, probe_threshold=10.0, window_hours=24.0)
        assert result["preceded_by_probe"].iloc[1] == 0

    def test_missing_customer_id_returns_nan(self):
        """Without customer_id, preceded_by_probe should be NaN for all rows."""
        df = _make_df().drop(columns=["customer_id"])
        result = add_probe_features(df)
        assert result["preceded_by_probe"].isna().all()


# ---------------------------------------------------------------------------
# engineer_features (integration)
# ---------------------------------------------------------------------------

class TestEngineerFeatures:
    def test_all_expected_columns_present(self):
        """engineer_features should produce all derived feature columns."""
        df = _make_df()
        result = engineer_features(df)
        expected = {
            "hour_of_day", "day_of_week", "is_weekend", "is_night",
            "txn_count_1h", "txn_count_6h", "txn_count_24h",
            "log_amount", "amount_percentile", "amount_zscore", "is_round_amount",
            "is_small_probe", "preceded_by_probe",
        }
        assert expected.issubset(result.columns)

    def test_row_count_unchanged(self):
        """Feature engineering must not add or drop rows."""
        df = _make_df()
        result = engineer_features(df)
        assert len(result) == len(df)

    def test_does_not_mutate_input(self):
        """engineer_features should not modify the original DataFrame."""
        df = _make_df()
        original_cols = set(df.columns)
        engineer_features(df)
        assert set(df.columns) == original_cols

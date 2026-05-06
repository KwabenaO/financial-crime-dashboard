"""Tests for rule_engine module."""

import pytest
import pandas as pd
import numpy as np
from src.rule_engine import (
    apply_all_rules,
    flag_amount_threshold,
    flag_velocity,
    flag_time_anomaly,
    flag_repeat_pattern,
)


def _base_df(**overrides) -> pd.DataFrame:
    """Minimal valid DataFrame for rule engine tests."""
    data = {
        "transaction_id": [0, 1, 2, 3, 4],
        "amount": [10.0, 20.0, 30.0, 40.0, 500.0],
        "timestamp": pd.to_datetime([
            "2023-01-01 14:00", "2023-01-01 14:10", "2023-01-01 14:20",
            "2023-01-01 14:30", "2023-01-01 02:00",
        ]),
        "label": [0, 0, 0, 0, 1],
        "hour_of_day": [14, 14, 14, 14, 2],
        "customer_id": ["A", "A", "A", "A", "B"],
        "txn_count_1h": [1.0, 2.0, 3.0, 4.0, 1.0],
    }
    data.update(overrides)
    return pd.DataFrame(data)


class TestAmountThreshold:
    def test_flags_above_percentile(self):
        """Transactions above the 95th percentile should be flagged."""
        df = pd.DataFrame({"amount": list(range(1, 101))})
        result = flag_amount_threshold(df, percentile=0.95)
        # quantile(0.95) of 1..100 ≈ 95.05; amounts 96-100 flagged
        assert result.sum() == 5
        assert result.iloc[99] is np.bool_(True)

    def test_does_not_flag_normal_amounts(self):
        """Transactions well below the threshold should not be flagged."""
        df = pd.DataFrame({"amount": [10.0, 20.0, 30.0, 40.0, 50.0]})
        result = flag_amount_threshold(df, percentile=0.95)
        # Only the top 5% (the single highest value) is flagged
        assert result.iloc[0] is np.bool_(False)
        assert result.iloc[1] is np.bool_(False)

    def test_returns_boolean_series(self):
        df = pd.DataFrame({"amount": [100.0, 200.0, 300.0]})
        result = flag_amount_threshold(df)
        assert result.dtype == bool


class TestVelocity:
    def test_flags_high_frequency_customer(self):
        """A customer exceeding max_txns in the window should be flagged."""
        df = _base_df(txn_count_1h=[1.0, 2.0, 3.0, 4.0, 6.0])
        result = flag_velocity(df, max_txns=5, window_hours=1)
        assert result.iloc[4] is np.bool_(True)

    def test_does_not_flag_below_max(self):
        """Customers within the velocity limit should not be flagged."""
        df = _base_df(txn_count_1h=[1.0, 2.0, 3.0, 4.0, 5.0])
        result = flag_velocity(df, max_txns=5, window_hours=1)
        assert result.any() == False

    def test_falls_back_to_recompute_when_column_missing(self):
        """Should recompute velocity when txn_count column is absent."""
        df = pd.DataFrame({
            "transaction_id": range(4),
            "amount": [100.0] * 4,
            "timestamp": pd.to_datetime([
                "2023-01-01 14:00", "2023-01-01 14:10",
                "2023-01-01 14:20", "2023-01-01 14:30",
            ]),
            "label": [0] * 4,
            "customer_id": ["A"] * 4,
        })
        result = flag_velocity(df, max_txns=2, window_hours=1)
        # 4 transactions in 30 minutes; rows 2 and 3 exceed max_txns=2
        assert result.iloc[2] is np.bool_(True)
        assert result.iloc[3] is np.bool_(True)


class TestTimeAnomaly:
    def test_flags_night_transactions(self):
        """Transactions between 10pm and 6am should be flagged."""
        df = pd.DataFrame({"hour_of_day": [2, 14, 23, 8, 0]})
        result = flag_time_anomaly(df, night_start=22, night_end=6)
        assert result.iloc[0] is np.bool_(True)   # 2am
        assert result.iloc[1] is np.bool_(False)  # 2pm
        assert result.iloc[2] is np.bool_(True)   # 11pm
        assert result.iloc[3] is np.bool_(False)  # 8am
        assert result.iloc[4] is np.bool_(True)   # midnight

    def test_falls_back_to_timestamp_when_column_missing(self):
        """Should extract hour from timestamp when hour_of_day is absent."""
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(["2023-01-01 03:00", "2023-01-01 12:00"]),
        })
        result = flag_time_anomaly(df)
        assert result.iloc[0] is np.bool_(True)   # 3am
        assert result.iloc[1] is np.bool_(False)  # noon

    def test_boundary_hours_not_flagged(self):
        """6am and 9pm are within business hours and should not be flagged."""
        df = pd.DataFrame({"hour_of_day": [6, 21]})
        result = flag_time_anomaly(df, night_start=22, night_end=6)
        assert result.iloc[0] is np.bool_(False)  # exactly 6am — end of night (exclusive)
        assert result.iloc[1] is np.bool_(False)  # 9pm — before night_start


class TestRepeatPattern:
    def test_flags_duplicate_amounts_to_same_merchant(self):
        """Same amount to same merchant within 30 minutes should be flagged."""
        df = pd.DataFrame({
            "transaction_id": [0, 1, 2],
            "amount": [100.0, 100.0, 100.0],
            "timestamp": pd.to_datetime([
                "2023-01-01 14:00",
                "2023-01-01 14:15",  # within 30 min of row 0
                "2023-01-01 16:00",  # 2 hours later — outside window
            ]),
            "label": [0, 0, 0],
            "merchant": ["Amazon", "Amazon", "Amazon"],
        })
        result = flag_repeat_pattern(df, window_minutes=30)
        assert result.iloc[0] is np.bool_(True)   # original, has a successor in window
        assert result.iloc[1] is np.bool_(True)   # repeat, has a predecessor in window
        assert result.iloc[2] is np.bool_(False)  # outside window

    def test_different_merchants_not_flagged(self):
        """Same amount to different merchants should not trigger the flag."""
        df = pd.DataFrame({
            "transaction_id": [0, 1],
            "amount": [100.0, 100.0],
            "timestamp": pd.to_datetime(["2023-01-01 14:00", "2023-01-01 14:05"]),
            "label": [0, 0],
            "merchant": ["Amazon", "Walmart"],
        })
        result = flag_repeat_pattern(df, window_minutes=30)
        assert result.any() == False

    def test_returns_false_when_merchant_absent(self):
        """Missing merchant column should return all-False Series without raising."""
        df = pd.DataFrame({
            "amount": [100.0, 100.0],
            "timestamp": pd.to_datetime(["2023-01-01 14:00", "2023-01-01 14:05"]),
        })
        result = flag_repeat_pattern(df)
        assert result.any() == False


class TestApplyAllRules:
    def test_rule_score_sums_flags(self):
        """rule_score should equal the count of triggered flags per transaction."""
        df = _base_df()
        df["merchant"] = ["X", "X", "Y", "Z", "Z"]
        result = apply_all_rules(df)
        assert "rule_score" in result.columns
        flag_cols = [c for c in result.columns if c.startswith("flag_")]
        expected = result[flag_cols].sum(axis=1).astype(int)
        pd.testing.assert_series_equal(result["rule_score"], expected, check_names=False)

    def test_probe_flag_added_when_column_present(self):
        """flag_probe should appear when preceded_by_probe is in the input."""
        df = _base_df()
        df["preceded_by_probe"] = [0, 0, 1, 1, 0]
        result = apply_all_rules(df)
        assert "flag_probe" in result.columns
        assert result["flag_probe"].iloc[2] is np.bool_(True)

    def test_custom_thresholds_override_defaults(self):
        """Passing custom thresholds should change which transactions are flagged."""
        df = _base_df()
        # Lower the amount percentile so more transactions are flagged
        result_strict = apply_all_rules(df, thresholds={"amount_percentile": 0.50})
        result_default = apply_all_rules(df)
        assert result_strict["flag_high_amount"].sum() >= result_default["flag_high_amount"].sum()

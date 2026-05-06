"""Tests for data_loader module."""

import pytest
import pandas as pd
from src.data_loader import (
    REQUIRED_COLUMNS,
    validate_schema,
    get_column_mapping_suggestions,
)


def _make_valid_df() -> pd.DataFrame:
    return pd.DataFrame({
        "transaction_id": [1, 2, 3],
        "amount": [100.0, 250.50, 75.0],
        "timestamp": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]),
        "label": [0, 1, 0],
    })


class TestValidateSchema:
    """Test dataset validation logic."""

    def test_valid_schema_passes(self):
        """A DataFrame with all required columns should pass validation."""
        df = _make_valid_df()
        result = validate_schema(df)
        assert set(REQUIRED_COLUMNS).issubset(result.columns)
        assert len(result) == len(df)

    def test_missing_required_column_raises(self):
        """A DataFrame missing a required column should raise ValueError."""
        df = _make_valid_df().drop(columns=["timestamp"])
        with pytest.raises(ValueError, match="timestamp"):
            validate_schema(df)

    def test_column_mapping_renames_correctly(self):
        """Column mapping should rename user columns to standard names."""
        df = pd.DataFrame({
            "txn_id": [1, 2],
            "txn_amount": [100.0, 200.0],
            "txn_time": pd.to_datetime(["2023-01-01", "2023-01-02"]),
            "is_fraud": [0, 1],
        })
        mapping = {
            "txn_id": "transaction_id",
            "txn_amount": "amount",
            "txn_time": "timestamp",
            "is_fraud": "label",
        }
        result = validate_schema(df, mapping)
        assert set(REQUIRED_COLUMNS).issubset(result.columns)
        assert "txn_id" not in result.columns

    def test_invalid_label_type_raises(self):
        """Label column with non-binary values should raise ValueError."""
        df = _make_valid_df()
        df["label"] = [2, 3, 4]
        with pytest.raises(ValueError, match="label"):
            validate_schema(df)

    def test_non_numeric_amount_raises(self):
        """Amount column with string values should raise ValueError."""
        df = _make_valid_df()
        df["amount"] = ["high", "medium", "low"]
        with pytest.raises(ValueError, match="amount"):
            validate_schema(df)

    def test_null_in_required_column_raises(self):
        """Null values in a required column should raise ValueError."""
        df = _make_valid_df()
        df.loc[0, "amount"] = None
        with pytest.raises(ValueError, match="null"):
            validate_schema(df)

    def test_does_not_mutate_input(self):
        """validate_schema should return a new DataFrame, not modify the original."""
        df = pd.DataFrame({
            "txn_id": [1],
            "txn_amount": [100.0],
            "txn_time": pd.to_datetime(["2023-01-01"]),
            "is_fraud": [0],
        })
        original_cols = list(df.columns)
        mapping = {"txn_id": "transaction_id", "txn_amount": "amount",
                   "txn_time": "timestamp", "is_fraud": "label"}
        validate_schema(df, mapping)
        assert list(df.columns) == original_cols

    def test_string_timestamp_parsed(self):
        """String timestamps should be parsed to datetime without raising."""
        df = _make_valid_df()
        df["timestamp"] = ["2023-01-01", "2023-01-02", "2023-01-03"]
        result = validate_schema(df)
        assert pd.api.types.is_datetime64_any_dtype(result["timestamp"])


class TestGetColumnMappingSuggestions:
    """Test fuzzy column name suggestion logic."""

    def test_exact_names_map_to_themselves(self):
        """Standard column names should map to themselves."""
        df = pd.DataFrame(columns=["transaction_id", "amount", "timestamp", "label"])
        suggestions = get_column_mapping_suggestions(df)
        assert suggestions.get("transaction_id") == "transaction_id"
        assert suggestions.get("amount") == "amount"
        assert suggestions.get("label") == "label"

    def test_fuzzy_names_are_suggested(self):
        """Close-but-not-exact column names should still be matched."""
        df = pd.DataFrame(columns=["txn_id", "amt", "date_time", "is_fraud"])
        suggestions = get_column_mapping_suggestions(df)
        assert suggestions.get("amount") == "amt"
        assert suggestions.get("label") == "is_fraud"

    def test_no_suggestion_below_threshold(self):
        """Completely unrelated column names should not produce suggestions."""
        df = pd.DataFrame(columns=["x1", "y2", "z3", "w4"])
        suggestions = get_column_mapping_suggestions(df)
        assert suggestions == {}

    def test_returns_dict(self):
        """Result is always a dict even with an empty DataFrame."""
        df = pd.DataFrame()
        result = get_column_mapping_suggestions(df)
        assert isinstance(result, dict)

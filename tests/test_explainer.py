"""Tests for explainer module."""

import numpy as np
import pandas as pd
import pytest
import shap
from src.explainer import (
    _select_positive_class,
    create_explainer,
    explain_transaction,
    get_global_importance,
)
from src.model import train_model


def _make_separable_df(n_legit: int = 150, n_fraud: int = 20, seed: int = 0) -> pd.DataFrame:
    """Synthetic two-class dataset where fraud is clearly offset from legitimate."""
    rng = np.random.default_rng(seed)
    X_legit = rng.standard_normal((n_legit, 4))
    X_fraud = rng.standard_normal((n_fraud, 4)) + 6.0
    X = np.vstack([X_legit, X_fraud])
    y = np.array([0] * n_legit + [1] * n_fraud)
    df = pd.DataFrame(X, columns=["f1", "f2", "f3", "f4"])
    df["label"] = y
    return df


@pytest.fixture(scope="module")
def trained_artifacts():
    """Train once and share the model, explainer, and data across all tests."""
    df = _make_separable_df()
    feature_cols = ["f1", "f2", "f3", "f4"]
    model, X_test, y_test, feature_cols = train_model(
        df, feature_cols, use_smote=True
    )
    background = df[feature_cols].sample(50, random_state=42)
    explainer = create_explainer(model, background)
    return model, explainer, X_test, df, feature_cols


# ---------------------------------------------------------------------------
# create_explainer
# ---------------------------------------------------------------------------

class TestCreateExplainer:
    def test_returns_tree_explainer(self, trained_artifacts):
        """create_explainer should return a shap.TreeExplainer instance."""
        _, explainer, _, _, _ = trained_artifacts
        assert isinstance(explainer, shap.TreeExplainer)

    def test_has_expected_value_attribute(self, trained_artifacts):
        """TreeExplainer must expose an expected_value (the base value)."""
        _, explainer, _, _, _ = trained_artifacts
        assert hasattr(explainer, "expected_value")

    def test_expected_value_is_finite(self, trained_artifacts):
        """Base value should be a finite number, not NaN or inf."""
        _, explainer, _, _, _ = trained_artifacts
        ev = explainer.expected_value
        # Some SHAP versions return a scalar, others a 1-element array
        val = float(ev[-1]) if hasattr(ev, "__len__") else float(ev)
        assert np.isfinite(val)


# ---------------------------------------------------------------------------
# explain_transaction
# ---------------------------------------------------------------------------

class TestExplainTransaction:
    def test_returns_explanation_object(self, trained_artifacts):
        """explain_transaction should return a shap.Explanation."""
        _, explainer, _, df, feature_cols = trained_artifacts
        row = df.iloc[[0]]
        result = explain_transaction(explainer, row, feature_cols)
        assert isinstance(result, shap.Explanation)

    def test_shape_matches_feature_count(self, trained_artifacts):
        """Explanation values shape should equal (n_features,) for a single row."""
        _, explainer, _, df, feature_cols = trained_artifacts
        row = df.iloc[[0]]
        result = explain_transaction(explainer, row, feature_cols)
        assert result.values.shape == (len(feature_cols),)

    def test_has_base_values(self, trained_artifacts):
        """Explanation should carry a scalar base value."""
        _, explainer, _, df, feature_cols = trained_artifacts
        row = df.iloc[[0]]
        result = explain_transaction(explainer, row, feature_cols)
        assert hasattr(result, "base_values")
        assert np.isfinite(float(result.base_values))

    def test_has_feature_names(self, trained_artifacts):
        """feature_names on the returned Explanation should match feature_cols."""
        _, explainer, _, df, feature_cols = trained_artifacts
        row = df.iloc[[0]]
        result = explain_transaction(explainer, row, feature_cols)
        assert list(result.feature_names) == feature_cols

    def test_shap_values_sum_to_score_minus_base(self, trained_artifacts):
        """SHAP values must add up to (prediction - base_value), within tolerance."""
        model, explainer, _, df, feature_cols = trained_artifacts
        row = df.iloc[[0]]
        result = explain_transaction(explainer, row, feature_cols)
        predicted_log_odds = float(model.predict(row[feature_cols], output_margin=True))
        base = float(result.base_values)
        shap_sum = float(result.values.sum())
        assert abs((base + shap_sum) - predicted_log_odds) < 1e-3

    def test_fraud_row_has_positive_dominant_shap(self, trained_artifacts):
        """For a clearly fraudulent row the net SHAP contribution should be positive."""
        _, explainer, _, df, feature_cols = trained_artifacts
        fraud_row = df[df["label"] == 1].iloc[[0]]
        result = explain_transaction(explainer, fraud_row, feature_cols)
        assert result.values.sum() > 0


# ---------------------------------------------------------------------------
# get_global_importance
# ---------------------------------------------------------------------------

class TestGetGlobalImportance:
    def test_returns_dataframe(self, trained_artifacts):
        """get_global_importance should return a pandas DataFrame."""
        _, explainer, _, df, feature_cols = trained_artifacts
        result = get_global_importance(explainer, df[feature_cols])
        assert isinstance(result, pd.DataFrame)

    def test_has_correct_columns(self, trained_artifacts):
        """Result must have exactly 'feature' and 'importance' columns."""
        _, explainer, _, df, feature_cols = trained_artifacts
        result = get_global_importance(explainer, df[feature_cols])
        assert list(result.columns) == ["feature", "importance"]

    def test_max_features_limits_rows(self, trained_artifacts):
        """Number of rows should not exceed max_features."""
        _, explainer, _, df, feature_cols = trained_artifacts
        for max_f in [1, 2, 4]:
            result = get_global_importance(explainer, df[feature_cols], max_features=max_f)
            assert len(result) == min(max_f, len(feature_cols))

    def test_sorted_descending_by_importance(self, trained_artifacts):
        """Rows should be ordered from highest to lowest importance."""
        _, explainer, _, df, feature_cols = trained_artifacts
        result = get_global_importance(explainer, df[feature_cols])
        importances = result["importance"].tolist()
        assert importances == sorted(importances, reverse=True)

    def test_all_importance_values_nonnegative(self, trained_artifacts):
        """Mean |SHAP| is always non-negative by definition."""
        _, explainer, _, df, feature_cols = trained_artifacts
        result = get_global_importance(explainer, df[feature_cols])
        assert (result["importance"] >= 0).all()

    def test_feature_names_are_strings(self, trained_artifacts):
        """Feature column should contain string names, not integers."""
        _, explainer, _, df, feature_cols = trained_artifacts
        result = get_global_importance(explainer, df[feature_cols])
        assert result["feature"].dtype == object


# ---------------------------------------------------------------------------
# _select_positive_class (internal helper)
# ---------------------------------------------------------------------------

class TestSelectPositiveClass:
    def test_2d_input_returned_unchanged(self):
        """A 2D Explanation (n_samples, n_features) should pass through unmodified."""
        values = np.array([[0.1, -0.2, 0.3]])
        exp = shap.Explanation(
            values=values,
            base_values=np.array([0.5]),
            data=values,
        )
        result = _select_positive_class(exp)
        np.testing.assert_array_equal(result.values, values)

    def test_3d_input_selects_positive_class_slice(self):
        """A 3D Explanation (n, f, 2) should return the index-1 slice."""
        values_3d = np.zeros((2, 3, 2))
        values_3d[:, :, 0] = 99.0   # negative class — should be discarded
        values_3d[:, :, 1] = 1.0    # positive class — should be kept
        exp = shap.Explanation(
            values=values_3d,
            base_values=np.array([0.5, 0.5]),
            data=np.zeros((2, 3)),
        )
        result = _select_positive_class(exp)
        assert result.values.shape == (2, 3)
        np.testing.assert_array_equal(result.values, 1.0)

"""Tests for model module."""

import os
import pytest
import numpy as np
import pandas as pd
from src.model import train_model, evaluate_model, predict_risk_scores, save_model, load_model


def _make_separable_df(n_legit: int = 200, n_fraud: int = 20, seed: int = 42) -> pd.DataFrame:
    """Synthetic dataset where fraud cluster is clearly separated from legitimate."""
    rng = np.random.default_rng(seed)
    X_legit = rng.standard_normal((n_legit, 3))
    X_fraud = rng.standard_normal((n_fraud, 3)) + 5.0  # offset makes classes separable
    X = np.vstack([X_legit, X_fraud])
    y = np.array([0] * n_legit + [1] * n_fraud)
    df = pd.DataFrame(X, columns=["f1", "f2", "f3"])
    df["label"] = y
    return df


@pytest.fixture(scope="module")
def trained():
    """Train once and share across all tests in this module."""
    df = _make_separable_df()
    model, X_test, y_test, feature_cols = train_model(df, ["f1", "f2", "f3"], use_smote=True)
    return model, X_test, y_test, df, feature_cols


class TestTrainModel:
    def test_returns_trained_model(self, trained):
        """Training should return a model object and test data."""
        model, X_test, y_test, _, feature_cols = trained
        assert model is not None
        assert len(X_test) > 0
        assert len(y_test) == len(X_test)
        assert feature_cols == ["f1", "f2", "f3"]

    def test_test_split_size(self, trained):
        """Test set should be approximately 20% of the full dataset."""
        _, X_test, _, df, _ = trained
        expected = int(len(df) * 0.2)
        assert abs(len(X_test) - expected) <= 2  # allow rounding

    def test_handles_class_imbalance(self, trained):
        """Model trained with SMOTE should achieve recall > 0.5 on separable data."""
        model, X_test, y_test, _, _ = trained
        metrics = evaluate_model(model, X_test, y_test)
        assert metrics["recall"] > 0.5

    def test_falls_back_when_too_few_fraud_samples(self):
        """Should not raise when minority class is too small for SMOTE."""
        df = _make_separable_df(n_legit=100, n_fraud=3)
        model, X_test, y_test, _ = train_model(
            df, ["f1", "f2", "f3"], use_smote=True
        )
        assert model is not None

    def test_drops_non_numeric_columns(self):
        """Non-numeric feature columns should be silently dropped."""
        df = _make_separable_df()
        df["category"] = "A"
        model, _, _, returned_cols = train_model(
            df, ["f1", "f2", "f3", "category"], use_smote=False
        )
        assert "category" not in returned_cols
        assert model is not None

    def test_smote_false_uses_scale_pos_weight(self):
        """use_smote=False should still produce a trained model."""
        df = _make_separable_df()
        model, X_test, y_test, _ = train_model(df, ["f1", "f2", "f3"], use_smote=False)
        assert model is not None


class TestEvaluateModel:
    def test_returns_expected_metric_keys(self, trained):
        """Evaluation should return all five expected keys."""
        model, X_test, y_test, _, _ = trained
        metrics = evaluate_model(model, X_test, y_test)
        assert set(metrics.keys()) == {"precision", "recall", "f1", "auc_roc", "confusion_matrix"}

    def test_metrics_in_valid_range(self, trained):
        """All scalar metrics should be valid probabilities between 0 and 1."""
        model, X_test, y_test, _, _ = trained
        metrics = evaluate_model(model, X_test, y_test)
        for key in ("precision", "recall", "f1", "auc_roc"):
            assert 0.0 <= metrics[key] <= 1.0, f"{key} out of range"

    def test_confusion_matrix_shape(self, trained):
        """Confusion matrix should be 2×2."""
        model, X_test, y_test, _, _ = trained
        metrics = evaluate_model(model, X_test, y_test)
        cm = metrics["confusion_matrix"]
        assert cm.shape == (2, 2)

    def test_auc_roc_better_than_random(self, trained):
        """AUC-ROC should exceed 0.5 on clearly separable data."""
        model, X_test, y_test, _, _ = trained
        metrics = evaluate_model(model, X_test, y_test)
        assert metrics["auc_roc"] > 0.5


class TestPredictRiskScores:
    def test_scores_between_zero_and_one(self, trained):
        """All risk scores should be valid probabilities."""
        model, _, _, df, feature_cols = trained
        scores = predict_risk_scores(model, df, feature_cols)
        assert scores.min() >= 0.0
        assert scores.max() <= 1.0

    def test_score_length_matches_input(self, trained):
        """One score should be returned per row."""
        model, _, _, df, feature_cols = trained
        scores = predict_risk_scores(model, df, feature_cols)
        assert len(scores) == len(df)

    def test_scores_indexed_like_df(self, trained):
        """Score index should match the input DataFrame's index."""
        model, _, _, df, feature_cols = trained
        scores = predict_risk_scores(model, df, feature_cols)
        pd.testing.assert_index_equal(scores.index, df.index)

    def test_fraud_scores_higher_than_legit(self, trained):
        """Mean fraud score should exceed mean legitimate score."""
        model, _, _, df, feature_cols = trained
        scores = predict_risk_scores(model, df, feature_cols)
        assert scores[df["label"] == 1].mean() > scores[df["label"] == 0].mean()


class TestSaveLoadModel:
    def test_save_and_load_roundtrip(self, trained, tmp_path):
        """Saved and reloaded model should produce identical predictions."""
        model, _, _, df, feature_cols = trained
        path = str(tmp_path / "test_model.joblib")
        save_model(model, path)
        loaded = load_model(path)
        original_scores = predict_risk_scores(model, df, feature_cols)
        loaded_scores = predict_risk_scores(loaded, df, feature_cols)
        pd.testing.assert_series_equal(original_scores, loaded_scores)

    def test_load_raises_when_file_missing(self):
        """load_model should raise FileNotFoundError for a nonexistent path."""
        with pytest.raises(FileNotFoundError):
            load_model("nonexistent/path/model.joblib")

    def test_save_creates_parent_directories(self, trained, tmp_path):
        """save_model should create any missing parent directories."""
        model, _, _, _, _ = trained
        path = str(tmp_path / "nested" / "dir" / "model.joblib")
        save_model(model, path)
        assert os.path.exists(path)

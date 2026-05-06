"""
SHAP Explainability Module.

Generates SHAP explanations for individual flagged transactions
so users can understand why a transaction was flagged by the ML model.

Provides both individual waterfall plots and global feature importance.

TreeExplainer is used because it computes exact Shapley values for tree-based
models in O(TLD) time — orders of magnitude faster than KernelSHAP, and exact
rather than approximate. This makes per-transaction explanations in the
dashboard feasible.
"""

import logging

import numpy as np
import pandas as pd
import shap

logger = logging.getLogger("fraud_dashboard")


def create_explainer(model, X_background: pd.DataFrame) -> shap.TreeExplainer:
    """Create a SHAP TreeExplainer for the trained XGBoost model.

    The background dataset anchors the base value (expected model output)
    used in waterfall plots. Pass a random sample of training data here —
    500 rows is enough to get a stable expected value without being slow.

    Args:
        model: Trained XGBoost model.
        X_background: Background dataset (typically a sample of training data).

    Returns:
        SHAP TreeExplainer instance.
    """
    explainer = shap.TreeExplainer(model, X_background)
    logger.info(
        "TreeExplainer created; base value: %.4f", explainer.expected_value
        if not hasattr(explainer.expected_value, "__len__")
        else explainer.expected_value[-1]
    )
    return explainer


def explain_transaction(
    explainer: shap.TreeExplainer,
    transaction: pd.DataFrame,
    feature_names: list,
) -> shap.Explanation:
    """Generate SHAP explanation for a single transaction.

    The returned Explanation object can be passed directly to
    shap.plots.waterfall() to render a per-feature contribution chart.

    Args:
        explainer: SHAP TreeExplainer instance.
        transaction: Single-row DataFrame of the transaction to explain.
        feature_names: List of feature column names used during training.

    Returns:
        SHAP Explanation object (single row) ready for a waterfall plot.
    """
    X = transaction[feature_names]
    shap_values = explainer(X)
    shap_values = _select_positive_class(shap_values)
    return shap_values[0]


def get_global_importance(
    explainer: shap.TreeExplainer,
    X: pd.DataFrame,
    max_features: int = 15,
) -> pd.DataFrame:
    """Compute global feature importance using mean absolute SHAP values.

    Mean |SHAP| across all rows gives a consistent, model-agnostic importance
    measure that accounts for feature interactions — unlike XGBoost's built-in
    gain importance, which ignores how features combine.

    Pass a sample of the dataset rather than the full 590K rows to keep
    computation time under a few seconds.

    Args:
        explainer: SHAP TreeExplainer instance.
        X: Feature DataFrame (a representative sample is sufficient).
        max_features: Number of top features to return.

    Returns:
        DataFrame with columns ['feature', 'importance'] sorted descending,
        limited to max_features rows.
    """
    shap_values = explainer(X)
    shap_values = _select_positive_class(shap_values)

    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)

    importance_df = (
        pd.DataFrame({"feature": X.columns.tolist(), "importance": mean_abs_shap})
        .sort_values("importance", ascending=False)
        .head(max_features)
        .reset_index(drop=True)
    )
    return importance_df


def _select_positive_class(shap_values: shap.Explanation) -> shap.Explanation:
    """Return SHAP values for the positive (fraud) class.

    For binary XGBoost models some SHAP versions return shape
    (n_samples, n_features, 2) — one slice per class. This helper
    normalises to (n_samples, n_features) so callers don't need to
    branch on the SHAP version.
    """
    if len(shap_values.shape) == 3:
        # Take the positive class slice (index 1)
        return shap_values[:, :, 1]
    return shap_values

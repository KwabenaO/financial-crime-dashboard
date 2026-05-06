"""
ML Model Module.

Trains and evaluates an XGBoost classifier for fraud detection.
Handles class imbalance via SMOTE or scale_pos_weight.
Evaluates with precision, recall, F1, and AUC-ROC.
Saves trained model for reuse with joblib.
"""

import os
import logging

import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

logger = logging.getLogger("fraud_dashboard")

# XGBoost hyperparameters — conservative defaults suitable for tabular fraud data
_XGB_PARAMS = {
    "n_estimators": 200,
    "learning_rate": 0.1,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "aucpr",  # average precision — better than AUC for imbalanced data
    "random_state": 42,
    "n_jobs": -1,
}


def train_model(
    df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str = "label",
    test_size: float = 0.2,
    use_smote: bool = True,
) -> Tuple:
    """Train an XGBoost classifier on the prepared dataset.

    Non-numeric feature columns are dropped automatically. NaN values in
    numeric features are filled with 0 before SMOTE; XGBoost handles NaN
    natively during prediction. If the minority class is too small for SMOTE
    (< 6 samples in training), falls back to scale_pos_weight automatically.

    Args:
        df: Feature-engineered DataFrame.
        feature_cols: List of column names to use as features.
        label_col: Name of the binary label column (0/1).
        test_size: Fraction of data reserved for testing.
        use_smote: Whether to apply SMOTE oversampling for class imbalance.

    Returns:
        Tuple of (trained_model, X_test, y_test, feature_cols) where
        feature_cols is the filtered list of numeric columns actually used.
    """
    # Keep only numeric columns — XGBoost needs encoded categoricals otherwise
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < len(feature_cols):
        dropped = set(feature_cols) - set(numeric_cols)
        logger.warning("Dropping non-numeric feature columns: %s", dropped)
    feature_cols = numeric_cols

    X = df[feature_cols]
    y = df[label_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )

    fraud_count = int((y_train == 1).sum())
    legit_count = int((y_train == 0).sum())
    logger.info("Training set: %d legitimate, %d fraud", legit_count, fraud_count)

    if use_smote and fraud_count >= 6:
        from imblearn.over_sampling import SMOTE

        k = min(5, fraud_count - 1)
        smote = SMOTE(k_neighbors=k, random_state=42)
        X_train_fit, y_train_fit = smote.fit_resample(X_train.fillna(0), y_train)
        logger.info("After SMOTE: %d training samples", len(y_train_fit))
        model = XGBClassifier(**_XGB_PARAMS)
    else:
        if use_smote:
            logger.warning(
                "Too few fraud samples for SMOTE (%d); using scale_pos_weight instead",
                fraud_count,
            )
        scale_pos_weight = legit_count / fraud_count if fraud_count > 0 else 1.0
        X_train_fit, y_train_fit = X_train, y_train
        model = XGBClassifier(**_XGB_PARAMS, scale_pos_weight=scale_pos_weight)

    model.fit(X_train_fit, y_train_fit)
    logger.info("Model trained successfully")

    return model, X_test, y_test, feature_cols


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
    """Evaluate the trained model and return performance metrics.

    Uses 0.5 decision threshold for precision, recall, and F1.
    AUC-ROC uses raw probabilities so it is threshold-independent.

    Args:
        model: Trained XGBoost model.
        X_test: Test features.
        y_test: Test labels.

    Returns:
        Dict with keys: precision, recall, f1, auc_roc, confusion_matrix.
        confusion_matrix is a 2×2 numpy array [[TN, FP], [FN, TP]].
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_test, y_proba)),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }


def predict_risk_scores(model, df: pd.DataFrame, feature_cols: List[str]) -> pd.Series:
    """Generate fraud probability scores for all transactions.

    Args:
        model: Trained XGBoost model.
        df: DataFrame with feature columns.
        feature_cols: List of feature column names (must match training columns).

    Returns:
        Series of fraud probabilities (0.0 to 1.0), indexed like df.
    """
    X = df[feature_cols]
    probabilities = model.predict_proba(X)[:, 1]
    return pd.Series(probabilities, index=df.index, name="risk_score")


def save_model(model, path: str = "models/xgb_fraud_model.joblib") -> None:
    """Save trained model to disk using joblib.

    Creates parent directories if they do not exist.

    Args:
        model: Trained model object.
        path: File path for saved model.
    """
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    joblib.dump(model, path)
    logger.info("Model saved to %s", path)


def load_model(path: str = "models/xgb_fraud_model.joblib"):
    """Load a previously trained model from disk.

    Args:
        path: File path of saved model.

    Returns:
        Loaded model object.

    Raises:
        FileNotFoundError: If the model file does not exist at the given path.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"No saved model found at '{path}'")
    model = joblib.load(path)
    logger.info("Model loaded from %s", path)
    return model

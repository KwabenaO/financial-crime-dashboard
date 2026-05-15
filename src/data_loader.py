"""
Data Loading and Validation Module.

Handles three data sources:
1. Default IEEE-CIS Fraud Detection dataset from data/raw/
2. Bundled sample dataset from data/sample/ (fallback when full dataset absent)
3. User-uploaded CSV via Streamlit file uploader

Validates schema, maps columns if needed, and returns a clean DataFrame
ready for feature engineering and model training.
"""

import os
import difflib
import logging
import pandas as pd
from typing import Optional

logger = logging.getLogger("fraud_dashboard")

# Minimum columns required for the pipeline to work
REQUIRED_COLUMNS = ["transaction_id", "amount", "timestamp", "label"]

# Optional columns that enable additional rule-based checks
OPTIONAL_COLUMNS = ["merchant", "category", "customer_id"]

# IEEE-CIS raw column names → standardized pipeline names
IEEE_COLUMN_MAP = {
    "TransactionID": "transaction_id",
    "TransactionAmt": "amount",
    "TransactionDT": "timestamp",
    "isFraud": "label",
    "ProductCD": "category",
}

# TransactionDT is a seconds-offset from this reference date (competition data spans 2017–2019)
_IEEE_REFERENCE_DATE = pd.Timestamp("2017-11-30")

# Bundled sample dataset used as a fallback when the full Kaggle files are absent
SAMPLE_DATASET_PATH = "data/sample/sample_40k.csv"

# Candidate synonyms for each required column — used by fuzzy name matching
_COLUMN_CANDIDATES = {
    "transaction_id": ["transaction_id", "txn_id", "trans_id", "id", "transactionid", "transaction"],
    "amount": ["amount", "amt", "transaction_amount", "txn_amount", "value", "price", "sum"],
    "timestamp": ["timestamp", "time", "date", "datetime", "txn_time", "trans_time", "created_at", "date_time"],
    "label": ["label", "fraud", "is_fraud", "isfraud", "target", "class", "fraudulent", "flag"],
}


def load_default_dataset(path: str = "data/raw/") -> pd.DataFrame:
    """Load the IEEE-CIS fraud detection dataset from local storage.

    First tries to load the full Kaggle dataset (two CSV files in ``path``).
    If either file is missing, falls back to the bundled sample at
    ``data/sample/sample_40k.csv``.

    Full dataset expects:
    - train_transaction.csv (transactions with fraud labels)
    - train_identity.csv (device and identity info)

    Both the full dataset and the sample use IEEE-CIS column names and are
    processed identically: joined on TransactionID, renamed to the standard
    pipeline schema, and timestamp converted from seconds-offset to datetime.

    Args:
        path: Directory containing the raw Kaggle CSV files.

    Returns:
        Cleaned DataFrame with standardized column names.

    Raises:
        FileNotFoundError: If neither the full dataset nor the sample file
            can be found.
    """
    txn_path = os.path.join(path, "train_transaction.csv")
    id_path = os.path.join(path, "train_identity.csv")

    if os.path.exists(txn_path) and os.path.exists(id_path):
        logger.info("Loading transaction data from %s", txn_path)
        transactions = pd.read_csv(txn_path)
        logger.info("Loaded %d transaction rows", len(transactions))

        logger.info("Loading identity data from %s", id_path)
        identity = pd.read_csv(id_path)
        logger.info("Loaded %d identity rows", len(identity))

        logger.info("Left-joining on TransactionID")
        df = transactions.merge(identity, on="TransactionID", how="left")
    else:
        logger.warning(
            "Full Kaggle dataset not found in %s — falling back to sample at %s",
            path,
            SAMPLE_DATASET_PATH,
        )
        if not os.path.exists(SAMPLE_DATASET_PATH):
            raise FileNotFoundError(
                f"Full dataset not found in '{path}' and sample file not found at "
                f"'{SAMPLE_DATASET_PATH}'. Download the IEEE-CIS dataset from Kaggle "
                "or ensure the sample file is present."
            )
        logger.info("Loading sample dataset from %s", SAMPLE_DATASET_PATH)
        df = pd.read_csv(SAMPLE_DATASET_PATH)
        logger.info("Loaded %d rows from sample", len(df))

    df = df.rename(columns=IEEE_COLUMN_MAP)

    # TransactionDT is an integer seconds-offset; convert to real datetimes
    df["timestamp"] = _IEEE_REFERENCE_DATE + pd.to_timedelta(df["timestamp"], unit="s")

    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    logger.info("Dataset ready: %d rows, %d fraud (%.2f%%)",
                len(df), df["label"].sum(), df["label"].mean() * 100)
    return df


def load_user_dataset(uploaded_file, column_mapping: Optional[dict] = None) -> pd.DataFrame:
    """Load and validate a user-uploaded CSV file.

    Args:
        uploaded_file: Streamlit UploadedFile object or any file-like object.
        column_mapping: Optional dict mapping user column names to standard names.
            Call get_column_mapping_suggestions first to generate candidates,
            then pass the user-confirmed mapping here.

    Returns:
        Validated DataFrame with standardized column names.

    Raises:
        ValueError: If required columns are missing or data types are invalid.
    """
    df = pd.read_csv(uploaded_file)
    return validate_schema(df, column_mapping)


def validate_schema(df: pd.DataFrame, column_mapping: Optional[dict] = None) -> pd.DataFrame:
    """Validate that the DataFrame has required columns and correct types.

    Args:
        df: Raw DataFrame to validate.
        column_mapping: Optional dict mapping user column names to expected names.
            Keys are current column names; values are standard pipeline names.

    Returns:
        Validated DataFrame with standardized column names.

    Raises:
        ValueError: If required columns are missing after mapping, amount is
            non-numeric, label is non-binary, timestamp is unparseable, or any
            required column contains null values.
    """
    df = df.copy()

    if column_mapping:
        df = df.rename(columns=column_mapping)

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Available columns: {list(df.columns[:20])}"
        )

    if not pd.api.types.is_numeric_dtype(df["amount"]):
        raise ValueError("Column 'amount' must be numeric.")

    unique_labels = set(df["label"].dropna().unique())
    if not unique_labels <= {0, 1}:
        raise ValueError(
            f"Column 'label' must contain only 0 and 1. Found: {unique_labels}"
        )

    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        try:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"Column 'timestamp' could not be parsed as datetime: {exc}"
            ) from exc

    null_cols = [col for col in REQUIRED_COLUMNS if df[col].isnull().any()]
    if null_cols:
        raise ValueError(f"Required columns contain null values: {null_cols}")

    return df


def get_column_mapping_suggestions(df: pd.DataFrame) -> dict:
    """Suggest column mappings based on column names and data types.

    Uses fuzzy string matching via difflib to suggest which user columns
    map to the required schema. Only suggestions with a similarity score
    above 0.6 are returned to avoid spurious matches.

    Args:
        df: User-uploaded DataFrame with unknown column names.

    Returns:
        Dict of {required_col: suggested_user_col} mappings. Required columns
        with no close match are omitted from the result.
    """
    suggestions: dict = {}
    for required_col, candidates in _COLUMN_CANDIDATES.items():
        best_match = None
        best_score = 0.0
        for user_col in df.columns:
            for candidate in candidates:
                score = difflib.SequenceMatcher(
                    None, user_col.lower(), candidate.lower()
                ).ratio()
                if score > best_score:
                    best_score = score
                    best_match = user_col
        if best_score >= 0.6:
            suggestions[required_col] = best_match
    return suggestions

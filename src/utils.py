"""
Shared Utilities.

Common helper functions used across modules:
logging setup, config management, and formatting.
"""

import logging


# Centralized config — override via dashboard sidebar
DEFAULT_CONFIG = {
    "amount_percentile": 0.95,
    "velocity_max_txns": 5,
    "velocity_window_hours": 1,
    "night_start_hour": 22,
    "night_end_hour": 6,
    "repeat_window_minutes": 30,
    "model_test_size": 0.2,
    "model_use_smote": True,
    "shap_background_sample": 500,
}


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure and return a logger for the application.

    Args:
        level: Logging level string (DEBUG, INFO, WARNING, ERROR).

    Returns:
        Configured logger instance.
    """
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger("fraud_dashboard")


def format_currency(value: float) -> str:
    """Format a number as currency string.

    Args:
        value: Numeric amount.

    Returns:
        Formatted string like '$1,234.56'.
    """
    return f"${value:,.2f}"


def format_percentage(value: float) -> str:
    """Format a decimal as percentage string.

    Args:
        value: Decimal value (e.g., 0.035).

    Returns:
        Formatted string like '3.50%'.
    """
    return f"{value * 100:.2f}%"

"""Intelligent Customer Churn Prediction System (Classical ML & A/B)."""
from importlib import metadata
try:
    __version__ = metadata.version("customer-churn-prediction-system")
except Exception:  # pragma: no cover
    __version__ = "1.0.0"
__all__ = ["__version__"]

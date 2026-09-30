"""mercy-ml - pooled financial models for the personal finance manager.

Top-level surface is intentionally small so callers can do::

    from mercy_ml import SpendMatrices, build_forecast_frame, load_transactions
    from mercy_ml import ExpenseForecaster, OverspendClassifier

The `models` subpackage is imported lazily to keep `import mercy_ml` cheap for
processes that only need the data/feature layer (for example the FastAPI app,
which scores from persisted artifacts it does not itself construct).
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

__version__ = "0.1.0"

from mercy_ml.config import MLSettings, get_settings, model_version
from mercy_ml.data import (
    load_budgets,
    load_categories,
    load_training_bundle,
    load_transactions,
    load_users,
    panel_periods,
    transactions_for_user,
)
from mercy_ml.features import (
    SpendMatrices,
    build_budget_recommendation_frame,
    build_cluster_frame,
    build_forecast_frame,
    build_forecast_inference_row,
    build_overspend_frame,
    build_overspend_inference_row,
    build_savings_frame,
    build_savings_inference_row,
)
from mercy_ml.periods import (
    elapsed_fraction,
    key_for,
    keys_between,
    month_key,
    next_period_key,
    parse_period_key,
    period_end,
    period_length,
    period_start,
    previous_period_key,
    week_key,
)
from mercy_ml.preprocess import OutlierHandler, build_preprocessor
from mercy_ml.schema import FEATURE_SCHEMA, FeatureContract, contract_digest, feature_frame

if TYPE_CHECKING:  # pragma: no cover - typing only
    from mercy_ml.models import (
        BaseMLModel,
        BudgetRecommender,
        ExpenseForecaster,
        OverspendClassifier,
        SavingsRecommender,
        SpendingClusterer,
        TrainingReport,
    )

_LAZY_MODELS = {
    "BaseMLModel",
    "BudgetRecommender",
    "EtsForecaster",
    "ExpenseForecaster",
    "OverspendClassifier",
    "SavingsRecommender",
    "SpendingClusterer",
    "TrainingReport",
    "compare_estimators",
    "compare_forecasters",
}

__all__ = [
    "FEATURE_SCHEMA",
    "MLSettings",
    "OutlierHandler",
    "SpendMatrices",
    "FeatureContract",
    "__version__",
    "build_budget_recommendation_frame",
    "build_cluster_frame",
    "build_forecast_frame",
    "build_forecast_inference_row",
    "build_overspend_frame",
    "build_overspend_inference_row",
    "build_preprocessor",
    "build_savings_frame",
    "build_savings_inference_row",
    "contract_digest",
    "elapsed_fraction",
    "feature_frame",
    "get_settings",
    "key_for",
    "keys_between",
    "load_budgets",
    "load_categories",
    "load_training_bundle",
    "load_transactions",
    "load_users",
    "model_version",
    "month_key",
    "next_period_key",
    "panel_periods",
    "parse_period_key",
    "period_end",
    "period_length",
    "period_start",
    "previous_period_key",
    "transactions_for_user",
    "week_key",
    *_LAZY_MODELS,
]


def __getattr__(name: str) -> Any:
    """Expose the model classes without importing scikit-learn at import time."""
    if name in _LAZY_MODELS:
        module = importlib.import_module("mercy_ml.models")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | _LAZY_MODELS)

"""Pydantic models for the `predictions` collection.

`value` is the single headline number for a prediction (forecast total,
probability, suggested budget, ...); `extra` carries the type-specific payload
that the frontend renders.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field

from app.core.constants import PredictionType
from app.models.base import MongoModel


class PredictionScope(BaseModel):
    model_config = ConfigDict(extra="allow")

    category_id: str | None = None
    category_name: str | None = None
    period_key: str | None = None
    period: str | None = Field(default=None, description="weekly | monthly | yearly")
    cluster_id: int | None = None
    budget_id: str | None = None
    goal_id: str | None = None


class PredictionWrite(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: PredictionType
    value: float
    confidence_score: float = Field(ge=0.0, le=1.0)
    model_version: str
    model_run_id: str
    scope: PredictionScope = Field(default_factory=PredictionScope)
    label: str | None = None
    features: dict[str, Any] | None = None
    extra: dict[str, Any] | None = None
    horizon_days: int | None = None
    expires_at: datetime | None = None
    generated_at: datetime | None = None


class PredictionOut(MongoModel):
    type: PredictionType
    value: float
    confidence_score: float
    model_version: str
    model_run_id: str
    scope: PredictionScope = Field(default_factory=PredictionScope)
    label: str | None = None
    features: dict[str, Any] | None = None
    extra: dict[str, Any] | None = None
    horizon_days: int | None = None
    generated_at: datetime | None = None
    expires_at: datetime | None = None


class PredictionSummary(BaseModel):
    type: PredictionType
    count: int
    latest_value: float | None = None
    average_confidence: float | None = None
    latest_generated_at: datetime | None = None


class OverspendRiskRequest(BaseModel):
    """On-demand inference payload for a specific budget in its current period."""

    model_config = ConfigDict(extra="forbid")

    category_id: str | None = None
    period: str | None = Field(default=None, description="Defaults to the budget's own period")
    limit_amount: float | None = Field(
        default=None, description="Override the stored budget limit (e.g. what-if analysis)"
    )
    as_of: datetime | None = None


class OverspendRiskItem(BaseModel):
    category_id: str | None = None
    category_name: str | None = None
    budget_id: str | None = None
    probability: float
    label: str = Field(description="low | medium | high")
    confidence_score: float
    model_version: str
    features: dict[str, Any] = Field(default_factory=dict)
    computed_at: datetime


class OverspendRiskResponse(BaseModel):
    run_id: str
    as_of: datetime
    model_version: str
    global_probability: float | None = None
    items: Annotated[list[OverspendRiskItem], Field(default_factory=list)]


class ExpenseForecastItem(BaseModel):
    category_id: str | None = None
    category_name: str | None = None
    period_key: str
    predicted_amount: float
    lower_bound: float | None = None
    upper_bound: float | None = None
    confidence_score: float
    trend: str | None = Field(default=None, description="up | down | flat")


class ExpenseForecastResponse(BaseModel):
    run_id: str
    model_version: str
    target_period_key: str
    period: str = "monthly"
    total_predicted: float = 0.0
    total_lower_bound: float | None = None
    total_upper_bound: float | None = None
    items: Annotated[list[ExpenseForecastItem], Field(default_factory=list)]


class BudgetRecommendationItem(BaseModel):
    category_id: str | None = None
    category_name: str | None = None
    current_limit: float | None = None
    recommended_limit: float
    basis: dict[str, Any] = Field(default_factory=dict)
    confidence_score: float


class BudgetRecommendationResponse(BaseModel):
    run_id: str
    model_version: str
    period: str = "monthly"
    items: Annotated[list[BudgetRecommendationItem], Field(default_factory=list)]


class SavingsRecommendationResponse(BaseModel):
    run_id: str
    model_version: str
    period: str = "monthly"
    average_income: float
    predicted_expenses: float
    discretionary_spend: float
    recommended_savings: float
    savings_rate_pct: float
    confidence_score: float
    basis: dict[str, Any] = Field(default_factory=dict)


class SpendingCluster(BaseModel):
    cluster_id: int
    label: str
    size: int
    share_pct: float
    mean_amount: float
    dominant_category: str | None = None
    dominant_day_of_week: int | None = None
    mean_day_of_month: float | None = None
    monthly_total: float = 0.0
    description: str | None = None


class SpendingClusterResponse(BaseModel):
    run_id: str
    model_version: str
    silhouette_score: float | None = None
    n_clusters: int
    clusters: Annotated[list[SpendingCluster], Field(default_factory=list)]
    per_user: dict[str, list[dict[str, Any]]] | None = Field(
        default=None, description="Optional per-user cluster assignment counts"
    )


class ModelInfo(BaseModel):
    model_type: PredictionType
    model_version: str
    run_id: str
    trained_at: datetime | None = None
    is_active: bool = True
    metrics: dict[str, Any] = Field(default_factory=dict)
    n_train: int | None = None
    n_test: int | None = None
    features: list[str] = Field(default_factory=list)


class ModelRunInfo(BaseModel):
    """Summary of the most recent training/inference run recorded in `model_runs`."""

    run_id: str
    run_type: str | None = None
    model_type: str | None = None
    status: str | None = None
    is_active: bool = True
    n_users: int | None = None
    n_predictions: int | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    duration_seconds: float | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    message: str | None = None


class ModelRegistryResponse(BaseModel):
    available: bool
    artifact_dir: str
    models: Annotated[list[ModelInfo], Field(default_factory=list)]
    latest_run: ModelRunInfo | None = None
    message: str | None = None

"""Aggregated dashboard payloads.

The dashboard is a single round-trip read model so the React app can render
without fanning out to a dozen endpoints. Everything is assembled from the
existing collections - there is no separate cache to keep warm.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.core.constants import AlertSeverity, BudgetPeriod, PredictionType
from app.models.alert import AlertOut
from app.models.budget import BudgetOut
from app.models.prediction import PredictionOut
from app.models.report import TimeBucket
from app.models.transaction import TransactionOut


class MoneyCard(BaseModel):
    label: str
    value: float
    previous_value: float | None = None
    change_pct: float | None = None
    trend: str | None = None
    currency: str = "KES"


class NetWorthPoint(BaseModel):
    period_key: str
    cumulative_income: float = 0.0
    cumulative_expense: float = 0.0
    balance: float = 0.0


class CategorySpend(BaseModel):
    category_id: str | None = None
    category_name: str
    category_color: str | None = None
    total: float
    share_pct: float
    change_pct: float | None = None


class PredictionWidget(BaseModel):
    type: PredictionType
    headline: str
    value: float
    confidence_score: float
    detail: str | None = None
    category_name: str | None = None
    generated_at: datetime | None = None
    prediction_id: str | None = None
    action: dict[str, Any] | None = Field(
        default=None, description="Hint for the UI, e.g. {'kind': 'create_budget', 'category_id': ...}"
    )


class DashboardResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "generated_at": "2026-09-26T12:00:00Z",
                "period": "monthly",
                "period_key": "2026-09",
                "cards": {
                    "balance": {"label": "Net this month", "value": 1840.22, "change_pct": 4.1, "trend": "up"}
                },
            }
        }
    )

    generated_at: datetime
    currency: str = "KES"
    period: str = "monthly"
    period_key: str

    cards: dict[str, MoneyCard] = Field(default_factory=dict)
    cash_flow: list[TimeBucket] = Field(default_factory=list)
    net_worth: list[NetWorthPoint] = Field(default_factory=list)
    top_categories: list[CategorySpend] = Field(default_factory=list)
    budgets: list[Any] = Field(default_factory=list)
    recent_transactions: list[Any] = Field(default_factory=list)
    predictions: list[Any] = Field(default_factory=list)
    alerts: list[Any] = Field(default_factory=list)
    goals_summary: dict[str, Any] = Field(default_factory=dict)
    insights: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str
    database: dict[str, Any] = Field(default_factory=dict)
    ml: dict[str, Any] = Field(default_factory=dict)
    scheduler: dict[str, Any] = Field(default_factory=dict)


class BudgetHealth(BaseModel):
    budget: BudgetOut
    period: BudgetPeriod
    risk_level: str = "unknown"
    risk_probability: float | None = None


class AlertRiskSummary(BaseModel):
    critical: int = 0
    warning: int = 0
    info: int = 0
    highest_severity: AlertSeverity | None = None

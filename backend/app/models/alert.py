"""Pydantic models for the `alerts` collection."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field

from app.core.constants import AlertSeverity, AlertType
from app.models.base import MongoModel


class AlertContext(BaseModel):
    """Structured payload so the frontend can deep-link an alert."""

    model_config = ConfigDict(extra="allow")

    category_id: str | None = None
    category_name: str | None = None
    budget_id: str | None = None
    goal_id: str | None = None
    transaction_id: str | None = None
    prediction_id: str | None = None
    period_key: str | None = None
    amount: float | None = None
    threshold_pct: float | None = None


class AlertCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: AlertType
    message: str = Field(min_length=1, max_length=1000)
    title: str | None = Field(default=None, max_length=200)
    severity: AlertSeverity = AlertSeverity.INFO
    context: AlertContext | None = None
    dedupe_key: str | None = Field(
        default=None,
        max_length=300,
        description="Stable key; a sparse unique index on (user_id, dedupe_key) prevents duplicates",
    )
    triggered_at: datetime | None = None
    expires_at: datetime | None = None


class AlertUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    is_read: bool | None = None
    severity: AlertSeverity | None = None


class AlertOut(MongoModel):
    type: AlertType
    title: str | None = None
    message: str
    severity: AlertSeverity = AlertSeverity.INFO
    context: AlertContext | None = None
    is_read: bool = False
    read_at: datetime | None = None
    triggered_at: datetime
    expires_at: datetime | None = None
    dedupe_key: str | None = None


class AlertCounts(BaseModel):
    total: int = 0
    unread: int = 0
    critical: int = 0
    warning: int = 0
    info: int = 0
    by_type: dict[str, int] = Field(default_factory=dict)


class AlertReadRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    alert_ids: Annotated[list[str], Field(min_length=1, max_length=500)] | None = None
    all: bool = Field(default=False, description="Mark every unread alert as read")
    type: AlertType | None = None
    severity: AlertSeverity | None = None


class AlertReadResult(BaseModel):
    matched: int
    modified: int
    unread_remaining: int


class AnyPayload(BaseModel):
    """Small helper for ad-hoc service responses."""

    model_config = ConfigDict(extra="allow")

    data: dict[str, Any] = Field(default_factory=dict)

"""Aggregate all routers under the `/api/v1` prefix."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.routers import (
    alerts,
    auth,
    budgets,
    categories,
    dashboard,
    goals,
    health,
    predictions,
    reports,
    transactions,
    users,
)
from app.core.config import settings

api_router = APIRouter(prefix=settings.api_v1_prefix)

api_router.include_router(auth.router)
api_router.include_router(users.router)
api_router.include_router(categories.router)
api_router.include_router(transactions.router)
api_router.include_router(budgets.router)
api_router.include_router(goals.router)
api_router.include_router(alerts.router)
api_router.include_router(predictions.router)
api_router.include_router(reports.router)
api_router.include_router(dashboard.router)
api_router.include_router(health.router)
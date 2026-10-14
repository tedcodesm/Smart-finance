"""Dashboard read model."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import Services, get_current_user, get_services
from app.models.dashboard import DashboardResponse

import logging

logger = logging.getLogger("mercy.api.dashboard")

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


@router.get("", response_model=DashboardResponse)
@router.get("/", response_model=DashboardResponse, include_in_schema=False)
def dashboard(
    period: str = "monthly",
    user: dict = Depends(get_current_user),
    services: Services = Depends(get_services),
) -> DashboardResponse:
    try:
        return services.dashboard_service().build(user["_id"], period=period)
    except Exception as exc:
        logger.exception("Failed to build dashboard for user %s: %s", user.get("_id"), exc)
        return services.dashboard_service().build_fallback(user, period=period)
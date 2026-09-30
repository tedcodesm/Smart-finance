"""Dashboard read model."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import Services, get_current_user, get_services
from app.models.dashboard import DashboardResponse

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


@router.get("", response_model=DashboardResponse)
def dashboard(
    period: str = "monthly",
    user: dict = Depends(get_current_user),
    services: Services = Depends(get_services),
) -> DashboardResponse:
    return services.dashboard_service().build(user["_id"], period=period)
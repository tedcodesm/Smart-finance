"""Health checks (public) and scheduler status (authenticated)."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import Services, get_current_user, get_services
from app.core.config import settings
from app.models.dashboard import HealthResponse

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
def health(services: Services = Depends(get_services)) -> HealthResponse:
    db = _db_status(services.db)
    ml = _ml_status(services)
    scheduler = {
        "enabled": settings.enable_scheduler,
        "ml_batch": f"{settings.ml_batch_hour:02d}:{settings.ml_batch_minute:02d} UTC",
        "alert_sweep": f"{settings.alert_sweep_hour:02d}:{settings.alert_sweep_minute:02d} UTC",
    }
    return HealthResponse(
        status="ok" if db.get("ok") else "degraded",
        version=settings.version,
        environment=settings.environment,
        database=db,
        ml=ml,
        scheduler=scheduler,
    )


@router.get("/admin/models", include_in_schema=False)
def admin_models(user: dict = Depends(get_current_user), services: Services = Depends(get_services)) -> dict:
    response = services.ml_service().models_status()
    return response.model_dump(mode="json")


def _db_status(db) -> dict:
    try:
        from app.core.time_utils import utc_now

        collections = sorted(db.list_collection_names())
        return {"ok": True, "name": db.name, "collections": collections, "checked_at": utc_now().isoformat()}
    except Exception as exc:  # pragma: no cover - connection failure
        return {"ok": False, "name": getattr(db, "name", "unknown"), "error": str(exc)}


def _ml_status(services: Services) -> dict:
    registry = services.ml_service().models_status()
    return {
        "available": registry.available,
        "models": [m.model_type.value for m in registry.models],
        "message": registry.message,
    }
"""Alerts: list, counts, mark-read, delete."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from app.api.deps import Services, get_current_user, get_services
from app.core.constants import AlertSeverity, AlertType
from app.core.pagination import Page, PageParams
from app.models.alert import AlertCounts, AlertOut, AlertReadRequest, AlertReadResult
from app.models.base import to_object_id

router = APIRouter(prefix="/alerts", tags=["alerts"])


@router.get("", response_model=Page[AlertOut])
def list_alerts(
    params: PageParams = Depends(),
    is_read: bool | None = None,
    types: str | None = Query(default=None, description="Comma-separated AlertType values"),
    severities: str | None = Query(default=None, description="Comma-separated AlertSeverity values"),
    user: dict = Depends(get_current_user),
    services: Services = Depends(get_services),
) -> Page[AlertOut]:
    type_values = _split_enum(types, AlertType)
    severity_values = _split_enum(severities, AlertSeverity)
    rows, total = services.alert_service().list_alerts(
        user["_id"],
        is_read=is_read,
        types=type_values,
        severities=severity_values,
        skip=params.skip,
        limit=params.limit,
    )
    return Page.build(items=services.alert_service().serialize_many(rows), total=total, params=params)


@router.get("/counts", response_model=AlertCounts)
def alert_counts(
    since: str | None = None,
    user: dict = Depends(get_current_user),
    services: Services = Depends(get_services),
) -> AlertCounts:
    from datetime import datetime

    since_dt = datetime.fromisoformat(since) if since else None
    return services.alert_service().counts(user["_id"], since=since_dt)


@router.patch("/read", response_model=AlertReadResult)
def mark_read(
    payload: AlertReadRequest,
    user: dict = Depends(get_current_user),
    services: Services = Depends(get_services),
) -> AlertReadResult:
    if payload.alert_ids:
        ids = [to_object_id(a, field="alert_id") for a in payload.alert_ids]
    else:
        ids = None
    matched = services.alert_service().mark_read(
        user["_id"],
        alert_ids=ids,
        all_alerts=payload.all or (payload.alert_ids is None and payload.type is None and payload.severity is None),
        types=[payload.type] if payload.type else None,
        severities=[payload.severity] if payload.severity else None,
    )
    _, unread = services.alert_service().list_alerts(user["_id"], is_read=False, limit=1)
    return AlertReadResult(matched=matched, modified=matched, unread_remaining=unread)


@router.delete("/{alert_id}", status_code=204)
def delete_alert(
    alert_id: str,
    user: dict = Depends(get_current_user),
    services: Services = Depends(get_services),
) -> None:
    services.alert_service().delete(user["_id"], [to_object_id(alert_id, field="alert_id")])
    return None


def _split_enum(raw: str | None, enum_cls) -> list[str] | None:
    if not raw:
        return None
    values: list[str] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            enum_cls(token)
        except ValueError as exc:  # type: ignore[attr-defined]
            from app.core.errors import ValidationError

            raise ValidationError(f"Invalid {enum_cls.__name__}: {token!r}") from exc
        values.append(token)
    return values or None
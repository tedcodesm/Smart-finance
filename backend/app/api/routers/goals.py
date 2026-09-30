"""Savings goal CRUD + contributions."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query, status as http_status

from app.api.deps import Services, get_current_user, get_services, require_doc
from app.core.constants import GoalStatus
from app.core.pagination import Page, PageParams
from app.models.base import serialize_doc, to_object_id
from app.models.goal import (
    GoalContribution,
    GoalCreate,
    GoalOut,
    GoalProgressEvent,
    GoalSummaryOut,
    GoalUpdate,
)

router = APIRouter(prefix="/goals", tags=["goals"])


@router.get("", response_model=Page[GoalOut])
def list_goals(
    params: PageParams = Depends(),
    status: GoalStatus | None = Query(default=None, alias="status"),
    user: dict = Depends(get_current_user),
    services: Services = Depends(get_services),
) -> Page[GoalOut]:
    rows = services.goal_service().list_goals(user["_id"], status=status)
    page_rows = [serialize_doc(row) for row in rows[params.skip : params.skip + params.limit]]
    return Page.build(items=[r for r in page_rows if r is not None], total=len(rows), params=params)


@router.get("/summary", response_model=GoalSummaryOut)
def goal_summary(
    user: dict = Depends(get_current_user),
    services: Services = Depends(get_services),
) -> GoalSummaryOut:
    return GoalSummaryOut(**services.goal_service().summary(user["_id"]))


@router.post("", response_model=GoalOut, status_code=http_status.HTTP_201_CREATED)
def create_goal(
    payload: GoalCreate,
    user: dict = Depends(get_current_user),
    services: Services = Depends(get_services),
) -> dict:
    doc = services.goal_service().create_goal(user["_id"], payload.model_dump())
    return services.goal_service().serialize(require_doc(doc, "Savings goal could not be created"))


@router.get("/{goal_id}", response_model=GoalOut)
def get_goal(
    goal_id: str,
    user: dict = Depends(get_current_user),
    services: Services = Depends(get_services),
) -> dict:
    doc = services.goal_service().get_goal(user["_id"], to_object_id(goal_id, field="goal_id"))
    return services.goal_service().serialize(require_doc(doc, "Savings goal not found"))


@router.patch("/{goal_id}", response_model=GoalOut)
def update_goal(
    goal_id: str,
    changes: GoalUpdate,
    user: dict = Depends(get_current_user),
    services: Services = Depends(get_services),
) -> dict:
    doc = services.goal_service().update_goal(
        user["_id"], to_object_id(goal_id, field="goal_id"), changes.model_dump(exclude_none=True)
    )
    return services.goal_service().serialize(require_doc(doc, "Savings goal not found"))


@router.delete("/{goal_id}", status_code=http_status.HTTP_204_NO_CONTENT)
def delete_goal(
    goal_id: str,
    user: dict = Depends(get_current_user),
    services: Services = Depends(get_services),
) -> None:
    services.goal_service().delete_goal(user["_id"], to_object_id(goal_id, field="goal_id"))
    return None


@router.post("/{goal_id}/contribute", response_model=GoalProgressEvent)
def contribute(
    goal_id: str,
    payload: GoalContribution,
    user: dict = Depends(get_current_user),
    services: Services = Depends(get_services),
) -> GoalProgressEvent:
    receipt = services.goal_service().contribute(
        user["_id"], to_object_id(goal_id, field="goal_id"), payload.amount, note=payload.note
    )
    return GoalProgressEvent(**receipt)

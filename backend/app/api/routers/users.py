"""Authenticated user profile endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import Services, get_current_user, get_services, require_doc
from app.models.base import serialize_doc
from app.models.user import PasswordChange, UserOut, UserUpdate

router = APIRouter(prefix="/users/me", tags=["users"])


@router.get("", response_model=UserOut)
def me(user: dict = Depends(get_current_user)) -> dict:
    return serialize_doc(require_doc(user, "Account not found")) or {}


@router.patch("", response_model=UserOut)
def update_me(
    changes: UserUpdate,
    user: dict = Depends(get_current_user),
    services: Services = Depends(get_services),
) -> dict:
    updated = services.auth_service().update_user(user["_id"], changes.model_dump(exclude_none=True))
    return serialize_doc(require_doc(updated, "Account not found")) or {}


@router.post("/password", response_model=UserOut)
def change_password(
    payload: PasswordChange,
    user: dict = Depends(get_current_user),
    services: Services = Depends(get_services),
) -> dict:
    services.auth_service().change_password(user["_id"], payload.current_password, payload.new_password)
    # Return the refreshed document so `updated_at` reflects the password change.
    refreshed = services.users().find_by_id(user["_id"])
    return serialize_doc(require_doc(refreshed, "Account not found")) or {}
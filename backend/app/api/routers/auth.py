"""Auth router: register, login, refresh, logout.

All endpoints are anonymous. Long-lived refresh tokens are stored server-side
(revocable) and rotated on every use.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict

from app.api.deps import Services, get_services
from app.models.user import AuthResponse, LoginRequest, RefreshRequest, UserCreate
from app.models.base import serialize_doc

router = APIRouter(prefix="/auth", tags=["auth"])


class LogoutResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    revoked: int


@router.post("/register", response_model=AuthResponse, status_code=201)
def register(payload: UserCreate, services: Services = Depends(get_services)) -> AuthResponse:
    auth = services.auth_service()
    user = auth.register(payload.model_dump())
    tokens = auth.issue_tokens_for(user["_id"])
    return AuthResponse(user=serialize_doc(user), tokens=tokens)  # type: ignore[arg-type]


@router.post("/login", response_model=AuthResponse)
def login(payload: LoginRequest, services: Services = Depends(get_services)) -> AuthResponse:
    result = services.auth_service().login(payload.email, payload.password)
    return AuthResponse(user=serialize_doc(result["user"]), tokens=result["tokens"])  # type: ignore[arg-type]


@router.post("/refresh", response_model=AuthResponse)
def refresh(payload: RefreshRequest, services: Services = Depends(get_services)) -> AuthResponse:
    result = services.auth_service().refresh(payload.refresh_token)
    return AuthResponse(user=serialize_doc(result["user"]), tokens=result["tokens"])  # type: ignore[arg-type]


@router.post("/logout", response_model=LogoutResponse)
def logout(payload: RefreshRequest, services: Services = Depends(get_services)) -> LogoutResponse:
    return LogoutResponse(revoked=services.auth_service().logout(payload.refresh_token))

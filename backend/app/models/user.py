"""Pydantic models for the `users` collection and auth payloads."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Annotated

from pydantic import AfterValidator, BaseModel, ConfigDict, EmailStr, Field, TypeAdapter, ValidationError, field_validator

from app.core.config import settings
from app.core.constants import StrEnum
from app.models.base import MongoModel

logger = logging.getLogger("mercy.models.user")


def _require_letter_and_digit(value: str) -> str:
    if not any(c.isalpha() for c in value) or not any(c.isdigit() for c in value):
        raise ValueError("Password must contain at least one letter and one digit")
    return value


#: Single source of truth for the password policy, shared by registration and
#: password changes. Lives here so `settings.demo_password` can be validated
#: against the same rule the API enforces.
StrongPassword = Annotated[
    str,
    Field(min_length=settings.password_min_length, max_length=256),
    AfterValidator(_require_letter_and_digit),
]

# Warn (do not raise) when the configured demo password could never be submitted
# to `POST /auth/register`. `demo_seed` creates the account through the service
# layer, which skips this policy, so a violation breaks self-service registration
# of the demo account without breaking the demo login itself.
# `StrongPassword(...)` cannot be used for this: an `Annotated` alias called as a
# constructor just invokes the underlying `str`, so the validators never run.
_DEMO_PASSWORD_OK = True
try:
    TypeAdapter(StrongPassword).validate_python(settings.demo_password)
except ValidationError as _exc:
    _DEMO_PASSWORD_OK = False
    logger.warning(
        "DEMO_PASSWORD does not satisfy the registration password policy, so the demo "
        "account cannot be recreated through POST /auth/register: %s",
        _exc.errors()[0].get("msg"),
    )


class UserCreate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    name: Annotated[str, Field(min_length=1, max_length=120)]
    email: EmailStr
    password: StrongPassword
    base_currency: str = Field(default="USD", min_length=3, max_length=3)

    @field_validator("base_currency")
    @classmethod
    def _upper(cls, value: str) -> str:
        return value.upper()


class UserUpdate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    name: Annotated[str | None, Field(default=None, min_length=1, max_length=120)] = None
    base_currency: Annotated[str | None, Field(default=None, min_length=3, max_length=3)] = None
    monthly_income_estimate: float | None = Field(default=None, ge=0)
    is_active: bool | None = None

    @field_validator("base_currency")
    @classmethod
    def _upper(cls, value: str | None) -> str | None:
        return value.upper() if value else value


class UserOut(MongoModel):
    name: str
    email: EmailStr
    base_currency: str = "USD"
    is_active: bool = True
    monthly_income_estimate: float | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    last_login_at: datetime | None = None


class PasswordChange(BaseModel):
    model_config = ConfigDict(extra="forbid")

    current_password: str
    new_password: StrongPassword


class TokenType(StrEnum):
    ACCESS = "access"
    REFRESH = "refresh"


class TokenPair(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = Field(description="Access token lifetime in seconds")


class LoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    email: EmailStr
    password: str


class RefreshRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    refresh_token: str


class AuthResponse(BaseModel):
    user: UserOut
    tokens: TokenPair

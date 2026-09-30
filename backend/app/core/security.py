"""Password hashing and JWT issuance/verification."""

from __future__ import annotations

import hashlib
import uuid
from datetime import timedelta
from typing import Any, Literal

import bcrypt
import jwt

from app.core.config import settings
from app.core.errors import AuthenticationError
from app.core.time_utils import strip_tz, utc_now

TokenType = Literal["access", "refresh"]

#: bcrypt silently truncates (older builds) or rejects (>=4.1) inputs longer
#: than 72 bytes. Pre-hashing to a fixed 64-char hex digest makes the effective
#: limit irrelevant while keeping the stored hash a plain bcrypt digest.
_BCRYPT_MAX_BYTES = 72


def _prehash(password: str) -> bytes:
    return hashlib.sha256(password.encode("utf-8")).hexdigest().encode("ascii")


def hash_password(password: str) -> str:
    """Hash a plaintext password with bcrypt (cost from settings)."""
    salt = bcrypt.gensalt(rounds=settings.bcrypt_rounds)
    return bcrypt.hashpw(_prehash(password), salt).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(_prehash(password), password_hash.encode("utf-8"))
    except (ValueError, TypeError):
        return False


def _encode(subject: str, token_type: TokenType, expires_delta: timedelta, claims: dict[str, Any]) -> str:
    now = utc_now()
    payload: dict[str, Any] = {
        "sub": subject,
        "type": token_type,
        "jti": uuid.uuid4().hex,
        "iat": int(now.timestamp()),
        "nbf": int(now.timestamp()),
        "exp": int((now + expires_delta).timestamp()),
        "iss": settings.app_name,
        **claims,
    }
    return jwt.encode(payload, settings.secret_key, algorithm=settings.jwt_algorithm)


def create_access_token(user_id: str, *, extra_claims: dict[str, Any] | None = None) -> tuple[str, str]:
    """Return `(token, jti)`. The jti is stored so tokens can be revoked."""
    token = _encode(
        user_id,
        "access",
        timedelta(minutes=settings.access_token_expire_minutes),
        extra_claims or {},
    )
    jti = jwt.decode(token, settings.secret_key, algorithms=[settings.jwt_algorithm])["jti"]
    return token, jti


def create_refresh_token(user_id: str) -> tuple[str, str, str]:
    """Return `(token, jti, expires_at)` for the refresh-token store."""
    expires_at = utc_now() + timedelta(days=settings.refresh_token_expire_days)
    token = _encode(user_id, "refresh", timedelta(days=settings.refresh_token_expire_days), {})
    jti = jwt.decode(token, settings.secret_key, algorithms=[settings.jwt_algorithm])["jti"]
    return token, jti, strip_tz(expires_at)


def decode_token(token: str, *, expected_type: TokenType | None = None) -> dict[str, Any]:
    try:
        payload = jwt.decode(
            token,
            settings.secret_key,
            algorithms=[settings.jwt_algorithm],
            options={"require": ["exp", "sub", "type"]},
        )
    except jwt.ExpiredSignatureError as exc:
        raise AuthenticationError("Token has expired") from exc
    except jwt.InvalidTokenError as exc:
        raise AuthenticationError("Invalid authentication token") from exc

    if expected_type and payload.get("type") != expected_type:
        raise AuthenticationError(f"Expected a {expected_type} token")
    return payload

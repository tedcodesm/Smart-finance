"""Registration, login, refresh and token revocation."""

from __future__ import annotations

from typing import Any

from app.core.config import settings
from app.core.errors import AuthenticationError, ConflictError, NotFoundError
from app.core.logging import get_logger
from app.core.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    verify_password,
)
from app.core.time_utils import utc_now
from app.repositories.user_repo import RefreshTokenRepository, UserRepository

logger = get_logger(__name__)

#: Uniform error for both unknown-email and wrong-password so the API does not
#: disclose which emails are registered.
_INVALID_CREDENTIALS = "Incorrect email or password"


class AuthService:
    def __init__(self, user_repo: UserRepository, refresh_repo: RefreshTokenRepository) -> None:
        self.users = user_repo
        self.refresh_tokens = refresh_repo

    # ------------------------------------------------------------------ auth
    def register(self, payload: dict[str, Any]) -> dict[str, Any]:
        email = payload["email"].strip().lower()
        if self.users.find_by_email(email):
            raise ConflictError("An account with that email already exists")

        user = self.users.create_user(
            name=payload["name"],
            email=email,
            password_hash=hash_password(payload["password"]),
            base_currency=payload.get("base_currency") or settings.default_currency,
        )
        logger.info("Registered user %s", user["_id"])

        # Every new user starts with the shared system category catalogue.
        from app.repositories.category_repo import CategoryRepository

        CategoryRepository(self.users.db).seed_system_categories()
        return user

    def login(self, email: str, password: str) -> dict[str, Any]:
        user = self.users.find_by_email(email.strip().lower())
        if user is None or not verify_password(password, user.get("password_hash", "")):
            raise AuthenticationError(_INVALID_CREDENTIALS)
        if not user.get("is_active", True):
            raise AuthenticationError("This account has been deactivated")

        self.users.touch_login(user["_id"])
        tokens = self._issue_tokens(user["_id"])
        return {"user": user, "tokens": tokens}

    def refresh(self, refresh_token: str) -> dict[str, Any]:
        payload = decode_token(refresh_token, expected_type="refresh")
        record = self.refresh_tokens.find_valid(payload["jti"])
        if record is None:
            raise AuthenticationError("Refresh token has been revoked or has expired")

        user = self.users.find_by_id(record["user_id"])
        if user is None or not user.get("is_active", True):
            raise AuthenticationError("Account is no longer active")

        # Rotate: the presented token is retired as the new pair is issued.
        self.refresh_tokens.revoke(payload["jti"])
        return {"user": user, "tokens": self._issue_tokens(user["_id"])}

    def logout(self, refresh_token: str | None) -> int:
        if not refresh_token:
            return 0
        try:
            payload = decode_token(refresh_token, expected_type="refresh")
        except AuthenticationError:
            return 0
        return self.refresh_tokens.revoke(payload["jti"])

    def logout_everywhere(self, user_id: Any) -> int:
        return self.refresh_tokens.revoke_all_for_user(user_id)

    # ----------------------------------------------------------------- users
    def get_user(self, user_id: Any) -> dict[str, Any]:
        user = self.users.find_by_id(user_id)
        if user is None:
            raise NotFoundError("User not found")
        return user

    def update_user(self, user_id: Any, changes: dict[str, Any]) -> dict[str, Any]:
        patch = {k: v for k, v in changes.items() if v is not None}
        if "is_active" in changes and changes["is_active"] is False:
            patch["is_active"] = False
        if not patch:
            return self.get_user(user_id)
        patch["updated_at"] = utc_now()
        self.users.update_one({"_id": user_id}, {"$set": patch})
        if patch.get("is_active") is False:
            self.refresh_tokens.revoke_all_for_user(user_id)
        return self.get_user(user_id)

    def change_password(self, user_id: Any, current_password: str, new_password: str) -> int:
        user = self.users.find_by_id(user_id)
        if user is None:
            raise NotFoundError("User not found")
        if not verify_password(current_password, user.get("password_hash", "")):
            raise AuthenticationError("Current password is incorrect")
        self.users.set_password_hash(user_id, hash_password(new_password))
        # Force every other device to re-authenticate.
        return self.refresh_tokens.revoke_all_for_user(user_id)

    # ---------------------------------------------------------------- tokens
    def issue_tokens_for(self, user_id: Any) -> dict[str, Any]:
        """Issue a token pair for an already-authenticated user (public helper)."""
        return self._issue_tokens(user_id)

    def _issue_tokens(self, user_id: Any) -> dict[str, Any]:
        access_token, _ = create_access_token(str(user_id))
        refresh_token, jti, expires_at = create_refresh_token(str(user_id))
        self.refresh_tokens.store(user_id=user_id, jti=jti, expires_at=expires_at)
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "expires_in": settings.access_token_expire_minutes * 60,
        }

    def purge_expired_tokens(self) -> int:
        return self.refresh_tokens.purge_expired()

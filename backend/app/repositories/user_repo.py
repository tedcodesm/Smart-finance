"""Repository for `users` and `refresh_tokens`."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from app.core.logging import get_logger
from app.core.time_utils import strip_tz, utc_now
from app.repositories.base import BaseRepository

logger = get_logger(__name__)


class UserRepository(BaseRepository):
    collection_name = "users"

    def find_by_email(self, email: str) -> dict[str, Any] | None:
        return self.find_one({"email": email.strip().lower()})

    def find_by_id(self, user_id: Any) -> dict[str, Any] | None:
        return self.find_one({"_id": user_id})

    def create_user(self, *, name: str, email: str, password_hash: str, base_currency: str) -> dict[str, Any]:
        now = utc_now()
        document = {
            "name": name,
            "email": email.strip().lower(),
            "password_hash": password_hash,
            "base_currency": base_currency.upper(),
            "is_active": True,
            "monthly_income_estimate": None,
            "created_at": now,
            "updated_at": now,
            "last_login_at": None,
        }
        self.collection.insert_one(document)
        return document

    def set_password_hash(self, user_id: Any, password_hash: str) -> None:
        self.update_one({"_id": user_id}, {"$set": {"password_hash": password_hash, "updated_at": utc_now()}})

    def touch_login(self, user_id: Any) -> None:
        self.update_one({"_id": user_id}, {"$set": {"last_login_at": utc_now()}})

    def list_active_user_ids(self) -> list[Any]:
        return [doc["_id"] for doc in self.find({"is_active": True}, projection={"_id": 1})]


class RefreshTokenRepository(BaseRepository):
    """Server-side registry so refresh tokens can be revoked before expiry."""

    collection_name = "refresh_tokens"

    def store(self, *, user_id: Any, jti: str, expires_at: datetime) -> str:
        self.collection.insert_one(
            {
                "user_id": user_id,
                "jti": jti,
                "revoked": False,
                "created_at": utc_now(),
                "expires_at": strip_tz(expires_at),
            }
        )
        return jti

    def find_valid(self, jti: str) -> dict[str, Any] | None:
        return self.find_one({"jti": jti, "revoked": False, "expires_at": {"$gt": utc_now()}})

    def revoke(self, jti: str) -> int:
        return self.update_one({"jti": jti}, {"$set": {"revoked": True, "revoked_at": utc_now()}})

    def revoke_all_for_user(self, user_id: Any) -> int:
        return self.update_many(
            {"user_id": user_id, "revoked": False}, {"$set": {"revoked": True, "revoked_at": utc_now()}}
        )

    def purge_expired(self) -> int:
        return self.delete_many({"expires_at": {"$lte": utc_now()}})

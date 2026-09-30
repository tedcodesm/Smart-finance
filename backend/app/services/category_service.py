"""Category management, including the system/user split."""

from __future__ import annotations

from typing import Any

from app.core.constants import TransactionType
from app.core.errors import ConflictError, NotFoundError, ValidationError
from app.core.logging import get_logger
from app.core.money import round_money
from app.core.time_utils import resolve_range
from app.models.base import serialize_doc
from app.repositories.category_repo import CategoryRepository
from app.repositories.transaction_repo import TransactionRepository

logger = get_logger(__name__)


class CategoryService:
    def __init__(
        self,
        category_repo: CategoryRepository,
        transaction_repo: TransactionRepository | None = None,
    ) -> None:
        self.categories = category_repo
        self.transactions = transaction_repo

    def list_categories(
        self,
        user_id: Any,
        *,
        type: TransactionType | None = None,
        include_archived: bool = False,
        with_usage: bool = True,
    ) -> list[dict[str, Any]]:
        rows = self.categories.list_for_user(user_id, type=type, include_archived=include_archived)
        if with_usage and rows:
            start, end, _ = resolve_range("monthly")
            counts = self.categories.usage_counts(user_id, start, end)
            for row in rows:
                row["transaction_count"] = counts.get(row["_id"], 0)
        return rows

    def create_category(self, user_id: Any, payload: dict[str, Any]) -> dict[str, Any]:
        name = payload["name"]
        type_value = TransactionType(payload["type"]).value
        if self.categories.find_by_name(user_id, name, type_value):
            raise ConflictError(f"A {type_value} category named '{name}' already exists")
        document = self.categories.create_category(
            user_id=user_id,
            name=name,
            type=type_value,
            icon=payload.get("icon"),
            color=payload.get("color"),
            monthly_typical_amount=(
                round_money(payload["monthly_typical_amount"])
                if payload.get("monthly_typical_amount") is not None
                else None
            ),
        )
        return document

    def update_category(self, user_id: Any, category_id: Any, changes: dict[str, Any]) -> dict[str, Any] | None:
        existing = self.categories.find_one({"_id": category_id, "user_id": user_id, "is_system": False})
        if existing is None:
            if self.categories.find_one({"_id": category_id, "is_system": True}):
                raise ValidationError("System categories cannot be modified")
            return None

        if changes.get("name"):
            clash = self.categories.find_by_name(user_id, changes["name"], existing["type"])
            if clash and str(clash["_id"]) != str(category_id):
                raise ConflictError(f"A {existing['type']} category named '{changes['name']}' already exists")

        patch = {k: v for k, v in changes.items() if v is not None}
        if "monthly_typical_amount" in patch and patch["monthly_typical_amount"] is not None:
            patch["monthly_typical_amount"] = round_money(patch["monthly_typical_amount"])
        self.categories.update_category(category_id, user_id, patch)
        return self.categories.find_one({"_id": category_id})

    def delete_category(self, user_id: Any, category_id: Any, *, force: bool = False) -> int:
        existing = self.categories.find_one({"_id": category_id, "user_id": user_id, "is_system": False})
        if existing is None:
            if self.categories.find_one({"_id": category_id, "is_system": True}):
                raise ValidationError("System categories cannot be deleted")
            return 0

        # Deleting a category that still has transactions would orphan history,
        # so require an explicit override (which the API exposes as ?force=true).
        if not force and self.transactions is not None:
            in_use = self.transactions.count({"user_id": user_id, "category_id": category_id})
            if in_use:
                raise ConflictError(
                    f"Category is used by {in_use} transaction(s). "
                    "Archive it instead, or re-issue the request with force=true."
                )
        return self.categories.delete_category(category_id, user_id)

    def usage_count(self, user_id: Any, category_id: Any) -> int:
        if self.transactions is None:
            return 0
        return self.transactions.count({"user_id": user_id, "category_id": category_id})

    def get_category(self, user_id: Any, category_id: Any) -> dict[str, Any] | None:
        return self.categories.get_owned(category_id, user_id)

    def require(self, user_id: Any, category_id: Any) -> dict[str, Any]:
        category = self.get_category(user_id, category_id)
        if category is None:
            raise NotFoundError("Category not found")
        return category

    def seed_defaults(self) -> int:
        return self.categories.seed_system_categories()

    def serialize(self, document: dict[str, Any] | None) -> dict[str, Any] | None:
        return serialize_doc(document)

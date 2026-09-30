"""Repository for `categories`."""

from __future__ import annotations

from typing import Any

from app.core.constants import SYSTEM_EXPENSE_CATEGORIES, SYSTEM_INCOME_CATEGORIES, TransactionType
from app.core.time_utils import utc_now
from app.repositories.base import BaseRepository


class CategoryRepository(BaseRepository):
    collection_name = "categories"

    # ------------------------------------------------------------- read paths
    def list_for_user(
        self,
        user_id: Any,
        *,
        type: TransactionType | str | None = None,
        include_archived: bool = False,
        system_only: bool = False,
    ) -> list[dict[str, Any]]:
        """System categories (`user_id: None`) plus the user's own."""
        query: dict[str, Any] = {"$or": [{"user_id": user_id}, {"user_id": None, "is_system": True}]}
        if type is not None:
            query["type"] = str(type)
        if not include_archived:
            query["is_archived"] = False
        if system_only:
            query = {"user_id": None, "is_system": True}
            if type is not None:
                query["type"] = str(type)
        return self.find(query, sort=[("type", 1), ("name", 1)])

    def find_by_name(self, user_id: Any, name: str, type: str) -> dict[str, Any] | None:
        return self.find_one({"user_id": user_id, "name": name, "type": type, "is_system": False})

    def get_owned(self, category_id: Any, user_id: Any) -> dict[str, Any] | None:
        """Fetch a category only if the user owns it or it is a system category."""
        return self.find_one(
            {"_id": category_id, "$or": [{"user_id": user_id}, {"user_id": None, "is_system": True}]}
        )

    def name_map(self, user_id: Any) -> dict[Any, dict[str, Any]]:
        return {doc["_id"]: doc for doc in self.list_for_user(user_id, include_archived=True)}

    def usage_counts(self, user_id: Any, start: Any, end: Any) -> dict[Any, int]:
        """Transaction count per category within a window (for list views)."""
        pipeline = [
            {"$match": {"user_id": user_id, "date": {"$gte": start, "$lte": end}}},
            {"$group": {"_id": "$category_id", "count": {"$sum": 1}}},
        ]
        return {row["_id"]: row["count"] for row in self.db["transactions"].aggregate(pipeline)}

    # ------------------------------------------------------------ write paths
    def create_category(
        self,
        *,
        user_id: Any,
        name: str,
        type: str,
        icon: str | None = None,
        color: str | None = None,
        monthly_typical_amount: float | None = None,
    ) -> dict[str, Any]:
        now = utc_now()
        document = {
            "user_id": user_id,
            "name": name,
            "type": type,
            "icon": icon,
            "color": color,
            "monthly_typical_amount": monthly_typical_amount,
            "is_system": False,
            "is_archived": False,
            "created_at": now,
            "updated_at": now,
        }
        self.collection.insert_one(document)
        return document

    def update_category(self, category_id: Any, user_id: Any, changes: dict[str, Any]) -> int:
        changes = {k: v for k, v in changes.items() if v is not None or k in {"icon", "color", "monthly_typical_amount"}}
        if not changes:
            return 0
        changes["updated_at"] = utc_now()
        return self.update_one({"_id": category_id, "user_id": user_id}, {"$set": changes})

    def delete_category(self, category_id: Any, user_id: Any) -> int:
        return self.delete_one({"_id": category_id, "user_id": user_id, "is_system": False})

    # ------------------------------------------------------------------ seed
    def seed_system_categories(self) -> int:
        """Idempotently insert the default category set. Returns rows created."""
        created = 0
        now = utc_now()
        for type_name, catalogue in (
            (TransactionType.EXPENSE.value, SYSTEM_EXPENSE_CATEGORIES),
            (TransactionType.INCOME.value, SYSTEM_INCOME_CATEGORIES),
        ):
            for spec in catalogue:
                if self.exists({"user_id": None, "name": spec["name"], "type": type_name}):
                    continue
                self.collection.insert_one(
                    {
                        "user_id": None,
                        "name": spec["name"],
                        "type": type_name,
                        "icon": spec.get("icon"),
                        "color": spec.get("color"),
                        "monthly_typical_amount": spec.get("monthly_typical_amount"),
                        "is_system": True,
                        "is_archived": False,
                        "created_at": now,
                        "updated_at": now,
                    }
                )
                created += 1
        return created

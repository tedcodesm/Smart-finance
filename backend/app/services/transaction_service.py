"""Transaction lifecycle.

Write path (requirement 5):
    create/update -> recompute affected budgets -> raise threshold Alerts.
The ML models are never called synchronously here; predictions are produced by
the nightly batch job and surfaced through the `predictions` collection.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Sequence

from app.core.constants import TransactionType
from app.core.errors import NotFoundError, ValidationError
from app.core.logging import get_logger
from app.core.money import round_money
from app.core.time_utils import strip_tz, utc_now
from app.models.base import to_object_id
from app.repositories.category_repo import CategoryRepository
from app.repositories.transaction_repo import TransactionRepository
from app.services.budget_service import BudgetService

logger = get_logger(__name__)


class TransactionService:
    def __init__(
        self,
        transaction_repo: TransactionRepository,
        category_repo: CategoryRepository,
        budget_service: BudgetService,
    ) -> None:
        self.transactions = transaction_repo
        self.categories = category_repo
        self.budgets = budget_service

    # ------------------------------------------------------------------ CRUD
    def create_transaction(self, user_id: Any, payload: dict[str, Any]) -> dict[str, Any]:
        category_id = self._resolve_category(user_id, payload.get("category_id"))
        txn_type = TransactionType(payload.get("type"))
        if txn_type is TransactionType.EXPENSE:
            category = self.categories.find_one({"_id": category_id}) or {}
            if category.get("type") != TransactionType.EXPENSE.value:
                raise ValidationError("An expense must reference an expense category")

        document = self.transactions.build_document(
            user_id=user_id,
            category_id=category_id,
            type=txn_type.value,
            amount=round_money(payload["amount"]),
            date=strip_tz(payload.get("date") or utc_now()),
            description=payload.get("description"),
            notes=payload.get("notes"),
            is_recurring=bool(payload.get("is_recurring", False)),
        )
        self.transactions.insert_one(document)
        self._after_write(user_id, txn_type, category_id)
        return self.enrich(document)

    def bulk_create(self, user_id: Any, payloads: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        documents: list[dict[str, Any]] = []
        touched: set[tuple[str, Any]] = set()
        for payload in payloads:
            category_id = self._resolve_category(user_id, payload.get("category_id"))
            txn_type = TransactionType(payload.get("type"))
            date = strip_tz(payload.get("date") or utc_now())
            document = self.transactions.build_document(
                user_id=user_id,
                category_id=category_id,
                type=txn_type.value,
                amount=round_money(payload["amount"]),
                date=date,
                description=payload.get("description"),
                notes=payload.get("notes"),
                is_recurring=bool(payload.get("is_recurring", False)),
            )
            documents.append(document)
            if txn_type is TransactionType.EXPENSE:
                touched.add((txn_type.value, category_id))

        self.transactions.insert_many(documents)
        for category_id in {category for _, category in touched}:
            self._after_write(user_id, TransactionType.EXPENSE, category_id)
        return self.enrich_many(documents)

    def update_transaction(self, user_id: Any, transaction_id: Any, changes: dict[str, Any]) -> dict[str, Any]:
        existing = self.transactions.find_one({"_id": transaction_id, "user_id": user_id})
        if existing is None:
            raise NotFoundError("Transaction not found")

        patch: dict[str, Any] = {}
        new_type = TransactionType(changes["type"]) if changes.get("type") else TransactionType(existing["type"])
        old_category = existing.get("category_id")

        if changes.get("amount") is not None:
            patch["amount"] = round_money(changes["amount"])
        if new_type.value != existing["type"]:
            patch["type"] = new_type.value
        if changes.get("category_id") is not None:
            new_category = self._resolve_category(user_id, changes["category_id"])
            if str(new_category) != str(old_category):
                patch["category_id"] = new_category
        if changes.get("date") is not None:
            new_date = strip_tz(changes["date"])
            patch["date"] = new_date
            from app.core.time_utils import time_buckets

            patch.update(time_buckets(new_date))
        for field in ("description", "notes", "is_recurring"):
            if field in changes:
                patch[field] = changes[field]
        if not patch:
            return self.enrich(existing)

        patch["updated_at"] = strip_tz(utc_now())
        updated = self.transactions.find_one_and_update(
            {"_id": transaction_id, "user_id": user_id}, {"$set": patch}, return_after=True
        )
        assert updated is not None

        # Both the old and the new category may now be over/under threshold.
        for category_id in {old_category, updated.get("category_id")}:
            self._after_write(user_id, new_type, category_id)
        return self.enrich(updated)

    def delete_transaction(self, user_id: Any, transaction_id: Any) -> dict[str, Any] | None:
        existing = self.transactions.find_one({"_id": transaction_id, "user_id": user_id})
        if existing is None:
            return None
        self.transactions.delete_one({"_id": transaction_id, "user_id": user_id})
        self._after_write(user_id, TransactionType(existing["type"]), existing.get("category_id"))
        return existing

    def get_transaction(self, user_id: Any, transaction_id: Any) -> dict[str, Any] | None:
        document = self.transactions.find_one({"_id": transaction_id, "user_id": user_id})
        return self.enrich(document) if document else None

    def list_transactions(
        self,
        user_id: Any,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        type: TransactionType | None = None,
        category_ids: list[Any] | None = None,
        min_amount: float | None = None,
        max_amount: float | None = None,
        search: str | None = None,
        sort: list[tuple[str, int]] | None = None,
        skip: int = 0,
        limit: int = 25,
    ) -> tuple[list[dict[str, Any]], int]:
        common = {
            "start": start,
            "end": end,
            "type": type,
            "category_ids": category_ids,
            "min_amount": min_amount,
            "max_amount": max_amount,
            "search": search,
        }
        rows = self.transactions.list_for_user(user_id, sort=sort, skip=skip, limit=limit, **common)
        total = self.transactions.count_for_user(user_id, **common)
        return self.enrich_many(rows), total

    # ---------------------------------------------------------------- helpers
    def _resolve_category(self, user_id: Any, category_id: Any) -> Any:
        if not category_id:
            raise ValidationError("category_id is required")
        oid = to_object_id(category_id, field="category_id")
        if self.categories.get_owned(oid, user_id) is None:
            raise ValidationError("category_id does not reference a category you can use")
        return oid

    def _after_write(self, user_id: Any, txn_type: TransactionType, category_id: Any) -> None:
        """Re-evaluate budgets and raise alerts, executed inline per requirement 5.

        Evaluation is always anchored to *now*, never to the transaction's date:
        a backdated expense must still be measured against the budget the user
        currently sees, and a future-dated one must not resurrect a period that
        has already closed.
        """
        if txn_type is not TransactionType.EXPENSE:
            return
        try:
            created = self.budgets.evaluate_budgets(user_id, as_of=strip_tz(utc_now()))
        except Exception:  # pragma: no cover - never fail a write on alerting
            logger.exception("Budget evaluation failed for user=%s", user_id)
            return
        for alert in created:
            logger.info(
                "Alert raised: type=%s budget=%s", alert.get("type"), (alert.get("context") or {}).get("budget_id")
            )

    def enrich(self, document: dict[str, Any]) -> dict[str, Any]:
        """Attach category display fields and a signed amount for the UI."""
        if document is None:
            return {}
        enriched = dict(document)
        category_id = enriched.get("category_id")
        if category_id is not None:
            category = self.categories.find_one({"_id": category_id}, projection={"name": 1, "color": 1, "icon": 1})
            if category:
                enriched["category_name"] = category.get("name")
                enriched["category_color"] = category.get("color")
                enriched["category_icon"] = category.get("icon")
        amount = float(enriched.get("amount") or 0.0)
        enriched["signed_amount"] = -amount if enriched.get("type") == TransactionType.EXPENSE.value else amount
        return enriched

    def enrich_many(self, documents: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        """Category-join in one query instead of N per-document lookups."""
        if not documents:
            return []
        category_ids = {doc["category_id"] for doc in documents if doc.get("category_id") is not None}
        lookup: dict[Any, dict[str, Any]] = {}
        if category_ids:
            lookup = {doc["_id"]: doc for doc in self.categories.find({"_id": {"$in": list(category_ids)}})}
        enriched: list[dict[str, Any]] = []
        for document in documents:
            row = dict(document)
            category = lookup.get(document.get("category_id"))
            if category:
                row["category_name"] = category.get("name")
                row["category_color"] = category.get("color")
                row["category_icon"] = category.get("icon")
            amount = float(row.get("amount") or 0.0)
            row["signed_amount"] = -amount if row.get("type") == TransactionType.EXPENSE.value else amount
            enriched.append(row)
        return enriched

    def summary(
        self, user_id: Any, *, start: datetime | None = None, end: datetime | None = None
    ) -> dict[str, float]:
        from app.models.transaction import TransactionSummary

        totals = self.transactions.totals(user_id, start=start, end=end)
        income = round_money(totals["income"])
        expense = round_money(totals["expense"])
        count = totals["income_count"] + totals["expense_count"]
        return TransactionSummary(
            income=income,
            expense=expense,
            net=round_money(income - expense),
            transaction_count=count,
            average_expense=round_money(totals["average_expense"]),
            largest_expense=round_money(totals["largest_expense"]) or None,
        ).model_dump()

    def reconcile_category(self, user_id: Any, category_id: Any) -> int:
        """Recompute a budget's cached spend from scratch (data repair helper)."""
        return self.budgets.budgets.reset_spent_for_category(user_id, category_id)

    def export_rows(
        self,
        user_id: Any,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        type: TransactionType | str | None = None,
        category_ids: list[Any] | None = None,
        limit: int = 5000,
    ) -> list[dict[str, Any]]:
        """Flat rows for CSV/PDF export (joined with category names)."""
        names = self.categories.name_map(user_id)
        rows = self.transactions.list_for_user(
            user_id,
            start=start,
            end=end,
            type=type,
            category_ids=category_ids,
            sort=[("date", 1)],
            limit=limit,
        )
        out: list[dict[str, Any]] = []
        for row in rows:
            category = names.get(row.get("category_id"), {})
            amount = float(row.get("amount") or 0.0)
            out.append(
                {
                    "date": row["date"].date().isoformat(),
                    "type": row.get("type"),
                    "category": category.get("name", "Uncategorised"),
                    "description": row.get("description") or "",
                    "amount": amount,
                    "signed_amount": -amount if row.get("type") == TransactionType.EXPENSE.value else amount,
                    "is_recurring": "yes" if row.get("is_recurring") else "no",
                }
            )
        return out


def parse_category_ids(raw: str | None) -> list[Any] | None:
    """Parse a comma-separated `category_id` query parameter."""
    if not raw:
        return None
    return [to_object_id(token.strip(), field="category_id") for token in raw.split(",") if token.strip()]

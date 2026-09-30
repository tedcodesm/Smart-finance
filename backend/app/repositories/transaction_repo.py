"""Repository for `transactions` - the highest-volume collection.

Reads are shaped around the denormalised `period_key` / `week_key` / `date`
fields so that grouping happens on indexed scalars rather than date arithmetic
inside aggregation pipelines.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable

from app.core.constants import TransactionType
from app.core.time_utils import time_buckets
from app.repositories.base import BaseRepository

AGGREGATE_PROJECTION = {"_id": 0}


class TransactionRepository(BaseRepository):
    collection_name = "transactions"

    # -------------------------------------------------------- document shaping
    @staticmethod
    def build_document(
        *,
        user_id: Any,
        category_id: Any,
        type: str,
        amount: float,
        date: datetime,
        description: str | None = None,
        notes: str | None = None,
        is_recurring: bool = False,
    ) -> dict[str, Any]:
        from app.core.time_utils import utc_now

        now = utc_now()
        document: dict[str, Any] = {
            "user_id": user_id,
            "category_id": category_id,
            "type": type,
            "amount": float(amount),
            "date": date,
            "description": description,
            "notes": notes,
            "is_recurring": bool(is_recurring),
            **time_buckets(date),
            "created_at": now,
            "updated_at": now,
        }
        return document

    # ------------------------------------------------------------------ query
    def list_for_user(
        self,
        user_id: Any,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        type: TransactionType | str | None = None,
        category_ids: Iterable[Any] | None = None,
        min_amount: float | None = None,
        max_amount: float | None = None,
        search: str | None = None,
        sort: list[tuple[str, int]] | None = None,
        skip: int = 0,
        limit: int = 25,
    ) -> list[dict[str, Any]]:
        return self.find(self._filter(user_id, start=start, end=end, type=type, category_ids=category_ids,
                                     min_amount=min_amount, max_amount=max_amount, search=search),
                         sort=sort or [("date", -1), ("created_at", -1)], skip=skip, limit=limit)

    def count_for_user(
        self,
        user_id: Any,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        type: TransactionType | str | None = None,
        category_ids: Iterable[Any] | None = None,
        min_amount: float | None = None,
        max_amount: float | None = None,
        search: str | None = None,
    ) -> int:
        return self.count(self._filter(user_id, start=start, end=end, type=type, category_ids=category_ids,
                                       min_amount=min_amount, max_amount=max_amount, search=search))

    @staticmethod
    def _filter(
        user_id: Any,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        type: TransactionType | str | None = None,
        category_ids: Iterable[Any] | None = None,
        min_amount: float | None = None,
        max_amount: float | None = None,
        search: str | None = None,
    ) -> dict[str, Any]:
        query: dict[str, Any] = {"user_id": user_id}
        if start is not None or end is not None:
            window: dict[str, Any] = {}
            if start is not None:
                window["$gte"] = start
            if end is not None:
                window["$lte"] = end
            query["date"] = window
        if type is not None:
            query["type"] = str(type)
        if category_ids:
            query["category_id"] = {"$in": list(category_ids)}
        if min_amount is not None or max_amount is not None:
            bounds: dict[str, Any] = {}
            if min_amount is not None:
                bounds["$gte"] = min_amount
            if max_amount is not None:
                bounds["$lte"] = max_amount
            query["amount"] = bounds
        if search:
            query["$or"] = [
                {"description": {"$regex": search, "$options": "i"}},
                {"notes": {"$regex": search, "$options": "i"}},
            ]
        return query

    # ------------------------------------------------------------- aggregates
    def totals(self, user_id: Any, *, start: datetime | None = None, end: datetime | None = None,
               category_ids: Iterable[Any] | None = None) -> dict[str, float]:
        match = self._filter(user_id, start=start, end=end, category_ids=category_ids)
        match.pop("$or", None)
        pipeline = [
            {"$match": match},
            {
                "$group": {
                    "_id": "$type",
                    "total": {"$sum": "$amount"},
                    "count": {"$sum": 1},
                    "avg": {"$avg": "$amount"},
                    "max": {"$max": "$amount"},
                }
            },
        ]
        rows = {row["_id"]: row for row in self.aggregate(pipeline)}
        income = rows.get("income", {})
        expense = rows.get("expense", {})
        return {
            "income": float(income.get("total", 0.0) or 0.0),
            "expense": float(expense.get("total", 0.0) or 0.0),
            "income_count": int(income.get("count", 0) or 0),
            "expense_count": int(expense.get("count", 0) or 0),
            "average_expense": float(expense.get("avg", 0.0) or 0.0),
            "largest_expense": float(expense["max"]) if expense.get("max") is not None else 0.0,
        }

    def totals_by_type(self, user_id: Any, type: TransactionType | str, *, start: datetime | None = None,
                       end: datetime | None = None) -> float:
        match = self._filter(user_id, start=start, end=end, type=type)
        match.pop("$or", None)
        rows = self.aggregate([{"$match": match}, {"$group": {"_id": None, "total": {"$sum": "$amount"}}}])
        return float(rows[0]["total"]) if rows else 0.0

    def by_category(
        self,
        user_id: Any,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        type: TransactionType | str = TransactionType.EXPENSE,
    ) -> list[dict[str, Any]]:
        match = self._filter(user_id, start=start, end=end, type=type)
        match.pop("$or", None)
        pipeline = [
            {"$match": match},
            {
                "$group": {
                    "_id": "$category_id",
                    "total": {"$sum": "$amount"},
                    "count": {"$sum": 1},
                    "avg": {"$avg": "$amount"},
                    "max": {"$max": "$amount"},
                }
            },
            {"$sort": {"total": -1}},
        ]
        return self.aggregate(pipeline)

    def by_period(
        self,
        user_id: Any,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        bucket: str = "period_key",
        type: TransactionType | str | None = None,
    ) -> list[dict[str, Any]]:
        """Grouped totals per time bucket, ascending.

        `bucket` is one of `period_key` (month), `week_key` (ISO week), `day_of_week`,
        `day_of_month`, `day`, or `year` - all denormalised onto the document.
        """
        match = self._filter(user_id, start=start, end=end, type=type)
        match.pop("$or", None)
        pipeline = [
            {"$match": match},
            {
                "$group": {
                    "_id": {
                        "bucket": f"${bucket}",
                        "type": "$type",
                    },
                    "total": {"$sum": "$amount"},
                    "count": {"$sum": 1},
                }
            },
            {"$sort": {"_id.bucket": 1}},
        ]
        return self.aggregate(pipeline)

    def category_period_matrix(
        self,
        user_id: Any,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        type: TransactionType | str = TransactionType.EXPENSE,
    ) -> list[dict[str, Any]]:
        match = self._filter(user_id, start=start, end=end, type=type)
        match.pop("$or", None)
        pipeline = [
            {"$match": match},
            {
                "$group": {
                    "_id": {"category_id": "$category_id", "period_key": "$period_key"},
                    "total": {"$sum": "$amount"},
                    "count": {"$sum": 1},
                }
            },
            {"$sort": {"_id.period_key": 1, "total": -1}},
        ]
        return self.aggregate(pipeline)

    def spend_for_budget(
        self,
        user_id: Any,
        *,
        start: datetime,
        end: datetime,
        category_id: Any | None = None,
    ) -> float:
        """Expense total inside a budget window, optionally for one category."""
        match: dict[str, Any] = {
            "user_id": user_id,
            "type": TransactionType.EXPENSE.value,
            "date": {"$gte": start, "$lt": end},
        }
        if category_id is not None:
            match["category_id"] = category_id
        rows = self.aggregate([{"$match": match}, {"$group": {"_id": None, "total": {"$sum": "$amount"}}}])
        return float(rows[0]["total"]) if rows else 0.0

    def first_and_last_dates(self, user_id: Any) -> tuple[datetime | None, datetime | None]:
        oldest = self.find_one({"user_id": user_id}, sort=[("date", 1)], projection={"date": 1, "_id": 0})
        newest = self.find_one({"user_id": user_id}, sort=[("date", -1)], projection={"date": 1, "_id": 0})
        return (oldest or {}).get("date"), (newest or {}).get("date")

    def monthly_rollup(self, user_id: Any, *, months: int = 12) -> list[dict[str, Any]]:
        """Per-month totals for the trailing `months` buckets (income/expense split)."""
        rows = self.aggregate(
            [
                {"$match": {"user_id": user_id}},
                {"$group": {"_id": {"pk": "$period_key", "t": "$type"}, "total": {"$sum": "$amount"},
                            "count": {"$sum": 1}}},
                {"$sort": {"_id.pk": -1}},
            ]
        )
        merged: dict[str, dict[str, Any]] = {}
        for row in rows:
            key = row["_id"]["pk"]
            bucket = merged.setdefault(key, {"period_key": key, "income": 0.0, "expense": 0.0, "count": 0})
            bucket["count"] += row["count"]
            if row["_id"]["t"] == "income":
                bucket["income"] = float(row["total"])
            else:
                bucket["expense"] = float(row["total"])
        ordered = sorted(merged.values(), key=lambda b: b["period_key"], reverse=True)[:months]
        for bucket in ordered:
            bucket["net"] = round(bucket["income"] - bucket["expense"], 2)
        return list(reversed(ordered))

    # ------------------------------------------------------------ ML exports
    def export_for_ml(
        self,
        *,
        user_ids: list[Any] | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        types: list[str] | None = None,
        limit: int = 0,
    ) -> list[dict[str, Any]]:
        """Flat, lean projection used by the ML training/inference pipeline."""
        query: dict[str, Any] = {}
        if user_ids:
            query["user_id"] = {"$in": user_ids}
        if start is not None or end is not None:
            window: dict[str, Any] = {}
            if start is not None:
                window["$gte"] = start
            if end is not None:
                window["$lte"] = end
            query["date"] = window
        if types:
            query["type"] = {"$in": types}
        projection = {
            "_id": 0,
            "user_id": 1,
            "category_id": 1,
            "type": 1,
            "amount": 1,
            "date": 1,
            "year": 1,
            "month": 1,
            "day": 1,
            "day_of_week": 1,
            "day_of_month": 1,
            "week_of_year": 1,
            "period_key": 1,
            "week_key": 1,
            "is_recurring": 1,
        }
        return self.find(query, sort=[("user_id", 1), ("date", 1)], limit=limit, projection=projection)

    def distinct_users(self) -> list[Any]:
        return self.distinct("user_id", {})

    def distinct_period_keys(self, user_id: Any) -> list[str]:
        return sorted(self.distinct("period_key", {"user_id": user_id}))

    # ----------------------------------------------------------------- writes
    def delete_for_user(self, user_id: Any, transaction_ids: list[Any]) -> int:
        if not transaction_ids:
            return 0
        return self.delete_many({"user_id": user_id, "_id": {"$in": transaction_ids}})

    def delete_all_for_user(self, user_id: Any) -> int:
        return self.delete_many({"user_id": user_id})

    def orphan_category_ids(self, user_id: Any) -> list[Any]:
        return self.distinct("category_id", {"user_id": user_id})

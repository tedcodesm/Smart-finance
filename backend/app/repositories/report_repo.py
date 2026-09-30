"""Repository for `reports`."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from app.core.constants import ReportType
from app.repositories.base import BaseRepository


class ReportRepository(BaseRepository):
    collection_name = "reports"

    def store(
        self,
        *,
        user_id: Any,
        report_type: ReportType | str,
        period: str,
        period_start: datetime,
        period_end: datetime,
        period_key: str | None,
        params: dict[str, Any],
        data: dict[str, Any],
        summary: dict[str, Any],
        expires_at: datetime | None = None,
    ) -> str:
        from app.core.time_utils import utc_now

        return self.insert_one(
            {
                "user_id": user_id,
                "report_type": str(report_type),
                "period": period,
                "period_start": period_start,
                "period_end": period_end,
                "period_key": period_key,
                "params": params,
                "data": data,
                "summary": summary,
                "format": "json",
                "generated_at": utc_now(),
                "expires_at": expires_at,
            }
        )

    def list_for_user(
        self,
        user_id: Any,
        *,
        report_type: ReportType | str | None = None,
        skip: int = 0,
        limit: int = 25,
    ) -> list[dict[str, Any]]:
        query: dict[str, Any] = {"user_id": user_id}
        if report_type is not None:
            query["report_type"] = str(report_type)
        return self.find(query, sort=[("generated_at", -1)], skip=skip, limit=limit)

    def count_for_user(self, user_id: Any, *, report_type: ReportType | str | None = None) -> int:
        query: dict[str, Any] = {"user_id": user_id}
        if report_type is not None:
            query["report_type"] = str(report_type)
        return self.count(query)

    def get_owned(self, report_id: Any, user_id: Any) -> dict[str, Any] | None:
        return self.find_one({"_id": report_id, "user_id": user_id})

    def delete_owned(self, report_id: Any, user_id: Any) -> int:
        return self.delete_one({"_id": report_id, "user_id": user_id})

    def purge_expired(self, as_of: datetime | None = None) -> int:
        from app.core.time_utils import utc_now

        return self.delete_many({"expires_at": {"$ne": None, "$lte": as_of or utc_now()}})

    def delete_all_for_user(self, user_id: Any) -> int:
        return self.delete_many({"user_id": user_id})

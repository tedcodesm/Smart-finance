"""Repository for `predictions` (ML output) and `model_runs` (training log)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable

from app.core.constants import PredictionType
from app.core.time_utils import utc_now
from app.repositories.base import BaseRepository


class PredictionRepository(BaseRepository):
    collection_name = "predictions"

    def replace_for_scope(
        self,
        user_id: Any,
        *,
        type: PredictionType | str,
        category_id: Any | None = None,
        period_key: str | None = None,
        documents: Iterable[dict[str, Any]],
    ) -> list[str]:
        """Drop the previous predictions for this (type, category, period) then insert fresh ones.

        Keeps the collection bounded - the audit trail of what produced each batch
        lives in `model_runs`, not here.
        """
        documents = list(documents)
        self.delete_many(
            {
                "user_id": user_id,
                "type": str(type),
                "scope.category_id": category_id,
                "scope.period_key": period_key,
            }
        )
        return self.insert_many(documents)

    def latest_by_type(
        self, user_id: Any, *, types: Iterable[PredictionType | str] | None = None, limit_per_type: int = 1
    ) -> list[dict[str, Any]]:
        """Most recent prediction(s) per type, for dashboard widgets."""
        type_list = [str(t) for t in types] if types else [t.value for t in PredictionType]
        if not type_list:
            return []
        rows = self.find({"user_id": user_id, "type": {"$in": type_list}}, sort=[("generated_at", -1)])
        seen: dict[str, int] = {}
        out: list[dict[str, Any]] = []
        for row in rows:
            key = str(row["type"])
            if seen.get(key, 0) >= limit_per_type:
                continue
            seen[key] = seen.get(key, 0) + 1
            out.append(row)
        return out

    def latest_for(
        self,
        user_id: Any,
        *,
        type: PredictionType | str,
        category_id: Any | None = None,
        period_key: str | None = None,
    ) -> dict[str, Any] | None:
        query: dict[str, Any] = {"user_id": user_id, "type": str(type)}
        if category_id is not None:
            query["scope.category_id"] = category_id
        if period_key is not None:
            query["scope.period_key"] = period_key
        return self.find_one_document(query, sort=[("generated_at", -1)])

    def list_for_user(
        self,
        user_id: Any,
        *,
        types: Iterable[PredictionType | str] | None = None,
        since: datetime | None = None,
        category_id: Any | None = None,
        skip: int = 0,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        query: dict[str, Any] = {"user_id": user_id}
        if types:
            query["type"] = {"$in": [str(t) for t in types]}
        if since is not None:
            query["generated_at"] = {"$gte": since}
        if category_id is not None:
            query["scope.category_id"] = category_id
        return self.find(query, sort=[("generated_at", -1), ("type", 1)], skip=skip, limit=limit)

    def count_for_user(
        self,
        user_id: Any,
        *,
        types: Iterable[PredictionType | str] | None = None,
        since: datetime | None = None,
    ) -> int:
        query: dict[str, Any] = {"user_id": user_id}
        if types:
            query["type"] = {"$in": [str(t) for t in types]}
        if since is not None:
            query["generated_at"] = {"$gte": since}
        return self.count(query)

    def delete_all_for_user(self, user_id: Any) -> int:
        return self.delete_many({"user_id": user_id})

    def purge_expired(self, as_of: datetime | None = None) -> int:
        as_of = as_of or utc_now()
        return self.delete_many({"expires_at": {"$ne": None, "$lte": as_of}})

    def runs_in_batch(self, model_run_id: str) -> list[dict[str, Any]]:
        return self.find({"model_run_id": model_run_id}, sort=[("type", 1), ("value", -1)])


class ModelRunRepository(BaseRepository):
    collection_name = "model_runs"

    def record(
        self,
        *,
        run_id: str,
        model_type: str,
        model_version: str,
        metrics: dict[str, Any] | None = None,
        n_train: int | None = None,
        n_test: int | None = None,
        features: list[str] | None = None,
        artifact_path: str | None = None,
        notes: str | None = None,
        is_active: bool = True,
    ) -> str:
        self.deactivate(model_type)
        self.collection.insert_one(
            {
                "run_id": run_id,
                "model_type": model_type,
                "model_version": model_version,
                "metrics": metrics or {},
                "n_train": n_train,
                "n_test": n_test,
                "features": features or [],
                "artifact_path": artifact_path,
                "notes": notes,
                "is_active": is_active,
                "trained_at": utc_now(),
            }
        )
        return run_id

    def deactivate(self, model_type: str) -> int:
        return self.update_many({"model_type": model_type, "is_active": True}, {"$set": {"is_active": False}})

    def active_runs(self) -> list[dict[str, Any]]:
        return self.find({"is_active": True}, sort=[("model_type", 1)])

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        return self.find_one({"run_id": run_id})

    def history(self, model_type: str | None = None, *, limit: int = 20) -> list[dict[str, Any]]:
        query = {"model_type": model_type} if model_type else {}
        return self.find(query, sort=[("trained_at", -1)], limit=limit)

"""Repository layer.

All MongoDB access funnels through these classes: routers never touch the driver
directly, and every query is scoped by `user_id` at this layer (enforced again
in `app.api.deps` ownership checks).
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from bson import ObjectId
from pymongo import ASCENDING, DESCENDING, ReturnDocument
from pymongo.database import Database

from app.core.logging import get_logger

logger = get_logger(__name__)

Filter = Mapping[str, Any]
Update = Mapping[str, Any]
Pipeline = Sequence[Mapping[str, Any]]


class BaseRepository:
    collection_name: str

    def __init__(self, db: Database) -> None:
        self.db = db
        self.collection = db[self.collection_name]

    # ---------------------------------------------------------------- writes
    def insert_one(self, document: dict[str, Any]) -> str:
        result = self.collection.insert_one(document)
        return str(result.inserted_id)

    def insert_many(self, documents: Iterable[dict[str, Any]]) -> list[str]:
        documents = list(documents)
        if not documents:
            return []
        result = self.collection.insert_many(documents)
        return [str(oid) for oid in result.inserted_ids]

    def update_one(self, filter: Filter, update: Update, *, upsert: bool = False) -> int:
        result = self.collection.update_one(filter, update, upsert=upsert)
        return result.modified_count

    def update_many(self, filter: Filter, update: Update) -> int:
        return self.collection.update_many(filter, update).modified_count

    def replace_one(self, filter: Filter, document: dict[str, Any], *, upsert: bool = False) -> int:
        return self.collection.replace_one(filter, document, upsert=upsert).modified_count

    def find_one_and_update(
        self,
        filter: Filter,
        update: Update,
        *,
        return_after: bool = True,
        upsert: bool = False,
    ) -> dict[str, Any] | None:
        return self.collection.find_one_and_update(
            filter,
            update,
            return_document=ReturnDocument.AFTER if return_after else ReturnDocument.BEFORE,
            upsert=upsert,
        )

    def find_one_and_replace(
        self, filter: Filter, replacement: dict[str, Any], *, upsert: bool = False
    ) -> dict[str, Any] | None:
        return self.collection.find_one_and_replace(
            filter, replacement, return_document=ReturnDocument.AFTER, upsert=upsert
        )

    def delete_one(self, filter: Filter) -> int:
        return self.collection.delete_one(filter).deleted_count

    def delete_many(self, filter: Filter) -> int:
        return self.collection.delete_many(filter).deleted_count

    # ----------------------------------------------------------------- reads
    def find_one(self, filter: Filter, *, projection: dict[str, int] | None = None) -> dict[str, Any] | None:
        return self.collection.find_one(filter, projection)

    def find(
        self,
        filter: Filter,
        *,
        sort: list[tuple[str, int]] | None = None,
        skip: int = 0,
        limit: int = 0,
        projection: dict[str, int] | None = None,
    ) -> list[dict[str, Any]]:
        cursor = self.collection.find(filter, projection)
        if sort:
            cursor = cursor.sort(sort)
        if skip:
            cursor = cursor.skip(skip)
        if limit:
            cursor = cursor.limit(limit)
        return list(cursor)

    def find_one_document(self, filter: Filter, *, sort: list[tuple[str, int]] | None = None) -> dict[str, Any] | None:
        cursor = self.collection.find(filter)
        if sort:
            cursor = cursor.sort(sort)
        return next(cursor, None)

    def count(self, filter: Filter) -> int:
        return self.collection.count_documents(filter)

    def exists(self, filter: Filter) -> bool:
        return self.collection.find_one(filter, {"_id": 1}) is not None

    def distinct(self, field: str, filter: Filter) -> list[Any]:
        return list(self.collection.distinct(field, filter))

    def aggregate(self, pipeline: Pipeline) -> list[dict[str, Any]]:
        return list(self.collection.aggregate(pipeline))

    def bulk_write(self, operations: list[tuple[str, dict[str, Any]]]) -> int:
        """Execute `(op, payload)` pairs. `op` is one of insert/update/delete/replace."""
        from pymongo import InsertOne, UpdateOne, DeleteOne, ReplaceOne

        mapping = {
            "insert": InsertOne,
            "update": UpdateOne,
            "delete": DeleteOne,
            "replace": ReplaceOne,
        }
        if not operations:
            return 0
        requests = [mapping[op](payload) for op, payload in operations]  # type: ignore[misc]
        return len(self.collection.bulk_write(requests).upserted_ids) or len(requests)

    # --------------------------------------------------------------- helpers
    @staticmethod
    def oid(value: str | ObjectId) -> ObjectId:
        return value if isinstance(value, ObjectId) else ObjectId(value)

    def ensure_indexes(self) -> list[str]:
        from app.db.index_definitions import normalised

        created: list[str] = []
        for spec in self.index_specs():
            result = self.collection.create_index(**normalised(spec))
            created.append(result)
        return created

    def index_specs(self) -> list[dict[str, Any]]:
        from app.db.index_definitions import COLLECTION_INDEXES

        return list(COLLECTION_INDEXES.get(self.collection_name, []))


ASC = ASCENDING
DESC = DESCENDING

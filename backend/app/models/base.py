"""Shared model plumbing: ObjectId coercion and BSON -> JSON serialisation."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from bson import ObjectId
from bson.decimal128 import Decimal128
from pydantic import BaseModel, ConfigDict, field_serializer

from app.core.errors import ValidationError

_OBJECT_ID_LEN = 24


def to_object_id(value: Any, *, field: str = "id") -> ObjectId:
    """Coerce str/ObjectId to ObjectId, raising a 422 on bad input."""
    if isinstance(value, ObjectId):
        return value
    if isinstance(value, str) and len(value) == _OBJECT_ID_LEN:
        try:
            return ObjectId(value)
        except Exception as exc:  # pragma: no cover - defensive
            raise ValidationError(f"Invalid {field}: {value!r}") from exc
    raise ValidationError(f"Invalid {field}: expected a 24-character ObjectId, got {value!r}")


def to_object_id_or_none(value: Any, *, field: str = "id") -> ObjectId | None:
    if value is None or value == "":
        return None
    return to_object_id(value, field=field)


def jsonify(value: Any) -> Any:
    """Recursively convert BSON-native values into JSON-safe Python."""
    if isinstance(value, ObjectId):
        return str(value)
    if isinstance(value, Decimal128):
        return float(value.to_decimal())
    if isinstance(value, datetime):
        return value.isoformat() + ("Z" if value.tzinfo is None else "")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonify(v) for v in value]
    return value


def serialize_doc(doc: dict[str, Any] | None) -> dict[str, Any] | None:
    """Rename `_id` -> `id` and JSON-sanitise a raw BSON document."""
    if doc is None:
        return None
    out = {key: jsonify(value) for key, value in doc.items() if key != "_id"}
    out["id"] = str(doc["_id"]) if "_id" in doc else None
    return out


def serialize_docs(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [serialize_doc(doc) for doc in docs]


class MongoModel(BaseModel):
    """Base for read models: `id` is a string, datetimes serialise as ISO-8601."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    id: str | None = None

    @field_serializer("*", when_used="json")
    def _serialize_types(self, value: Any) -> Any:  # pragma: no cover - passthrough
        return jsonify(value)

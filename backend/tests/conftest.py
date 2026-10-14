"""Shared pytest fixtures: mongomock-backed TestClient with a registered user."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Iterator

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("ENABLE_SCHEDULER", "false")
# The API and `mercy_ml` share one MongoDB client, so the in-memory database has to
# use the configured name on both sides (otherwise the ML loaders read an empty db).
os.environ.setdefault("MONGO_DB_NAME", "mercy_test")
# Assigned, not setdefault: each test injects its own mongomock database via
# `override_database`, and the lifespan's IN_MEMORY_DB branch would replace that
# with one process-wide database, leaking state between tests. A developer's
# `.env` or shell export must not be able to turn it back on.
os.environ["IN_MEMORY_DB"] = "false"

import mongomock  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.db.client import override_database  # noqa: E402


@pytest.fixture()
def db() -> Iterator[Any]:
    client = mongomock.MongoClient()
    database = client[settings.mongo_db_name]
    override_database(database)
    yield database
    override_database(None)


@pytest.fixture()
def client(db: Any) -> Iterator[TestClient]:
    from app.main import app

    with TestClient(app) as test_client:
        yield test_client


def register(client: TestClient, **overrides: Any) -> dict[str, Any]:
    payload = {
        "name": overrides.get("name", "Ada Lovelace"),
        "email": overrides.get("email", "ada@example.com"),
        "password": overrides.get("password", "analytical1"),
        "base_currency": overrides.get("base_currency", "KES"),
    }
    response = client.post("/api/v1/auth/register", json=payload)
    assert response.status_code == 201, response.text
    return response.json()


def auth_headers(client: TestClient, account: dict[str, Any] | None = None) -> dict[str, str]:
    account = account or register(client)
    return {"Authorization": f"Bearer {account['tokens']['access_token']}"}


@pytest.fixture()
def account(client: TestClient) -> dict[str, Any]:
    return register(client)


@pytest.fixture()
def user_id(account: dict[str, Any]) -> str:
    return account["user"]["id"]


@pytest.fixture()
def headers(account: dict[str, Any]) -> dict[str, str]:
    return {"Authorization": f"Bearer {account['tokens']['access_token']}"}


def category_id(client: TestClient, headers: dict[str, str], name: str = "Groceries", type_: str = "expense") -> str:
    response = client.get("/api/v1/categories", headers=headers, params={"type": type_})
    assert response.status_code == 200, response.text
    for item in response.json()["items"]:
        if item["name"] == name:
            return item["id"]
    created = client.post(
        "/api/v1/categories",
        headers=headers,
        json={"name": name, "type": type_, "color": "#22c55e", "monthly_typical_amount": 400.0},
    )
    assert created.status_code == 201, created.text
    return created.json()["id"]

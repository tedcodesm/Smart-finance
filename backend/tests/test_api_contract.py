"""Regressions for the API contract bugs fixed alongside the pagination rework.

Each test here corresponds to a defect that shipped in the documented contract:
pagination query parameters that were silently ignored, an error envelope that
unmatched routes bypassed, and a demo account that could not sign in.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings

PAGE_SIZE = 3


def _seed(client: TestClient, headers: dict[str, str], count: int) -> None:
    cats = client.get("/api/v1/categories", headers=headers, params={"type": "expense"}).json()["items"]
    cat_id = cats[0]["id"]
    for index in range(count):
        response = client.post(
            "/api/v1/transactions",
            headers=headers,
            json={
                "amount": 10.0 + index,
                "type": "expense",
                "category_id": cat_id,
                "date": "2026-09-%02d" % (index + 1),
                "description": f"tx-{index}",
            },
        )
        assert response.status_code == 201, response.text


def test_limit_and_skip_are_honoured(client: TestClient, headers: dict[str, str]) -> None:
    """`?limit=` and `?skip=` are the documented contract; they must not be ignored."""
    _seed(client, headers, 10)

    limited = client.get("/api/v1/transactions", headers=headers, params={"limit": PAGE_SIZE})
    assert limited.status_code == 200
    assert len(limited.json()["items"]) == PAGE_SIZE
    assert limited.json()["meta"]["limit"] == PAGE_SIZE

    first = client.get("/api/v1/transactions", headers=headers, params={"limit": PAGE_SIZE, "sort": "date"})
    second = client.get(
        "/api/v1/transactions", headers=headers, params={"limit": PAGE_SIZE, "skip": PAGE_SIZE, "sort": "date"}
    )
    first_ids = [item["id"] for item in first.json()["items"]]
    second_ids = [item["id"] for item in second.json()["items"]]
    assert first_ids and second_ids
    assert set(first_ids).isdisjoint(second_ids), "skip did not advance the window"


def test_page_and_page_size_still_work(client: TestClient, headers: dict[str, str]) -> None:
    """The page-based style keeps working alongside the offset style."""
    _seed(client, headers, 10)

    page_two = client.get(
        "/api/v1/transactions", headers=headers, params={"page": 2, "page_size": PAGE_SIZE, "sort": "date"}
    )
    assert page_two.status_code == 200
    assert len(page_two.json()["items"]) == PAGE_SIZE
    assert page_two.json()["meta"]["page"] == 2
    assert page_two.json()["meta"]["has_prev"] is True

    first = client.get(
        "/api/v1/transactions", headers=headers, params={"page": 1, "page_size": PAGE_SIZE, "sort": "date"}
    )
    assert {item["id"] for item in first.json()["items"]}.isdisjoint(
        {item["id"] for item in page_two.json()["items"]}
    )


def test_meta_reports_offset_style(client: TestClient, headers: dict[str, str]) -> None:
    _seed(client, headers, 10)
    meta = client.get("/api/v1/transactions", headers=headers, params={"skip": 3, "limit": 4}).json()["meta"]
    assert meta["skip"] == 3
    assert meta["limit"] == 4
    assert meta["total"] == 10
    assert meta["has_next"] is True
    assert meta["has_prev"] is True


def test_out_of_range_pagination_is_rejected(client: TestClient, headers: dict[str, str]) -> None:
    """Non-numeric and out-of-bounds values must 422, not silently fall back."""
    assert client.get("/api/v1/transactions", headers=headers, params={"limit": "abc"}).status_code == 422
    assert client.get("/api/v1/transactions", headers=headers, params={"skip": "abc"}).status_code == 422
    assert client.get("/api/v1/transactions", headers=headers, params={"skip": -1}).status_code == 422
    assert client.get("/api/v1/transactions", headers=headers, params={"limit": 0}).status_code == 422
    assert client.get("/api/v1/transactions", headers=headers, params={"limit": 5000}).status_code == 422
    assert client.get("/api/v1/transactions", headers=headers, params={"page": 0}).status_code == 422


def test_unmatched_route_uses_error_envelope(client: TestClient) -> None:
    """Unknown paths must return the documented {code, message} shape."""
    response = client.get("/api/v1/does-not-exist")
    assert response.status_code == 404
    body = response.json()
    assert body["code"] == "not_found"
    assert isinstance(body["message"], str) and body["message"]


def test_wrong_method_uses_error_envelope(client: TestClient, headers: dict[str, str]) -> None:
    response = client.delete("/api/v1/dashboard", headers=headers)
    assert response.status_code == 405
    assert response.json()["code"] == "method_not_allowed"


def test_demo_account_can_sign_in(client: TestClient) -> None:
    """The configured demo credentials must work through POST /auth/login.

    `app.demo_seed` creates the account through the service layer, which is not
    subject to the `UserCreate` password policy, so `DEMO_PASSWORD` does not have
    to satisfy it. What has to hold is that those credentials actually sign in.
    """
    from app.api.deps import Services
    from app.db.client import get_db

    Services(get_db()).auth_service().register(
        {"name": "Demo", "email": settings.demo_email, "password": settings.demo_password}
    )

    response = client.post(
        "/api/v1/auth/login",
        json={"email": settings.demo_email, "password": settings.demo_password},
    )
    assert response.status_code == 200, response.text
    assert response.json()["user"]["email"] == settings.demo_email


@pytest.mark.parametrize("weak", ["demopassword", "12345678", "abcdefgh"])
def test_weak_passwords_are_still_rejected(client: TestClient, weak: str) -> None:
    response = client.post(
        "/api/v1/auth/register",
        json={"name": "Weak", "email": f"{weak}@example.com", "password": weak},
    )
    assert response.status_code == 422

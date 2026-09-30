"""Budgets, savings goals, alerts and the dashboard aggregate."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.core.time_utils import strip_tz, utc_now
from tests.conftest import category_id


def test_budget_crud_and_progress(client: TestClient, headers: dict[str, str]) -> None:
    groceries = category_id(client, headers)
    created = client.post(
        "/api/v1/budgets",
        headers=headers,
        json={"limit_amount": 500.0, "period": "monthly", "category_id": groceries, "name": "Groceries"},
    )
    assert created.status_code == 201
    body = created.json()
    assert body["limit_amount"] == 500.0
    assert body["period_key"]
    budget_id = body["id"]

    listed = client.get("/api/v1/budgets", headers=headers)
    assert listed.status_code == 200
    assert listed.json()["meta"]["total"] == 1

    progress = client.get(f"/api/v1/budgets/{budget_id}/progress", headers=headers)
    assert progress.status_code == 200
    assert progress.json()["limit_amount"] == 500.0

    updated = client.patch(f"/api/v1/budgets/{budget_id}", headers=headers, json={"limit_amount": 600.0})
    assert updated.status_code == 200
    assert updated.json()["limit_amount"] == 600.0

    assert client.delete(f"/api/v1/budgets/{budget_id}", headers=headers).status_code == 204
    assert client.get(f"/api/v1/budgets/{budget_id}", headers=headers).status_code == 404


def test_budget_progress_reflects_spend(client: TestClient, headers: dict[str, str]) -> None:
    groceries = category_id(client, headers)
    budget = client.post(
        "/api/v1/budgets",
        headers=headers,
        json={"limit_amount": 200.0, "period": "monthly", "category_id": groceries},
    ).json()

    client.post(
        "/api/v1/transactions", headers=headers, json={"amount": 50.0, "type": "expense", "category_id": groceries}
    )
    progress = client.get(f"/api/v1/budgets/{budget['id']}/progress", headers=headers).json()
    assert progress["spent_amount"] == 50.0
    assert progress["remaining_amount"] == 150.0
    assert progress["is_exceeded"] is False


def test_budget_created_mid_period_counts_the_whole_period(
    client: TestClient, headers: dict[str, str]
) -> None:
    """A budget created on the 26th must still account for spend from the 1st.

    Regression: the window used to be clipped to `start_date`, so `spent_amount`
    ignored earlier spend in the current month and no alert ever fired.
    """
    groceries = category_id(client, headers)
    now = strip_tz(utc_now())
    earlier_this_month = now.replace(day=1, hour=10, minute=0) if now.day > 1 else None
    if earlier_this_month is None:  # pragma: no cover - only on the 1st of a month
        pytest.skip("cannot test intra-period spend on the first day of the month")

    budget = client.post(
        "/api/v1/budgets",
        headers=headers,
        json={"limit_amount": 100.0, "period": "monthly", "category_id": groceries},
    ).json()

    spent_before_budget = client.post(
        "/api/v1/transactions",
        headers=headers,
        json={
            "amount": 120.0,
            "type": "expense",
            "category_id": groceries,
            "date": earlier_this_month.isoformat(),
        },
    )
    assert spent_before_budget.status_code == 201

    progress = client.get(f"/api/v1/budgets/{budget['id']}/progress", headers=headers).json()
    assert progress["spent_amount"] == 120.0
    assert progress["remaining_amount"] == 0.0
    assert progress["is_exceeded"] is True

    alerts = client.get("/api/v1/alerts", headers=headers).json()["items"]
    assert [a["type"] for a in alerts] == ["budget_exceeded"]


def test_goal_lifecycle_with_contributions(client: TestClient, headers: dict[str, str]) -> None:
    created = client.post(
        "/api/v1/goals",
        headers=headers,
        json={"name": "Emergency fund", "target_amount": 1000.0, "target_date": "2027-06-30"},
    )
    assert created.status_code == 201
    goal = created.json()
    assert goal["current_amount"] == 0.0
    assert goal["progress_pct"] == 0.0

    contributed = client.post(f"/api/v1/goals/{goal['id']}/contribute", headers=headers, json={"amount": 250.0})
    assert contributed.status_code == 200
    receipt = contributed.json()
    assert receipt["new_balance"] == 250.0
    assert receipt["goal_achieved"] is False
    assert receipt["goal_id"] == goal["id"]

    summary = client.get("/api/v1/goals/summary", headers=headers)
    assert summary.status_code == 200
    assert summary.json()["active_goals"] == 1
    assert summary.json()["total_saved"] == 250.0

    listed = client.get("/api/v1/goals", headers=headers)
    assert listed.json()["meta"]["total"] == 1

    assert client.delete(f"/api/v1/goals/{goal['id']}", headers=headers).status_code == 204


def test_goal_target_reached_updates_status(client: TestClient, headers: dict[str, str]) -> None:
    goal = client.post("/api/v1/goals", headers=headers, json={"name": "Laptop", "target_amount": 500.0}).json()
    client.post(f"/api/v1/goals/{goal['id']}/contribute", headers=headers, json={"amount": 500.0})
    refreshed = client.get(f"/api/v1/goals/{goal['id']}", headers=headers).json()
    assert refreshed["status"] == "achieved"
    assert refreshed["progress_pct"] == 100.0


def test_alert_counts_and_mark_read(client: TestClient, headers: dict[str, str]) -> None:
    groceries = category_id(client, headers)
    client.post(
        "/api/v1/budgets",
        headers=headers,
        json={"limit_amount": 10.0, "period": "monthly", "category_id": groceries},
    )
    client.post(
        "/api/v1/transactions", headers=headers, json={"amount": 99.0, "type": "expense", "category_id": groceries}
    )

    listed = client.get("/api/v1/alerts", headers=headers)
    assert listed.status_code == 200
    items = listed.json()["items"]
    assert items
    assert all(item["is_read"] is False for item in items)

    counts = client.get("/api/v1/alerts/counts", headers=headers)
    assert counts.status_code == 200
    assert counts.json()["unread"] == len(items)

    read = client.patch("/api/v1/alerts/read", headers=headers, json={"all": True})
    assert read.status_code == 200
    assert read.json()["matched"] == len(items)
    assert read.json()["unread_remaining"] == 0

    counts_after = client.get("/api/v1/alerts/counts", headers=headers).json()
    assert counts_after["unread"] == 0


def test_alert_dedupe_and_delete(client: TestClient, headers: dict[str, str]) -> None:
    groceries = category_id(client, headers)
    client.post(
        "/api/v1/budgets",
        headers=headers,
        json={"limit_amount": 10.0, "period": "monthly", "category_id": groceries},
    )
    for _ in range(3):
        client.post(
            "/api/v1/transactions", headers=headers, json={"amount": 50.0, "type": "expense", "category_id": groceries}
        )

    items = client.get("/api/v1/alerts", headers=headers).json()["items"]
    exceeded = [item for item in items if item["type"] == "budget_exceeded"]
    assert len(exceeded) == 1

    assert client.delete(f"/api/v1/alerts/{exceeded[0]['id']}", headers=headers).status_code == 204
    remaining = client.get("/api/v1/alerts", headers=headers).json()["items"]
    assert all(item["id"] != exceeded[0]["id"] for item in remaining)


def test_dashboard_aggregate(client: TestClient, headers: dict[str, str]) -> None:
    groceries = category_id(client, headers)
    salary = category_id(client, headers, "Salary", "income")
    client.post(
        "/api/v1/budgets",
        headers=headers,
        json={"limit_amount": 400.0, "period": "monthly", "category_id": groceries},
    )
    client.post(
        "/api/v1/transactions", headers=headers, json={"amount": 3000.0, "type": "income", "category_id": salary}
    )
    client.post(
        "/api/v1/transactions", headers=headers, json={"amount": 120.0, "type": "expense", "category_id": groceries}
    )
    client.post("/api/v1/goals", headers=headers, json={"name": "Holiday", "target_amount": 800.0})

    response = client.get("/api/v1/dashboard", headers=headers)
    assert response.status_code == 200
    body = response.json()

    assert body["currency"]
    assert body["period_key"]
    assert body["cards"]["income"]["value"] == 3000.0
    assert body["cards"]["expense"]["value"] == 120.0
    assert body["cards"]["net"]["value"] == 2880.0
    assert body["cards"]["savings_rate"]["value"] > 0
    assert body["cards"]["income"]["label"]
    assert body["cash_flow"]
    assert body["budgets"]
    assert body["budgets"][0]["limit_amount"] == 400.0
    assert body["recent_transactions"]
    assert body["recent_transactions"][0]["category_id"]
    assert body["top_categories"]
    assert body["goals_summary"]["active_goals"] == 1
    assert isinstance(body["insights"], list)


def test_dashboard_is_empty_for_new_user(client: TestClient, headers: dict[str, str]) -> None:
    body = client.get("/api/v1/dashboard", headers=headers).json()
    assert body["cards"]["income"]["value"] == 0.0
    assert body["cards"]["expense"]["value"] == 0.0
    assert body["recent_transactions"] == []
    assert body["predictions"] == []

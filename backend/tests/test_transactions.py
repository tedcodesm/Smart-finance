"""Category CRUD and the transaction -> budget -> alert write path."""

from __future__ import annotations

from fastapi.testclient import TestClient

from tests.conftest import category_id


def test_system_categories_seeded(client: TestClient, headers: dict[str, str]) -> None:
    response = client.get("/api/v1/categories", headers=headers, params={"type": "expense"})
    assert response.status_code == 200
    names = {item["name"] for item in response.json()["items"]}
    assert {"Groceries", "Rent"} <= names or "Groceries" in names


def test_create_update_delete_category(client: TestClient, headers: dict[str, str]) -> None:
    created = client.post(
        "/api/v1/categories",
        headers=headers,
        json={"name": "Coffee", "type": "expense", "color": "#f59e0b", "monthly_typical_amount": 42.0},
    )
    assert created.status_code == 201
    cat_id = created.json()["id"]

    duplicate = client.post("/api/v1/categories", headers=headers, json={"name": "Coffee", "type": "expense"})
    assert duplicate.status_code == 409

    updated = client.patch(f"/api/v1/categories/{cat_id}", headers=headers, json={"color": "#111111"})
    assert updated.status_code == 200
    assert updated.json()["color"] == "#111111"

    fetched = client.get(f"/api/v1/categories/{cat_id}", headers=headers)
    assert fetched.json()["name"] == "Coffee"

    assert client.delete(f"/api/v1/categories/{cat_id}", headers=headers).status_code == 204
    assert client.get(f"/api/v1/categories/{cat_id}", headers=headers).status_code == 404


def test_create_transaction_and_list(client: TestClient, headers: dict[str, str]) -> None:
    groceries = category_id(client, headers)
    created = client.post(
        "/api/v1/transactions",
        headers=headers,
        json={"amount": 25.5, "type": "expense", "category_id": groceries, "description": "Weekly shop"},
    )
    assert created.status_code == 201
    body = created.json()
    assert body["signed_amount"] == -25.5
    assert body["category_name"] == "Groceries"
    assert body["period_key"]

    listed = client.get("/api/v1/transactions", headers=headers, params={"type": "expense"})
    assert listed.status_code == 200
    assert listed.json()["meta"]["total"] == 1

    summary = client.get("/api/v1/transactions/summary", headers=headers)
    assert summary.status_code == 200
    assert summary.json()["expense"] == 25.5


def test_income_transaction_has_positive_signed_amount(client: TestClient, headers: dict[str, str]) -> None:
    salary = category_id(client, headers, "Salary", "income")
    created = client.post(
        "/api/v1/transactions", headers=headers, json={"amount": 3000, "type": "income", "category_id": salary}
    )
    assert created.status_code == 201
    assert created.json()["signed_amount"] == 3000.0


def test_bulk_create_transactions(client: TestClient, headers: dict[str, str]) -> None:
    groceries = category_id(client, headers)
    response = client.post(
        "/api/v1/transactions/bulk",
        headers=headers,
        json={
            "transactions": [
                {"amount": amount, "type": "expense", "category_id": groceries}
                for amount in (10.0, 20.0, 30.0)
            ]
        },
    )
    assert response.status_code == 201
    assert len(response.json()["items"]) == 3


def test_transaction_write_triggers_budget_alert(client: TestClient, headers: dict[str, str]) -> None:
    """A transaction that breaches the budget must raise a deduped alert."""
    groceries = category_id(client, headers)
    budget = client.post(
        "/api/v1/budgets",
        headers=headers,
        json={"limit_amount": 100.0, "period": "monthly", "category_id": groceries, "name": "Grocery budget"},
    )
    assert budget.status_code == 201

    client.post(
        "/api/v1/transactions",
        headers=headers,
        json={"amount": 130.0, "type": "expense", "category_id": groceries},
    )
    alerts = client.get("/api/v1/alerts", headers=headers).json()
    assert alerts["meta"]["total"] >= 1
    types = {item["type"] for item in alerts["items"]}
    assert "budget_exceeded" in types

    # Idempotent: a second identical write must not duplicate the alert.
    client.post(
        "/api/v1/transactions",
        headers=headers,
        json={"amount": 5.0, "type": "expense", "category_id": groceries},
    )
    alerts_after = client.get("/api/v1/alerts", headers=headers).json()
    assert alerts_after["meta"]["total"] == alerts["meta"]["total"]


def test_update_and_delete_transaction(client: TestClient, headers: dict[str, str]) -> None:
    groceries = category_id(client, headers)
    created = client.post(
        "/api/v1/transactions", headers=headers, json={"amount": 15.0, "type": "expense", "category_id": groceries}
    )
    txn_id = created.json()["id"]

    updated = client.patch(f"/api/v1/transactions/{txn_id}", headers=headers, json={"amount": 20.0, "description": "Updated"})
    assert updated.status_code == 200
    assert updated.json()["amount"] == 20.0
    assert updated.json()["description"] == "Updated"

    assert client.delete(f"/api/v1/transactions/{txn_id}", headers=headers).status_code == 204
    assert client.get(f"/api/v1/transactions/{txn_id}", headers=headers).status_code == 404


def test_export_transactions_csv_and_pdf(client: TestClient, headers: dict[str, str]) -> None:
    groceries = category_id(client, headers)
    client.post("/api/v1/transactions", headers=headers, json={"amount": 12.0, "type": "expense", "category_id": groceries})

    csv_response = client.get("/api/v1/transactions/export", headers=headers, params={"format": "csv"})
    assert csv_response.status_code == 200
    assert "text/csv" in csv_response.headers["content-type"]
    assert "Groceries" in csv_response.text

    pdf_response = client.get("/api/v1/transactions/export", headers=headers, params={"format": "pdf"})
    assert pdf_response.status_code == 200
    assert pdf_response.content[:4] == b"%PDF"


def test_transaction_requires_ownership(client: TestClient) -> None:
    """A user cannot read another user's transaction."""
    from tests.conftest import register

    first_headers = {"Authorization": f"Bearer {register(client)['tokens']['access_token']}"}
    groceries = category_id(client, first_headers)
    created = client.post(
        "/api/v1/transactions", headers=first_headers, json={"amount": 9.0, "type": "expense", "category_id": groceries}
    )
    txn_id = created.json()["id"]

    second = register(client, email="bob@example.com")
    second_headers = {"Authorization": f"Bearer {second['tokens']['access_token']}"}
    assert client.get(f"/api/v1/transactions/{txn_id}", headers=second_headers).status_code == 404

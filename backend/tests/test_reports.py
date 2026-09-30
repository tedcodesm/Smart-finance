"""Report generation, persistence and export."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from tests.conftest import category_id


def _seed(client: TestClient, headers: dict[str, str]) -> None:
    groceries = category_id(client, headers)
    salary = category_id(client, headers, "Salary", "income")
    client.post(
        "/api/v1/budgets",
        headers=headers,
        json={"limit_amount": 300.0, "period": "monthly", "category_id": groceries},
    )
    client.post("/api/v1/transactions", headers=headers, json={"amount": 2500.0, "type": "income", "category_id": salary})
    client.post("/api/v1/transactions", headers=headers, json={"amount": 120.0, "type": "expense", "category_id": groceries})
    client.post("/api/v1/transactions", headers=headers, json={"amount": 60.0, "type": "expense", "category_id": groceries})


def test_generate_report_persists_and_lists(client: TestClient, headers: dict[str, str]) -> None:
    _seed(client, headers)
    generated = client.post(
        "/api/v1/reports",
        headers=headers,
        json={"report_type": "income_vs_expense", "period": "monthly"},
    )
    assert generated.status_code == 201
    report = generated.json()
    assert report["report_type"] == "income_vs_expense"
    assert report["data"]["totals"]["expense"] == 180.0
    assert report["summary"]["expense"] == 180.0

    listed = client.get("/api/v1/reports", headers=headers)
    assert listed.status_code == 200
    assert listed.json()["meta"]["total"] == 1

    fetched = client.get(f"/api/v1/reports/{report['id']}", headers=headers)
    assert fetched.status_code == 200
    assert fetched.json()["id"] == report["id"]


def test_generate_endpoint_is_ephemeral(client: TestClient, headers: dict[str, str]) -> None:
    _seed(client, headers)
    generated = client.post(
        "/api/v1/reports/generate",
        headers=headers,
        json={"report_type": "income_vs_expense", "period": "monthly"},
    )
    assert generated.status_code == 200
    assert generated.json()["totals"]["expense"] == 180.0
    assert generated.json()["timeline"]
    assert client.get("/api/v1/reports", headers=headers).json()["meta"]["total"] == 0


def test_budget_report_shape(client: TestClient, headers: dict[str, str]) -> None:
    _seed(client, headers)
    body = client.post(
        "/api/v1/reports",
        headers=headers,
        json={"report_type": "budget_performance", "period": "monthly"},
    ).json()
    assert body["data"]["budgets"]
    assert body["data"]["budgets"][0]["limit_amount"] == 300.0


def test_report_exports(client: TestClient, headers: dict[str, str]) -> None:
    _seed(client, headers)
    report_id = client.post(
        "/api/v1/reports",
        headers=headers,
        json={"report_type": "income_vs_expense", "period": "monthly"},
    ).json()["id"]

    as_json = client.get(f"/api/v1/reports/{report_id}/export", headers=headers, params={"format": "json"})
    assert as_json.status_code == 200
    assert as_json.json()["report_type"] == "income_vs_expense"

    as_csv = client.get(f"/api/v1/reports/{report_id}/export", headers=headers, params={"format": "csv"})
    assert as_csv.status_code == 200
    assert "text/csv" in as_csv.headers["content-type"]
    assert as_csv.text.strip()

    as_pdf = client.get(f"/api/v1/reports/{report_id}/export", headers=headers, params={"format": "pdf"})
    assert as_pdf.status_code == 200
    assert as_pdf.content[:4] == b"%PDF"


def test_delete_report(client: TestClient, headers: dict[str, str]) -> None:
    _seed(client, headers)
    report_id = client.post(
        "/api/v1/reports",
        headers=headers,
        json={"report_type": "spending_by_category", "period": "monthly"},
    ).json()["id"]
    assert client.delete(f"/api/v1/reports/{report_id}", headers=headers).status_code == 204
    assert client.get(f"/api/v1/reports/{report_id}", headers=headers).status_code == 404


def test_report_is_scoped_to_owner(client: TestClient) -> None:
    from tests.conftest import register

    first = register(client)
    first_headers = {"Authorization": f"Bearer {first['tokens']['access_token']}"}
    _seed(client, first_headers)
    report_id = client.post(
        "/api/v1/reports",
        headers=first_headers,
        json={"report_type": "income_vs_expense", "period": "monthly"},
    ).json()["id"]

    second = register(client, email="carol@example.com")
    second_headers = {"Authorization": f"Bearer {second['tokens']['access_token']}"}
    assert client.get(f"/api/v1/reports/{report_id}", headers=second_headers).status_code == 404
    assert client.get("/api/v1/reports", headers=second_headers).json()["meta"]["total"] == 0


def test_report_json_export_is_valid_json(client: TestClient, headers: dict[str, str]) -> None:
    _seed(client, headers)
    report_id = client.post(
        "/api/v1/reports",
        headers=headers,
        json={"report_type": "income_vs_expense", "period": "monthly"},
    ).json()["id"]
    payload = client.get(f"/api/v1/reports/{report_id}/export", headers=headers, params={"format": "json"})
    assert json.loads(payload.content)["period"]



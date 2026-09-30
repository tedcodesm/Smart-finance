"""Predictions + ML bridge, health checks, and the scheduler jobs.

ML inference is stubbed at the `mercy_ml.infer.run_inference` boundary: the tests
assert the API/DB wiring (persisted prediction documents, model_runs bookkeeping,
idempotent on-demand runs) without paying for a real model fit.
"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from typing import Any

import pytest
from bson import ObjectId
from fastapi.testclient import TestClient

from app.core.time_utils import strip_tz, utc_now
from tests.conftest import category_id


def _prediction_doc(
    user_id: Any,
    *,
    prediction_type: str = "expense_forecast",
    value: float = 42.0,
    generated_at: Any = None,
    period_key: str = "2026-09",
) -> dict[str, Any]:
    """A `predictions` document in the shape `mercy_ml.infer` writes."""
    now = generated_at or strip_tz(utc_now())
    return {
        "_id": ObjectId(),
        "user_id": user_id,
        "type": prediction_type,
        "scope": {"category_id": None, "period_key": period_key},
        "value": value,
        "confidence_score": 0.8,
        "model_type": "linear_regression",
        "model_version": "test-1",
        "model_run_id": "run-test",
        "generated_at": now,
        "expires_at": now + timedelta(days=30),
        "features": {},
    }


@pytest.fixture()
def stub_inference(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Replace `run_inference` and record the calls made through the API."""
    calls: list[dict[str, Any]] = []

    def fake_run_inference(*, user_ids: list[str] | None = None, as_of: Any = None, **_kwargs: Any) -> dict[str, Any]:
        calls.append({"user_ids": user_ids, "as_of": as_of})
        return {
            "run_id": f"run-{len(calls)}",
            "users_scored": len(user_ids) if user_ids else 0,
            "predictions_written": 3,
            "errors": [],
        }

    monkeypatch.setattr("mercy_ml.infer.run_inference", fake_run_inference)
    return calls


def test_predictions_empty_for_new_user(client: TestClient, headers: dict[str, str]) -> None:
    listed = client.get("/api/v1/predictions", headers=headers)
    assert listed.status_code == 200
    assert listed.json()["meta"]["total"] == 0

    latest = client.get("/api/v1/predictions/latest", headers=headers, params={"type": "expense_forecast"})
    assert latest.status_code == 200
    assert latest.json() == []


def test_predictions_listing_and_filtering(
    client: TestClient, headers: dict[str, str], db: Any
) -> None:
    from app.core.constants import Collections

    reg = client.post(
        "/api/v1/auth/register",
        json={"name": "Dana", "email": "dana@example.com", "password": "analytical1"},
    )
    token = reg.json()["tokens"]["access_token"]
    user_id = reg.json()["user"]["id"]
    auth = {"Authorization": f"Bearer {token}"}
    oid = ObjectId(user_id)

    now = strip_tz(utc_now())
    db[Collections.PREDICTIONS].insert_many(
        [
            _prediction_doc(oid, value=10.0, generated_at=now - timedelta(days=2)),
            _prediction_doc(oid, value=20.0, generated_at=now - timedelta(days=1)),
            _prediction_doc(oid, prediction_type="overspend_risk", value=0.4, generated_at=now),
        ]
    )

    listed = client.get("/api/v1/predictions", headers=auth)
    assert listed.status_code == 200
    body = listed.json()
    assert body["meta"]["total"] == 3
    # Newest first.
    assert body["items"][0]["type"] == "overspend_risk"

    forecasts = client.get("/api/v1/predictions", headers=auth, params={"type": "expense_forecast"})
    assert forecasts.json()["meta"]["total"] == 2

    latest = client.get("/api/v1/predictions/latest", headers=auth, params={"type": "expense_forecast"})
    assert latest.status_code == 200
    assert len(latest.json()) == 1
    summary = latest.json()[0]
    assert summary["type"] == "expense_forecast"
    assert summary["latest_value"] == 20.0
    assert summary["count"] == 1
    assert summary["average_confidence"] == 0.8


def test_predictions_are_scoped_to_owner(client: TestClient, db: Any) -> None:
    from app.core.constants import Collections
    from tests.conftest import register

    first = register(client)
    second = register(client, email="erin@example.com")
    db[Collections.PREDICTIONS].insert_one(_prediction_doc(ObjectId(first["user"]["id"])))

    second_headers = {"Authorization": f"Bearer {second['tokens']['access_token']}"}
    assert client.get("/api/v1/predictions", headers=second_headers).json()["meta"]["total"] == 0


def test_on_demand_run_inference_is_scoped_to_caller(
    client: TestClient, headers: dict[str, str], stub_inference: list[dict[str, Any]]
) -> None:
    response = client.post("/api/v1/predictions/inference/run", headers=headers)
    assert response.status_code == 200
    body = response.json()
    assert body["users_scored"] == 1
    assert len(stub_inference) == 1
    assert len(stub_inference[0]["user_ids"]) == 1


def test_expense_forecast_endpoint(client: TestClient, headers: dict[str, str], db: Any, stub_inference: Any) -> None:
    from app.core.constants import Collections

    user_id = ObjectId(headers and client.get("/api/v1/users/me", headers=headers).json()["id"])
    db[Collections.PREDICTIONS].insert_one(_prediction_doc(user_id, value=321.0))

    response = client.post("/api/v1/predictions/expense-forecast", headers=headers)
    assert response.status_code == 200
    body = response.json()
    assert body["total_predicted"] == 321.0
    assert body["items"]


def test_model_status_reports_registry(client: TestClient, headers: dict[str, str]) -> None:
    response = client.get("/api/v1/predictions/models/status", headers=headers)
    assert response.status_code == 200
    body = response.json()
    assert "models" in body
    assert isinstance(body["models"], list)
    assert "latest_run" in body


def test_budget_recommendation_endpoint(client: TestClient, headers: dict[str, str], db: Any, stub_inference: Any) -> None:
    from app.core.constants import Collections

    user_id = ObjectId(client.get("/api/v1/users/me", headers=headers).json()["id"])
    db[Collections.PREDICTIONS].insert_one(_prediction_doc(user_id, prediction_type="budget_recommendation", value=275.0))

    response = client.post("/api/v1/predictions/budget-recommendations", headers=headers)
    assert response.status_code == 200
    assert response.json()["items"]


def test_overspend_risk_endpoint(client: TestClient, headers: dict[str, str], db: Any, stub_inference: Any) -> None:
    from app.core.constants import Collections

    user_id = ObjectId(client.get("/api/v1/users/me", headers=headers).json()["id"])
    db[Collections.PREDICTIONS].insert_one(_prediction_doc(user_id, prediction_type="overspend_risk", value=0.77))

    response = client.post("/api/v1/predictions/overspend-risk", headers=headers)
    assert response.status_code == 200
    body = response.json()
    assert body["items"]
    assert body["items"][0]["probability"] == 0.77


def test_health_endpoint(client: TestClient) -> None:
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] in {"ok", "degraded"}
    assert "database" in body


def test_root_and_openapi(client: TestClient) -> None:
    root = client.get("/")
    assert root.status_code == 200
    assert root.json()["name"]

    schema = client.get("/openapi.json")
    assert schema.status_code == 200
    paths = schema.json()["paths"]
    for expected in (
        "/api/v1/auth/login",
        "/api/v1/users/me",
        "/api/v1/categories",
        "/api/v1/transactions",
        "/api/v1/budgets",
        "/api/v1/goals",
        "/api/v1/alerts",
        "/api/v1/reports",
        "/api/v1/predictions",
        "/api/v1/dashboard",
        "/api/v1/health",
    ):
        assert expected in paths, expected


def test_scheduler_alert_sweep_and_batch_inference(
    client: TestClient, headers: dict[str, str], stub_inference: list[dict[str, Any]]
) -> None:
    """`alert_sweep_job` and `ml_batch_job` execute end-to-end against mongomock."""
    from app.scheduler import alert_sweep_job, ml_batch_job

    groceries = category_id(client, headers)
    client.post(
        "/api/v1/budgets",
        headers=headers,
        json={"limit_amount": 5.0, "period": "monthly", "category_id": groceries},
    )
    client.post("/api/v1/transactions", headers=headers, json={"amount": 50.0, "type": "expense", "category_id": groceries})

    sweep = alert_sweep_job()
    assert sweep.get("status") != "error", sweep
    assert sweep["users"] == 1
    assert client.get("/api/v1/alerts", headers=headers).json()["meta"]["total"] >= 1

    # Budget checks are idempotent: a second sweep must not add duplicate alerts.
    before = client.get("/api/v1/alerts", headers=headers).json()["meta"]["total"]
    alert_sweep_job()
    assert client.get("/api/v1/alerts", headers=headers).json()["meta"]["total"] == before

    batch = ml_batch_job()
    assert batch.get("status") != "error", batch
    assert stub_inference, "batch inference should delegate to mercy_ml.infer.run_inference"


def test_scheduler_status_reports_disabled(client: TestClient) -> None:
    from app.scheduler import get_scheduler_status, start_scheduler, stop_scheduler

    status = get_scheduler_status()
    assert status["enabled"] is False
    assert status["running"] is False
    assert status["jobs"] == []

    assert start_scheduler() is None
    stop_scheduler()


def test_ml_imports_are_warmed_during_startup(client: TestClient) -> None:
    """Startup must load `mercy_ml` before any request thread can race on it.

    `MLService` imports `mercy_ml.infer` / `mercy_ml.registry` lazily, from
    `def` endpoints that FastAPI runs in anyio worker threads. Concurrent first
    imports deadlock on CPython's per-module import locks (observed as a 500
    from `sklearn.ensemble._forest`), so the lifespan resolves them up front.
    """
    for module in ("mercy_ml.infer", "mercy_ml.registry"):
        assert module in sys.modules, f"{module} was not imported during startup"


def test_concurrent_ml_endpoints_do_not_deadlock(client: TestClient, headers: dict[str, str]) -> None:
    """`/models/status` and `/expense-forecast` firing together must both succeed."""
    with ThreadPoolExecutor(max_workers=2) as pool:
        status_future = pool.submit(client.get, "/api/v1/predictions/models/status")
        forecast_future = pool.submit(
            client.post, "/api/v1/predictions/expense-forecast", headers=headers, json={}
        )

        status = status_future.result(timeout=30)
        forecast = forecast_future.result(timeout=30)

    assert status.status_code == 200, status.text
    assert forecast.status_code in (200, 404), forecast.text



"""The real `mercy_ml.infer.run_inference` path against the API's mongomock DB.

Verifies the client bridge: the API writes transactions, `mercy_ml` reads them
through the shared client, and the resulting `predictions` documents are
readable through the prediction endpoints.
"""

from __future__ import annotations

from datetime import timedelta

from fastapi.testclient import TestClient

from tests.conftest import category_id


def test_end_to_end_inference_through_api(client: TestClient, headers: dict[str, str], db) -> None:
    """Run the real ML inference and read the results back over HTTP."""
    from app.core.time_utils import strip_tz, utc_now

    groceries = category_id(client, headers)
    salary = category_id(client, headers, "Salary", "income")
    user_id = db["users"].find_one({})["_id"]

    now = strip_tz(utc_now())
    rows: list[dict] = []
    for month_back in range(6, 0, -1):
        when = now - timedelta(days=30 * month_back)

        def expense_row(day: int, amount: float) -> dict:
            return {
                "user_id": user_id,
                "amount": amount,
                "type": "expense",
                "category_id": groceries,
                "date": when.replace(day=day),
                "description": "Groceries",
                "notes": None,
                "is_recurring": False,
                "year": when.year,
                "month": when.month,
                "day": day,
                "day_of_week": when.weekday(),
                "day_of_month": day,
                "week_of_year": when.isocalendar()[1],
                "period_key": when.strftime("%Y-%m"),
                "week_key": f"{when.isocalendar()[0]}-W{when.isocalendar()[1]:02d}",
                "created_at": when,
                "updated_at": when,
            }

        for day in range(2, 27, 5):
            rows.append(expense_row(day, 40.0 + day))
        rows.append(
            {
                **expense_row(1, 3000.0),
                "amount": 3000.0,
                "type": "income",
                "category_id": salary,
                "description": "Salary",
                "is_recurring": True,
            }
        )

    db["transactions"].insert_many(rows)

    # The overspend classifier and savings recommender key off budgets/goals.
    budget = client.post(
        "/api/v1/budgets",
        headers=headers,
        json={"limit_amount": 200.0, "period": "monthly", "category_id": groceries, "name": "Groceries"},
    )
    assert budget.status_code == 201
    goal = client.post("/api/v1/goals", headers=headers, json={"name": "Rainy day", "target_amount": 5000.0})
    assert goal.status_code == 201

    run = client.post("/api/v1/predictions/inference/run", headers=headers)
    assert run.status_code == 200, run.text
    body = run.json()
    assert body["users_scored"] == 1
    assert body["predictions_written"] > 0

    listed = client.get("/api/v1/predictions", headers=headers)
    assert listed.json()["meta"]["total"] == body["predictions_written"]

    latest = client.get("/api/v1/predictions/latest", headers=headers)
    assert latest.status_code == 200
    types = {item["type"] for item in latest.json()}
    assert "expense_forecast" in types
    assert "spending_cluster" in types

    forecast = client.post("/api/v1/predictions/expense-forecast", headers=headers)
    assert forecast.status_code == 200, forecast.text
    assert forecast.json()["total_predicted"] > 0
    assert forecast.json()["items"]

    risk = client.post("/api/v1/predictions/overspend-risk", headers=headers)
    assert risk.status_code == 200, risk.text

    recommendations = client.post("/api/v1/predictions/budget-recommendations", headers=headers)
    assert recommendations.status_code == 200, recommendations.text

    savings = client.post("/api/v1/predictions/savings-recommendation", headers=headers)
    assert savings.status_code == 200, savings.text

    clusters = client.get("/api/v1/predictions/clusters", headers=headers)
    assert clusters.status_code == 200, clusters.text

    # The dashboard should now surface the freshly written predictions.
    dashboard = client.get("/api/v1/dashboard", headers=headers)
    assert dashboard.status_code == 200
    assert dashboard.json()["predictions"]

    # A model_runs row records the batch; the on-demand endpoints that follow
    # reuse those fresh predictions instead of re-scoring.
    runs = list(db["model_runs"].find({}))
    assert len(runs) == 1
    assert runs[0]["run_id"] == body["run_id"]

    # Idempotency: a second explicit run replaces, never duplicates.
    again = client.post("/api/v1/predictions/inference/run", headers=headers)
    assert again.status_code == 200
    total_after = client.get("/api/v1/predictions", headers=headers).json()["meta"]["total"]
    assert total_after == body["predictions_written"]
    assert len(list(db["model_runs"].find({}))) == 2

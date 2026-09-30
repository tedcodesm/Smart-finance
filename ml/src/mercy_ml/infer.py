"""Batch inference: score every active user and refresh the `predictions` docs.

Run by the nightly scheduler (and by the on-demand API path for one user). It is
idempotent - each run deletes the previous set of predictions for the target
period/type before writing fresh ones, recording the whole thing in `model_runs`.
"""

from __future__ import annotations

import argparse
import logging
import math
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any, Iterable, Sequence

import pandas as pd

from mercy_ml.config import get_settings
from mercy_ml.data import get_db, load_categories, load_transactions
from mercy_ml.features import (
    PeriodHistory,
    build_budget_recommendation_frame,
    build_cluster_frame,
    build_forecast_inference_row,
    build_overspend_inference_row,
    build_savings_inference_row,
)
from mercy_ml.models import BudgetRecommender, confidence_for_forecast
from mercy_ml.periods import month_key, next_period_key
from mercy_ml.registry import load_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("mercy_ml.infer")

HORIZON_DAYS: dict[str, int] = {
    "expense_forecast": 30,
    "overspend_risk": 1,
    "budget_recommendation": 30,
    "savings_recommendation": 30,
    "spending_cluster": 30,
    "unusual_spending": 7,
}


def _object_id() -> Any:
    from bson import ObjectId

    return ObjectId()


def _bson_safe(value: Any) -> Any:
    """Recursively convert numpy/NaN values to MongoDB-encodable JSON types."""

    def scan(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {str(k): scan(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [scan(v) for v in obj]
        if obj is None or isinstance(obj, (bool, int, str)):
            return obj
        if isinstance(obj, float):
            return obj if math.isfinite(obj) else None
        cls = type(obj)
        module = getattr(cls, "__module__", "")
        if module == "numpy":
            scalar = obj.item() if hasattr(obj, "item") else obj
            if isinstance(scalar, (list, tuple)):
                return [scan(v) for v in scalar]
            return scan(scalar)
        return obj

    return scan(value)


def _insert_predictions(collection: Any, doc: dict[str, Any]) -> None:
    collection.insert_one(_bson_safe(doc))


def run_inference(
    *,
    user_ids: Iterable[Any] | None = None,
    as_of: date | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Score all (or the given) users and upsert predictions. Returns a summary."""
    from bson import ObjectId

    settings = get_settings()
    as_of = as_of or date.today()
    current = month_key(as_of)
    upcoming = next_period_key("monthly", current)
    db = get_db()
    run_id = str(_object_id())
    started = time.time()

    forecaster = load_model("expense_forecaster")
    classifier = load_model("overspend_classifier")
    clusterer = load_model("spending_clusters")
    recommender = load_model("budget_recommender") or BudgetRecommender()
    saver = load_model("savings_recommender")

    if forecaster is None and classifier is None and clusterer is None and saver is None:
        raise RuntimeError("No artifacts found - run mercy-ml-train first.")

    transactions = load_transactions(user_ids=list(user_ids) if user_ids else None)
    categories = load_categories(user_ids=list(user_ids) if user_ids else None)
    names = dict(zip(categories["category_id"], categories["category_name"]))
    category_types = dict(zip(categories["category_id"], categories["category_type"]))
    if transactions.empty:
        return {"run_id": run_id, "users_scored": 0, "predictions_written": 0, "errors": []}

    history = PeriodHistory.build(transactions)
    budget_stats = build_budget_recommendation_frame(transactions, categories)
    users = (
        [uid for uid in user_ids]
        if user_ids
        else sorted(history.period_keys, key=lambda uid: str(uid))
    )

    # Load active budgets once, keyed by user.
    active_budgets: dict[Any, list[dict[str, Any]]] = {}
    for budget in db.budgets.find(
        {"is_active": True, "period": "monthly"}, {"_id": 1, "user_id": 1, "category_id": 1,
                                                    "limit_amount": 1, "name": 1}
    ):
        uid = budget.get("user_id")
        active_budgets.setdefault(uid, []).append(budget)

    # Load active goals once, keyed by user.
    active_goals: dict[Any, list[dict[str, Any]]] = {}
    for goal in db.savings_goals.find(
        {"status": "active"}, {"user_id": 1, "target_amount": 1, "current_amount": 1,
                                "target_date": 1, "name": 1}
    ):
        uid = goal.get("user_id")
        active_goals.setdefault(uid, []).append(goal)

    predictions_written = 0
    users_scored = 0
    errors: list[str] = []

    for user_id in users:
        user_txns = transactions[transactions["user_id"] == user_id]
        if user_txns.empty:
            continue
        try:
            n = _score_user(
                db=db,
                user_id=user_id,
                txns=user_txns,
                history=history,
                budget_stats=budget_stats,
                names=names,
                category_types=category_types,
                budgets=active_budgets.get(user_id, []),
                goals=active_goals.get(user_id, []),
                forecaster=forecaster,
                classifier=classifier,
                clusterer=clusterer,
                recommender=recommender,
                saver=saver,
                current=current,
                upcoming=upcoming,
                as_of=as_of,
                run_id=run_id,
            )
            predictions_written += n
            users_scored += 1
        except Exception as exc:  # pragma: no cover - one bad user must not kill the run
            errors.append(f"{user_id}: {exc}")
            logger.exception("Skipping user %s", user_id)

    db["model_runs"].insert_one(
        {
            "status": "success" if not errors else "partial",
            "run_type": "inference",
            "run_id": run_id,
            "started_at": datetime.now(timezone.utc) - timedelta(seconds=max(0, time.time() - started)),
            "finished_at": datetime.now(timezone.utc),
            "as_of": str(as_of),
            "period_key": current,
            "users_scored": users_scored,
            "predictions_written": predictions_written,
            "errors": errors,
            "models": [m for m in ("expense_forecaster", "overspend_classifier",
                                   "spending_clusters", "budget_recommender",
                                   "savings_recommender")],
            "created_at": datetime.now(timezone.utc),
        }
    )

    summary = {
        "run_id": run_id,
        "as_of": str(as_of),
        "users_scored": users_scored,
        "predictions_written": predictions_written,
        "errors": errors,
        "duration_seconds": round(time.time() - started, 2),
    }
    if verbose:
        logger.info(summary)
    return summary


def _score_user(
    *,
    db: Any,
    user_id: Any,
    txns: pd.DataFrame,
    history: PeriodHistory,
    budget_stats: pd.DataFrame,
    names: dict[Any, str],
    category_types: dict[Any, str],
    budgets: list[dict[str, Any]],
    goals: list[dict[str, Any]],
    forecaster: Any,
    classifier: Any,
    clusterer: Any,
    recommender: Any,
    saver: Any,
    current: str,
    upcoming: str,
    as_of: date,
    run_id: str,
) -> int:
    """Score one user and return the number of prediction documents written."""
    written = 0
    predictions = db["predictions"]
    user_txns = txns

    # --------------------------------------------- expense forecasts (next month)
    if forecaster is not None:
        prior = history.keys_before(user_id, upcoming)
        forecast_history = forecaster.report.metrics if forecaster.report else {}
        category_keys = sorted(
            {category for (uid, category, _key) in history.category_period_expense if uid == user_id},
            key=lambda cid: str(cid),
        )
        for category_id in category_keys:
            if category_types.get(category_id) == "income":
                continue
            name = names.get(category_id, "Unknown")
            feature_row = build_forecast_inference_row(
                user_id=user_id,
                category_id=category_id,
                category_name=name,
                period_key=current,
                transactions=user_txns,
                categories=pd.DataFrame(columns=["category_id", "category_name", "category_type", "is_system"]),
            )
            try:
                result = forecaster.predict_with_interval(pd.DataFrame([feature_row]))
            except Exception as exc:  # pragma: no cover
                logger.warning("forecast failed for %s/%s: %s", user_id, category_id, exc)
                continue
            predictions.delete_many(
                {"user_id": user_id, "type": "expense_forecast",
                 "scope.category_id": str(category_id), "scope.period_key": upcoming}
            )
            _insert_predictions(predictions,
                {
                    "user_id": user_id,
                    "type": "expense_forecast",
                    "value": result["predicted_amount"],
                    "confidence_score": confidence_for_forecast(forecast_history, n_history=len(prior)),
                    "model_version": forecaster.report.model_version if forecaster.report else forecaster.name,
                    "model_run_id": run_id,
                    "scope": {
                        "category_id": str(category_id),
                        "category_name": name,
                        "period_key": upcoming,
                        "period": "monthly",
                    },
                    "label": None,
                    "features": {k: v for k, v in feature_row.items() if k != "user_id"},
                    "extra": {"lower_bound": result["lower_bound"], "upper_bound": result["upper_bound"]},
                    "horizon_days": HORIZON_DAYS["expense_forecast"],
                    "generated_at": datetime.now(timezone.utc),
                    "expires_at": datetime.now(timezone.utc) + timedelta(days=HORIZON_DAYS["expense_forecast"]),
                }
            )
            written += 1

    # ---------------------------------------------- overspend risk (active budgets)
    if classifier is not None:
        for budget in budgets:
            category_id = budget.get("category_id")
            limit = float(budget.get("limit_amount") or 0.0)
            if limit <= 0:
                continue

            category_name = "Overall" if category_id is None else names.get(category_id, "Unknown")
            spent = 0.0
            if category_id is None:
                spent = float(
                    user_txns[(user_txns["type"] == "expense")
                              & (user_txns["period_key"] == current)]["amount"].sum()
                )
            else:
                spent = float(
                    user_txns[(user_txns["type"] == "expense")
                              & (user_txns["category_id"] == category_id)
                              & (user_txns["period_key"] == current)]["amount"].sum()
                )
            feature_row = build_overspend_inference_row(
                user_id=user_id,
                category_id=category_id,
                category_name=category_name,
                limit_amount=limit,
                spent_so_far=spent,
                as_of=as_of,
                period="monthly",
                period_key=current,
                transactions=user_txns,
                categories=budget_stats.iloc[0:0],
            )
            result = classifier.predict_risk(pd.DataFrame([feature_row]))

            predictions.delete_many(
                {"user_id": user_id, "type": "overspend_risk",
                 "scope.budget_id": str(budget["_id"])}
            )
            _insert_predictions(predictions,
                {
                    "user_id": user_id,
                    "type": "overspend_risk",
                    "value": result["probability"],
                    "confidence_score": result["confidence_score"],
                    "model_version": classifier.report.model_version if classifier.report else classifier.name,
                    "model_run_id": run_id,
                    "scope": {
                        "budget_id": str(budget["_id"]),
                        "category_id": str(category_id) if category_id is not None else None,
                        "category_name": category_name,
                        "period_key": current,
                        "period": "monthly",
                    },
                    "label": result["label"],
                    "features": {k: v for k, v in feature_row.items() if k != "user_id"},
                    "extra": {"threshold": result["threshold"]},
                    "horizon_days": HORIZON_DAYS["overspend_risk"],
                    "generated_at": datetime.now(timezone.utc),
                    "expires_at": datetime.now(timezone.utc) + timedelta(days=HORIZON_DAYS["overspend_risk"]),
                }
            )
            written += 1

    # --------------------------------------------- budget recommendations / user
    if recommender is not None:
        user_stats = budget_stats[budget_stats["user_id"].astype(str) == str(user_id)]
        for _, stat_row in user_stats.iterrows():
            suggestion = recommender.recommend(stat_row)
            predictions.delete_many(
                {"user_id": user_id, "type": "budget_recommendation",
                 "scope.category_id": str(stat_row.get("category_id"))}
            )
            _insert_predictions(predictions,
                {
                    "user_id": user_id,
                    "type": "budget_recommendation",
                    "value": suggestion["recommended_limit"],
                    "confidence_score": suggestion["confidence_score"],
                    "model_version": recommender.name,
                    "model_run_id": run_id,
                    "scope": {
                        "category_id": str(stat_row.get("category_id")),
                        "category_name": str(stat_row.get("category_name") or "Overall"),
                        "period": "monthly",
                    },
                    "label": suggestion["action"],
                    "features": {k: v for k, v in suggestion["basis"].items()},
                    "extra": {
                        "delta": suggestion["delta"],
                        "delta_pct": suggestion["delta_pct"],
                        "rationale": suggestion["rationale"],
                        "current_limit": suggestion["current_limit"],
                    },
                    "horizon_days": HORIZON_DAYS["budget_recommendation"],
                    "generated_at": datetime.now(timezone.utc),
                    "expires_at": datetime.now(timezone.utc) + timedelta(days=HORIZON_DAYS["budget_recommendation"]),
                }
            )
            written += 1

    # ------------------------------------------------- savings recommendation / user
    if saver is not None:
        savings_row = pd.DataFrame(
            [
                build_savings_inference_row(
                    user_id=user_id, period_key=current, transactions=user_txns
                )
            ]
        )
        suggested_total = 0.0
        goal_recommendations: list[dict[str, Any]] = []
        for goal in goals[:3]:
            rec = saver.recommend(
                savings_row,
                target_amount=float(goal.get("target_amount") or 0.0),
                current_amount=float(goal.get("current_amount") or 0.0),
                active_goals=max(0, len(goals) - 1),
                months_remaining=_months_until(goal.get("target_date"), as_of),
            )
            suggested_total += rec["suggested_contribution"]
            goal_recommendations.append({**rec, "goal_id": str(goal["_id"]), "name": goal.get("name")})
        predicted_net = saver.forecast_next_savings(savings_row)

        predictions.delete_many(
            {"user_id": user_id, "type": "savings_recommendation", "scope.period_key": current}
        )
        _insert_predictions(predictions,
            {
                "user_id": user_id,
                "type": "savings_recommendation",
                "value": round(predicted_net, 2),
                "confidence_score": (
                    saver.report.metrics.get("mae_skill_vs_naive", 0.5) if saver.report else 0.5
                ),
                "model_version": saver.report.model_version if saver.report else saver.name,
                "model_run_id": run_id,
                "scope": {"period_key": current, "period": "monthly"},
                "label": "surplus" if predicted_net > 0 else "deficit",
                "features": {k: v for k, v in savings_row.iloc[0].items() if k != "user_id"},
                "extra": {
                    "suggested_contribution_total": round(suggested_total, 2),
                    "goals": goal_recommendations,
                    "savings_rate_pct": round(float(savings_row.iloc[0].get("savings_rate", 0)) * 100.0, 2),
                },
                "horizon_days": HORIZON_DAYS["savings_recommendation"],
                "generated_at": datetime.now(timezone.utc),
                "expires_at": datetime.now(timezone.utc) + timedelta(days=HORIZON_DAYS["savings_recommendation"]),
            }
        )
        written += 1

    # --------------------------------------------------- spending clusters / user
    if clusterer is not None:
        cluster_frame = build_cluster_frame(user_txns, budget_stats.iloc[0:0])
        if len(cluster_frame):
            profile = clusterer.user_cluster_profile(cluster_frame, str(user_id))
            predictions.delete_many(
                {"user_id": user_id, "type": "spending_cluster", "scope.period_key": current}
            )
            top = max(profile, key=lambda item: item["count"], default={})
            _insert_predictions(predictions,
                {
                    "user_id": user_id,
                    "type": "spending_cluster",
                    "value": float(top.get("count", 0)),
                    "confidence_score": (
                        clusterer.report.metrics.get("silhouette_score", 0.0) if clusterer.report else 0.0
                    ),
                    "model_version": clusterer.report.model_version if clusterer.report else clusterer.name,
                    "model_run_id": run_id,
                    "scope": {"period_key": current, "period": "monthly"},
                    "label": top.get("label"),
                    "features": None,
                    "extra": {"profile": profile,
                              "n_clusters": clusterer.centroid_frame.shape[0] if clusterer.centroid_frame is not None else 0},
                    "horizon_days": HORIZON_DAYS["spending_cluster"],
                    "generated_at": datetime.now(timezone.utc),
                    "expires_at": datetime.now(timezone.utc) + timedelta(days=HORIZON_DAYS["spending_cluster"]),
                }
            )
            written += 1

    return written


def _months_until(target: Any, as_of: date) -> float | None:
    import pandas as pd

    if target is None:
        return None
    value = pd.to_datetime(target, errors="coerce")
    if pd.isna(value):
        return None
    return max(0.0, (value.to_pydatetime().date() - as_of).days / 30.44)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mercy-ml-infer",
        description="Score active users and refresh persisted predictions.",
    )
    parser.add_argument("--user-ids", default=None, help="Comma-separated ObjectIds (default: all)")
    parser.add_argument("--as-of", default=None, help="YYYY-MM-DD; defaults to today")
    parser.add_argument("--mongomock", action="store_true", help="Run against an in-memory synthetic fixture")
    parser.add_argument("--users", type=int, default=60)
    parser.add_argument("--months", type=int, default=14)
    args = parser.parse_args(argv)

    try:
        if args.mongomock:
            from mercy_ml import synthetic
            from mercy_ml.data import set_client

            client, counts = synthetic.seed_mongomock(n_users=args.users, months=args.months)
            set_client(client)
            logger.info("mongomock fixture: %s", counts)

        user_ids = [uid.strip() for uid in (args.user_ids or "").split(",") if uid.strip()] or None
        as_of = date.fromisoformat(args.as_of) if args.as_of else None
        summary = run_inference(user_ids=user_ids, as_of=as_of, verbose=True)
        import json

        print(json.dumps(summary, indent=2, default=str))
        return 0
    except Exception as exc:  # pragma: no cover
        logger.error("Inference failed: %s", exc)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
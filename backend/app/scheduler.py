"""Nightly jobs driven by APScheduler.

Two scheduled jobs, both idempotent so a missed run or a manual trigger is safe:

- ``ml_batch``  (default 02:17 UTC): score all active users and refresh `predictions`.
- ``alert_sweep`` (default 07:23 UTC): re-evaluate budgets, goal deadlines, purge
  expired alerts/predictions/reports/refresh tokens, and roll budget periods.
"""

from __future__ import annotations

import logging
from typing import Any

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from app.api.deps import Services
from app.core.config import settings
from app.core.constants import BudgetPeriod
from app.core.time_utils import strip_tz, utc_now
from app.db.client import get_db

logger = logging.getLogger("mercy.scheduler")

_scheduler: BackgroundScheduler | None = None


def start_scheduler() -> BackgroundScheduler | None:
    """Start the background scheduler if enabled and not already running."""
    global _scheduler
    if not settings.enable_scheduler:
        logger.info("Scheduler disabled (ENABLE_SCHEDULER=false)")
        return None
    if _scheduler is not None and _scheduler.running:
        return _scheduler

    _scheduler = BackgroundScheduler(
        timezone=settings.scheduler_timezone,
        job_defaults={"coalesce": True, "max_instances": 1, "misfire_grace_time": 3600},
    )
    _scheduler.add_job(
        ml_batch_job,
        CronTrigger(hour=settings.ml_batch_hour, minute=settings.ml_batch_minute),
        id="ml_batch",
        replace_existing=True,
    )
    _scheduler.add_job(
        alert_sweep_job,
        CronTrigger(hour=settings.alert_sweep_hour, minute=settings.alert_sweep_minute),
        id="alert_sweep",
        replace_existing=True,
    )
    _scheduler.start()
    logger.info(
        "Scheduler started (ml_batch=%02d:%02d, alert_sweep=%02d:%02d UTC)",
        settings.ml_batch_hour,
        settings.ml_batch_minute,
        settings.alert_sweep_hour,
        settings.alert_sweep_minute,
    )
    return _scheduler


def stop_scheduler() -> None:
    global _scheduler
    if _scheduler is not None and _scheduler.running:
        _scheduler.shutdown(wait=False)
        logger.info("Scheduler stopped")
    _scheduler = None


def get_scheduler_status() -> dict[str, Any]:
    return {
        "enabled": settings.enable_scheduler,
        "running": bool(_scheduler and _scheduler.running),
        "jobs": [
            {"id": job.id, "next_run_time": str(getattr(job, "next_run_time", None))}
            for job in (_scheduler.jobs if _scheduler else [])
        ],
    }


# --------------------------------------------------------------------------- #
# Jobs
# --------------------------------------------------------------------------- #
def ml_batch_job() -> dict[str, Any]:
    """Run `mercy_ml.infer` for all active users; swallow errors so the job doesn't crash."""
    try:
        services = Services(get_db())
        result = services.ml_service().run_batch_inference()
        logger.info("ML batch complete: %s", result)
        return result
    except Exception as exc:  # pragma: no cover - scheduler safety
        logger.exception("ML batch job failed: %s", exc)
        return {"status": "error", "error": str(exc)}


def alert_sweep_job() -> dict[str, Any]:
    """Re-evaluate budgets + goal deadlines for all users, then purge expired data."""
    try:
        db = get_db()
        services = Services(db)
        as_of = strip_tz(utc_now())

        users_created = 0
        budgets_rolled = 0
        for user_id in services.users().list_active_user_ids():
            users_created += 1
            services.budget_service().evaluate_budgets(user_id, as_of=as_of)
            services.goal_service().check_deadlines(user_id, as_of=as_of)
            budgets_rolled += services.budget_service().roll_periods(user_id, as_of=as_of)

        purged = {
            "alerts": services.alert_service().purge_expired(),
            "predictions": services.predictions().purge_expired(),
            "reports": services.report_service().purge_expired(),
            "refresh_tokens": services.auth_service().purge_expired_tokens(),
        }
        logger.info("Alert sweep complete: users=%d, rolled=%d, purged=%s", users_created, budgets_rolled, purged)
        return {"users": users_created, "budgets_rolled": budgets_rolled, "purged": purged}
    except Exception as exc:  # pragma: no cover - scheduler safety
        logger.exception("Alert sweep failed: %s", exc)
        return {"status": "error", "error": str(exc)}

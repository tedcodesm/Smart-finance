# Mercy MongoDB schema

Reference for the collections the FastAPI backend and `mercy_ml` share. Every index
listed here is created by `backend/app/migrations.py` (run on startup, or manually via
`backend/scripts/init_db.py`).

Conventions:

- `_id` is an `ObjectId`; API responses expose it as the string field `id`.
- All datetimes are stored naive UTC (`tz_aware=False` driver setting). `mercy_ml` writes
  `generated_at`/`expires_at` as tz-aware UTC, which BSON stores identically.
- Money is stored as a BSON `double`, rounded to 2 decimals on write (`app/core/money.py`).
- Every user-owned document carries `user_id`; every query is scoped by it.
- `period_key` is the denormalised reporting bucket (`2026-09`, `2026-W38`, `2026`).

## Collections at a glance

| Collection      | Purpose                                     | Owner writes |
| --------------- | ------------------------------------------- | ------------ |
| `users`         | Accounts, credentials, profile              | backend      |
| `refresh_tokens`| Refresh-token revocation list               | backend      |
| `categories`    | System (shared) + user categories           | backend      |
| `transactions`  | Ledger of income/expense                    | backend      |
| `budgets`       | Per-category or overall spending limits     | backend      |
| `savings_goals` | Goal tracker with derived progress metrics  | backend      |
| `alerts`        | Deduplicated notifications                  | backend      |
| `reports`       | Persisted generated reports                 | backend      |
| `predictions`   | ML outputs (forecast/risk/recs/clusters)    | `mercy_ml`   |
| `model_runs`    | ML training + inference bookkeeping         | `mercy_ml`   |
| `schema_migrations` | Applied-migration versions              | backend      |

## users

```jsonc
{
  "_id": ObjectId,
  "name": "Ada Lovelace",
  "email": "ada@example.com",        // lowercased, unique
  "password_hash": "<bcrypt sha256 pre-hash digest>",
  "base_currency": "USD",
  "is_active": true,
  "monthly_income_estimate": null,   // optional override for ML priors
  "last_login_at": ISODate | null,
  "created_at": ISODate,
  "updated_at": ISODate
}
```

Indexes: `uq_users_email` (unique `email`), `ix_users_created_at`, `ix_users_active`.

## refresh_tokens

```jsonc
{
  "_id": ObjectId,
  "user_id": ObjectId,
  "jti": "<jwt id>",                 // unique; matches the refresh token's `jti` claim
  "revoked": false,
  "revoked_at": ISODate | null,
  "created_at": ISODate,
  "expires_at": ISODate
}
```

Indexes: `uq_refresh_jti` (unique), `ix_refresh_user`, `ix_refresh_expires` (TTL 0).

## categories

System categories are shared (`user_id: null`, `is_system: true`) and seeded on first
startup; user categories are private.

```jsonc
{
  "_id": ObjectId,
  "user_id": ObjectId | null,        // null => system category
  "name": "Groceries",
  "type": "expense",                 // expense | income
  "color": "#22c55e",
  "icon": "utensils",
  "monthly_typical_amount": 400.0,   // ML prior
  "is_system": false,
  "is_archived": false,
  "created_at": ISODate,
  "updated_at": ISODate
}
```

Indexes: `ix_categories_user_type`, `uq_categories_user_name_type` (unique, partial
`is_system: false`), `uq_categories_system_name_type` (unique, partial `is_system: true`),
`ix_categories_archived`.

## transactions

`amount` is always positive; `type` carries the sign (`signed_amount` is derived in the
API layer). The date parts are denormalised so reports and ML features never need
aggregation pipelines.

```jsonc
{
  "_id": ObjectId,
  "user_id": ObjectId,
  "category_id": ObjectId | null,    // null = uncategorised
  "type": "expense",                 // expense | income
  "amount": 42.5,
  "date": ISODate,
  "description": "Weekly shop",
  "notes": null,
  "is_recurring": false,
  "year": 2026, "month": 9, "day": 26,
  "day_of_week": 5, "day_of_month": 26, "week_of_year": 39,
  "period_key": "2026-09",
  "week_key": "2026-W39",
  "created_at": ISODate,
  "updated_at": ISODate
}
```

Indexes: `ix_txn_user_date`, `ix_txn_user_period_type_cat`, `ix_txn_user_cat_date`,
`ix_txn_user_type_date`, `ix_txn_user_created`, `ix_txn_user_dow`,
`ix_txn_description_text` (text index on `description`).

## budgets

One active budget per `(user, category, period)`; `category_id: null` is the overall
budget. Progress fields are recomputed on every transaction write, budget write and
nightly sweep.

```jsonc
{
  "_id": ObjectId,
  "user_id": ObjectId,
  "category_id": ObjectId | null,
  "name": "Groceries",
  "limit_amount": 300.0,
  "period": "monthly",               // weekly | monthly | yearly
  "start_date": ISODate,
  "end_date": ISODate | null,
  "period_key": "2026-09",
  "alert_threshold_pct": 0.8,
  "rollover": false,
  "is_active": true,
  "spent_amount": 180.0,             // cached
  "period_elapsed_pct": 82.6,        // cached
  "as_of": "2026-09",                // bucket the cache was computed for
  "last_alerted_at": ISODate | null,
  "created_at": ISODate,
  "updated_at": ISODate
}
```

`remaining_amount`, `usage_pct`, `days_remaining`, `is_exceeded`,
`is_threshold_crossed`, `projected_total` and `pace_status` are derived in the service
layer, not stored.

Indexes: `ix_budget_user_active_period`, `ix_budget_user_cat_period_start`,
`ix_budget_user_period_key`, `ix_budget_user_start_date`.

## savings_goals

```jsonc
{
  "_id": ObjectId,
  "user_id": ObjectId,
  "name": "Emergency fund",
  "description": null,
  "target_amount": 5000.0,
  "current_amount": 1250.0,
  "target_date": ISODate | null,
  "status": "active",                // active | achieved | cancelled
  "color": "#38bdf8",
  "linked_category_id": ObjectId | null,
  "progress_pct": 25.0,              // derived, stored
  "remaining_amount": 3750.0,        // derived, stored
  "days_remaining": 214,             // derived, stored
  "required_monthly_contribution": 168.0,
  "required_daily_contribution": 5.55,
  "on_track": true,
  "achieved_at": ISODate | null,
  "created_at": ISODate,
  "updated_at": ISODate
}
```

Indexes: `ix_goals_user_status`, `ix_goals_user_target_date`, `ix_goals_user_name`.

## alerts

`dedupe_key` is what makes alerts idempotent: a sparse unique index over
`(user_id, dedupe_key)` means re-running a budget check or the nightly sweep can never
insert the same alert twice.

```jsonc
{
  "_id": ObjectId,
  "user_id": ObjectId,
  "type": "budget_exceeded",          // budget_threshold | budget_exceeded | goal_deadline | ...
  "title": "Groceries budget exceeded",
  "message": "You have spent 130.00 of 100.00 this month.",
  "severity": "warning",              // info | warning | critical
  "context": { "budget_id": ObjectId, "category_id": ObjectId, "limit_amount": 100.0 },
  "is_read": false,
  "read_at": ISODate | null,
  "triggered_at": ISODate,
  "expires_at": ISODate | null,
  "dedupe_key": "budget_exceeded:6ab7...:2026-09"
}
```

Indexes: `ix_alert_user_unread_recent`, `ix_alert_user_type_recent`,
`uq_alert_user_dedupe` (unique, sparse), `ix_alert_severity`.

## reports

```jsonc
{
  "_id": ObjectId,
  "user_id": ObjectId,
  "report_type": "income_vs_expense", // spending_by_category | income_vs_expense | budget_performance | net_cash_flow | goal_progress
  "period": "monthly",
  "period_start": ISODate,
  "period_end": ISODate,
  "period_key": "2026-09",
  "params": { "limit_top": 20 },      // echoed request parameters
  "data": { "totals": {...}, "by_category": [...], "timeline": [...], "budgets": [...], "goals": [...], "insights": [...] },
  "summary": { "income": 2500.0, "expense": 180.0, "net": 2320.0, "...": "..." },
  "format": "json",
  "generated_at": ISODate,
  "expires_at": ISODate               // report_retention_days
}
```

Indexes: `ix_report_user_type_recent`, `ix_report_user_range`, `ix_report_user_recent`.

## predictions

Written exclusively by `mercy_ml.infer`. Inference is idempotent: it deletes the prior
documents for the same `(user, type, scope.category_id, scope.period_key)` and inserts
the fresh set, so a re-run replaces rather than duplicates.

```jsonc
{
  "_id": ObjectId,
  "user_id": ObjectId,
  "type": "expense_forecast",        // expense_forecast | overspend_risk | budget_recommendation | savings_recommendation | spending_cluster
  "value": 412.75,                   // amount / probability / limit, depending on type
  "confidence_score": 0.81,
  "model_type": "expense_forecaster",
  "model_version": "expense_forecaster_v3",
  "model_run_id": "6ab7...",
  "scope": {
    "category_id": "6ab7...",        // string, not ObjectId (ML-owned namespace)
    "category_name": "Groceries",
    "period_key": "2026-10",
    "period": "monthly"
  },
  "label": "up",                     // optional
  "features": { "...": "..." },
  "horizon_days": 30,                // optional
  "generated_at": ISODate,
  "expires_at": ISODate              // drives the alert/prediction purge job
}
```

Indexes: `ix_pred_user_type_recent`, `ix_pred_user_recent`, `ix_pred_user_scope`,
`ix_pred_run`, `ix_pred_user_model_version`, `ix_pred_expires` (declared as a TTL index
but disabled by default - see `MIGRATION_NOTES` in `index_definitions.py`).

## model_runs

```jsonc
{
  "_id": ObjectId,
  "run_id": "6ab7...",
  "run_type": "inference",           // training | inference
  "status": "success",               // success | partial | failed
  "is_active": true,
  "model_type": "expense_forecaster",
  "trained_at": ISODate,             // training runs
  "n_samples": 1840, "n_users": 6,   // training runs
  "models": ["expense_forecaster", "overspend_classifier", "..."],
  "users_scored": 1,                 // inference runs
  "predictions_written": 14,
  "as_of": "2026-09-26", "period_key": "2026-09",
  "errors": [],
  "started_at": ISODate, "finished_at": ISODate, "created_at": ISODate
}
```

Indexes: `uq_model_run_id` (unique), `ix_model_run_type_active`, `ix_model_run_trained`.

## refresh-token and purge lifecycle

| Data         | Cleanup                                            |
| ------------ | -------------------------------------------------- |
| Refresh tokens | TTL index on `refresh_tokens.expires_at`          |
| Predictions  | `ml_service`/scheduler delete `expires_at <= now`   |
| Alerts       | nightly sweep deletes `expires_at <= now`          |
| Reports      | nightly sweep deletes `expires_at <= now`          |

## Applying the schema

```powershell
# indexes + system categories on app startup (automatic)
uvicorn app.main:app --reload

# or explicitly
python scripts/init_db.py
```

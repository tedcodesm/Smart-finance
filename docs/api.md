# Mercy API

FastAPI backend for Mercy, a personal-finance tracker with ML-assisted forecasting.
All endpoints are mounted under `/api/v1`; OpenAPI docs are served at `/docs` and
`/openapi.json`.

MongoDB document shapes are documented in [`docs/schema.md`](../docs/schema.md).

## Running

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
pip install -e ml

# indexes + system categories are created on startup
uvicorn app.main:app --reload --app-dir backend
```

Environment (see `backend/.env.example`): `MONGO_URI`, `MONGO_DB_NAME`, `SECRET_KEY`,
`ACCESS_TOKEN_EXPIRE_MINUTES`, `REFRESH_TOKEN_EXPIRE_DAYS`, `ENABLE_SCHEDULER`,
`ML_BATCH_HOUR`/`ML_BATCH_MINUTE`, `ALERT_SWEEP_HOUR`/`ALERT_SWEEP_MINUTE`,
`ML_ARTIFACT_DIR`.

## Auth

`POST /api/v1/auth/register` and `/login` return an access + refresh token pair.
Send `Authorization: Bearer <access_token>` on every other endpoint. Access tokens are
short-lived JWTs; refresh tokens are tracked in `refresh_tokens` so they can be revoked
server-side before they expire.

## Endpoints

| Method | Path | Notes |
| ------ | ---- | ----- |
| GET | `/` | service banner |
| GET | `/api/v1/health` | liveness + database status (public) |
| POST | `/api/v1/auth/register` | create account, returns tokens |
| POST | `/api/v1/auth/login` | returns tokens |
| POST | `/api/v1/auth/refresh` | rotate a refresh token |
| POST | `/api/v1/auth/logout` | revoke a refresh token |
| GET/PATCH | `/api/v1/users/me` | profile |
| POST | `/api/v1/users/me/password` | change password |
| GET/POST | `/api/v1/categories` | list / create (`?type=`, `?include_archived=`) |
| GET/PATCH/DELETE | `/api/v1/categories/{id}` | `?force=true` unassigns transactions on delete |
| GET/POST | `/api/v1/transactions` | paginated list / create |
| GET/PATCH/DELETE | `/api/v1/transactions/{id}` | |
| POST | `/api/v1/transactions/bulk` | create many |
| GET | `/api/v1/transactions/summary` | income/expense/net totals |
| GET | `/api/v1/transactions/export` | `?format=csv\|pdf` |
| GET/POST | `/api/v1/budgets` | list / create |
| GET/PATCH/DELETE | `/api/v1/budgets/{id}` | |
| GET | `/api/v1/budgets/{id}/progress` | spend, pace, projection |
| GET/POST | `/api/v1/goals` | list / create |
| GET | `/api/v1/goals/summary` | aggregate progress |
| GET/PATCH/DELETE | `/api/v1/goals/{id}` | |
| POST | `/api/v1/goals/{id}/contribute` | returns a `GoalProgressEvent` |
| GET | `/api/v1/alerts` | `?is_read=`, `?types=`, `?severities=` |
| GET | `/api/v1/alerts/counts` | unread/total since a date |
| PATCH | `/api/v1/alerts/read` | mark by ids, type, severity, or all |
| DELETE | `/api/v1/alerts/{id}` | |
| POST | `/api/v1/reports` | generate + persist, returns the stored report |
| POST | `/api/v1/reports/generate` | generate without persisting |
| GET | `/api/v1/reports` | persisted reports |
| GET/DELETE | `/api/v1/reports/{id}` | |
| GET | `/api/v1/reports/{id}/export` | `?format=csv\|pdf\|json` |
| GET | `/api/v1/predictions` | stored predictions (`?type=` or `?types=`) |
| GET | `/api/v1/predictions/latest` | most recent value per type |
| GET | `/api/v1/predictions/models/status` | artifact registry + last run |
| POST | `/api/v1/predictions/inference/run` | score the caller now |
| POST | `/api/v1/predictions/expense-forecast` | forecast for the next period |
| POST | `/api/v1/predictions/overspend-risk` | per-budget overspend probability |
| POST | `/api/v1/predictions/budget-recommendations` | suggested limits |
| POST | `/api/v1/predictions/savings-recommendation` | goal contribution plan |
| GET | `/api/v1/predictions/clusters` | spending clusters for the caller |
| GET | `/api/v1/dashboard` | cards, cash flow, budgets, goals, predictions, insights |

On-demand prediction endpoints reuse stored predictions when they are younger than 12
hours; call `POST /api/v1/predictions/inference/run` to force a re-score.

List endpoints return `{ "items": [...], "meta": { "total", "skip", "limit" } }` and
accept `?skip=`, `?limit=` (max 200) and `?sort=`.

## Errors

```jsonc
{ "code": "not_found", "message": "Transaction not found", "details": null }
```

`validation_error` 422, `unauthorized` 401, `forbidden` 403, `not_found` 404,
`conflict` 409, `model_unavailable` 503, `internal_error` 500.

## Scheduled jobs

`ENABLE_SCHEDULER=true` starts APScheduler with two idempotent jobs: `ml_batch`
(scores every active user) and `alert_sweep` (re-evaluates budgets and goal deadlines,
then purges expired alerts, predictions, reports and refresh tokens).

## Tests

```powershell
cd backend
..\.venv\Scripts\python.exe -m pytest
```

45 tests run against `mongomock`; the ML client is pointed at the same in-memory database,
so `tests/test_ml_integration.py` exercises the real `mercy_ml.infer` path end to end.

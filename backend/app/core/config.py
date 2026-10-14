"""Application settings, loaded from environment / `.env`."""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_DIR.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(REPO_ROOT / ".env", BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- app ---
    app_name: str = "Mercy"
    version: str = "0.1.0"
    environment: Literal["development", "staging", "production"] = "development"
    debug: bool = True
    api_v1_prefix: str = "/api/v1"
    log_level: str = "INFO"

    # --- mongodb ---
    mongo_uri: str = "mongodb://localhost:27017"
    mongo_db_name: str = "mercy"
    mongo_server_selection_timeout_ms: int = 5000
    #: Run against an in-process `mongomock` database instead of a MongoDB server.
    #: Everything (including `mercy_ml`) keeps working; data lives only for the
    #: lifetime of the process. Intended for demos and local UI work.
    in_memory_db: bool = False
    #: Seed the demo account (see `app/demo_seed.py`) when `in_memory_db` is on.
    seed_demo_data: bool = True
    demo_email: str = "demo@example.com"
    #: Must satisfy the same policy `POST /auth/register` enforces (letter + digit),
    #: otherwise the seeded account could never be recreated through the API.
    demo_password: str = "demopassword1"

    # --- auth ---
    secret_key: str = "dev-only-insecure-secret-change-me-0000000000000000"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60
    refresh_token_expire_days: int = 30
    password_min_length: int = 8
    bcrypt_rounds: int = 12

    # --- cors ---
    cors_origins: list[str] | str = Field(
        default_factory=lambda: ["http://localhost:5173", "http://127.0.0.1:5173"]
    )

    # --- domain defaults ---
    default_currency: str = "KES"
    budget_alert_threshold_pct: float = 0.8

    # --- retention ---
    report_retention_days: int = 365
    prediction_retention_days: int = 180

    # --- scheduler ---
    enable_scheduler: bool = True
    scheduler_timezone: str = "UTC"
    ml_batch_hour: int = 2
    ml_batch_minute: int = 17
    alert_sweep_hour: int = 7
    alert_sweep_minute: int = 23

    # --- ml ---
    ml_artifact_dir: str = "../ml/artifacts"
    ml_random_seed: int = 42
    ml_train_fraction: float = 0.8

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> object:
        """Accept a comma-separated string or a JSON list from the env."""
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return []
            if value.startswith("["):
                return value
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @property
    def artifact_dir(self) -> Path:
        """Absolute path to the ML artifact directory."""
        raw = Path(self.ml_artifact_dir)
        resolved = raw if raw.is_absolute() else (BACKEND_DIR / raw).resolve()
        if not any(resolved.glob("*.joblib")):
            backend_artifacts = BACKEND_DIR / "artifacts"
            if any(backend_artifacts.glob("*.joblib")):
                return backend_artifacts
        return resolved

    @property
    def is_production(self) -> bool:
        return self.environment == "production"


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    if settings.is_production and settings.secret_key.startswith("dev-only"):
        raise RuntimeError(
            "SECRET_KEY must be set to a strong random value in production. "
            "Generate one with: python -c \"import secrets;print(secrets.token_urlsafe(64))\""
        )
    return settings


settings = get_settings()

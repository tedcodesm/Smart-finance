"""ML configuration.

Paths resolve relative to this file so the package works from the repo root, from
`ml/`, and from an editable install without environment gymnastics.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PACKAGE_DIR = Path(__file__).resolve().parent
ML_DIR = PACKAGE_DIR.parents[1]
REPO_ROOT = ML_DIR.parent
DEFAULT_ARTIFACT_DIR = ML_DIR / "artifacts"


class MLSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(REPO_ROOT / ".env", ML_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- data access (read-only; the ML module never writes user collections) ---
    mongo_uri: str = "mongodb://localhost:27017"
    mongo_db_name: str = "mercy"
    mongo_server_selection_timeout_ms: int = 5000

    # --- artifacts ---
    artifact_dir: str = Field(default=str(DEFAULT_ARTIFACT_DIR))

    # --- reproducibility ---
    random_seed: int = 42
    train_fraction: float = 0.8
    #: Reject an observation whose `amount` is further than this many IQRs out.
    outlier_iqr_k: float = 3.0
    #: `log1p` amounts are winsorised to this percentile pair before clustering.
    winsorize: tuple[float, float] = (0.001, 0.999)

    # --- training windows ---
    min_months_history: int = 3
    rolling_window: int = 3
    #: Observations with fewer transactions than this are dropped as noise.
    min_transactions_per_period: int = 1

    # --- model selection ---
    kmeans_k_min: int = 2
    kmeans_k_max: int = 6
    #: Silhouette below this is treated as "no useful structure found".
    silhouette_floor: float = 0.12

    # --- budgets ---
    budget_buffer_pct: float = 0.05
    budget_rounding: float = 5.0

    @property
    def artifact_path(self) -> Path:
        raw = Path(self.artifact_dir)
        resolved = raw if raw.is_absolute() else (ML_DIR / raw).resolve()
        if not any(resolved.glob("*.joblib")):
            backend_artifacts = REPO_ROOT / "backend" / "artifacts"
            if any(backend_artifacts.glob("*.joblib")):
                return backend_artifacts
        return resolved

    def artifact(self, name: str) -> Path:
        return self.artifact_path / name


@functools.lru_cache(maxsize=1)
def get_settings() -> MLSettings:
    return MLSettings()


settings = get_settings()


def ensure_artifact_dir() -> Path:
    path = settings.artifact_path
    path.mkdir(parents=True, exist_ok=True)
    return path


def model_version(model_name: str) -> str:
    """Version tag stamped onto every prediction document."""
    return f"{model_name}_v1"


def artifact_name(model_name: str) -> str:
    return f"{model_name}.joblib"


def metrics_name(model_name: str) -> str:
    return f"{model_name}_metrics.json"


def manifest_name() -> str:
    return "manifest.json"


def as_jsonable(value: Any) -> Any:
    """Best-effort conversion of numpy/pandas scalars for JSON serialisation."""
    if isinstance(value, dict):
        return {str(k): as_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_jsonable(v) for v in value]
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return value.item()
        except (ValueError, AttributeError):  # pragma: no cover - defensive
            return str(value)
    if isinstance(value, float) and value != value:  # NaN
        return None
    return value

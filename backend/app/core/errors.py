"""Domain exceptions mapped to HTTP responses by `app.api.errors`."""

from __future__ import annotations

from typing import Any


class AppError(Exception):
    """Base class for expected, user-facing failures."""

    status_code: int = 400
    code: str = "app_error"

    def __init__(self, message: str, *, details: Any = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.details is not None:
            payload["details"] = self.details
        return payload


class NotFoundError(AppError):
    status_code = 404
    code = "not_found"


class ConflictError(AppError):
    status_code = 409
    code = "conflict"


class ValidationError(AppError):
    status_code = 422
    code = "validation_error"


class AuthenticationError(AppError):
    status_code = 401
    code = "authentication_error"


class PermissionError_(AppError):
    status_code = 403
    code = "permission_denied"


class ServiceUnavailableError(AppError):
    """ML artifacts missing / not trained yet."""

    status_code = 503
    code = "model_unavailable"

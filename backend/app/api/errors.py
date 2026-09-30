"""Exception handlers mapping domain errors to HTTP responses.

`AppError` subclasses carry their own status code; everything else falls through
to a generic 500 so the API never leaks Python tracebacks to clients.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette import status
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.errors import AppError

logger = logging.getLogger("mercy.api.errors")

#: Starlette renamed this constant; support both so the import never warns.
HTTP_422 = getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", None) or status.HTTP_422_UNPROCESSABLE_ENTITY

#: Maps an HTTP status onto the stable error code documented in `docs/api.md`, so
#: clients can branch on `code` without inspecting the numeric status.
_STATUS_CODES: dict[int, str] = {
    status.HTTP_400_BAD_REQUEST: "bad_request",
    status.HTTP_401_UNAUTHORIZED: "unauthorized",
    status.HTTP_403_FORBIDDEN: "forbidden",
    status.HTTP_404_NOT_FOUND: "not_found",
    status.HTTP_405_METHOD_NOT_ALLOWED: "method_not_allowed",
    status.HTTP_409_CONFLICT: "conflict",
    HTTP_422: "validation_error",
    status.HTTP_429_TOO_MANY_REQUESTS: "rate_limited",
    status.HTTP_503_SERVICE_UNAVAILABLE: "model_unavailable",
}


def _payload(code: str, message: str, *, details: Any = None) -> dict[str, Any]:
    body: dict[str, Any] = {"code": code, "message": message}
    if details is not None:
        body["details"] = details
    return body


def _handle_app_error(request: Request, exc: AppError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content=exc.to_payload())


def _handle_http(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    headers = getattr(exc, "headers", None)
    code = _STATUS_CODES.get(exc.status_code, "http_error")
    detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content=_payload(code, detail),
        headers=headers,
    )


def _handle_pydantic(request: Request, exc: RequestValidationError) -> JSONResponse:
    errors = [
        {
            "loc": ".".join(str(part) for part in err.get("loc", ())),
            "msg": err.get("msg", "invalid"),
            "type": err.get("type", "validation_error"),
        }
        for err in exc.errors()
    ]
    return JSONResponse(
        status_code=HTTP_422,
        content=_payload("validation_error", "Request validation failed", details=errors),
    )


def _handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=_payload("internal_error", "Internal server error"),
    )


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(AppError, _handle_app_error)
    # Registered against Starlette's class, not FastAPI's subclass: unmatched
    # routes and unsupported methods raise `starlette.HTTPException`, whose MRO
    # does not include `fastapi.HTTPException`, so the handler would be missed.
    app.add_exception_handler(StarletteHTTPException, _handle_http)
    app.add_exception_handler(RequestValidationError, _handle_pydantic)
    app.add_exception_handler(Exception, _handle_unexpected)
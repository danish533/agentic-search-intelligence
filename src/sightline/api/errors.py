"""HTTP error handling.

One envelope for every failure (:class:`ErrorResponse`), so a client needs one branch rather
than one per endpoint, and one mapping from domain error to status code - declared here rather
than as a ``try/except`` in each route.

Every error response carries the ``correlation_id``. That is the whole point of the trace: a
user reporting "it returned a 500" hands over an identifier that reconstructs the request
node by node.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from sightline.api.v1.schemas.common import ErrorResponse
from sightline.application.ports.errors import (
    AuthenticationError,
    CircuitOpenError,
    ExternalServiceError,
)
from sightline.domain.errors import (
    DomainError,
    DomainValidationError,
    DuplicateProfileError,
    EntityNotFoundError,
)
from sightline.observability.correlation import CORRELATION_ID_HEADER, current_correlation_id
from sightline.observability.logging import get_logger

_logger = get_logger(__name__)

_ERROR_SUFFIX = re.compile(r"Error$")
_CAMEL_BOUNDARY = re.compile(r"(?<!^)(?=[A-Z])")


def error_code(exc: BaseException) -> str:
    """Derive a stable machine-readable code from the exception type.

    ``ProfileNotFoundError`` -> ``profile_not_found``. Generated rather than hand-maintained,
    so a new error type cannot ship with a missing or mistyped code.
    """
    name = _ERROR_SUFFIX.sub("", type(exc).__name__)
    return _CAMEL_BOUNDARY.sub("_", name).lower()


def _correlation_id(request: Request | None) -> str | None:
    """The request's trace id, from state first and the context var second.

    State is authoritative because the catch-all handler runs outside the middleware that
    binds the context var, and that is exactly the response a user most needs an id for.
    """
    if request is not None:
        stored = getattr(request.state, "correlation_id", None)
        if isinstance(stored, str) and stored:
            return stored
    return current_correlation_id()


def _envelope(
    exc: BaseException,
    *,
    status_code: int,
    request: Request | None = None,
    message: str | None = None,
    detail: list[dict[str, Any]] | None = None,
) -> JSONResponse:
    body = ErrorResponse(
        error=error_code(exc),
        message=message or str(exc),
        detail=detail,
        correlation_id=_correlation_id(request),
    )
    response = JSONResponse(status_code=status_code, content=body.model_dump(mode="json"))
    if body.correlation_id:
        response.headers[CORRELATION_ID_HEADER] = body.correlation_id
    return response


async def domain_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Map a domain error onto its HTTP status.

    Matched by type rather than by an ``isinstance`` ladder in each route, so adding an error
    type is one entry here and nothing else.
    """
    match exc:
        case EntityNotFoundError():
            status_code = status.HTTP_404_NOT_FOUND
        case DuplicateProfileError():
            status_code = status.HTTP_409_CONFLICT
        case DomainValidationError():
            status_code = status.HTTP_422_UNPROCESSABLE_CONTENT
        case _:
            status_code = status.HTTP_400_BAD_REQUEST

    _logger.info("http.domain_error", error_type=type(exc).__name__, status_code=status_code)
    return _envelope(exc, status_code=status_code, request=request)


async def external_service_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Map an upstream dependency failure onto 502/503.

    503 for an open circuit or an outage the caller can usefully retry; 502 for an upstream
    that answered but unusably. An auth failure is *our* misconfiguration, not the client's,
    so it is a 500-class error and never echoes the credential.
    """
    if isinstance(exc, AuthenticationError):
        _logger.error("http.upstream_auth_failure", service=exc.service)
        return _envelope(
            exc,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            request=request,
            message="The service is misconfigured for its upstream data provider.",
        )

    status_code = (
        status.HTTP_503_SERVICE_UNAVAILABLE
        if isinstance(exc, CircuitOpenError)
        else status.HTTP_502_BAD_GATEWAY
    )
    _logger.warning("http.upstream_error", error_type=type(exc).__name__)
    return _envelope(exc, status_code=status_code, request=request)


async def validation_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Render FastAPI's request-validation failures in the same envelope as everything else."""
    detail: list[dict[str, Any]] = []
    if isinstance(exc, RequestValidationError):
        detail = [
            {
                "field": ".".join(str(part) for part in error.get("loc", ())[1:]) or "<body>",
                "problem": error.get("msg", ""),
            }
            for error in exc.errors()
        ]
    return _envelope(
        exc,
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        request=request,
        message="Request validation failed.",
        detail=detail,
    )


async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Last resort. Logs the traceback, returns nothing internal to the client."""
    _logger.exception("http.unhandled_error", error_type=type(exc).__name__)
    return _envelope(
        exc,
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        request=request,
        # Never leak an internal message: it can contain a DSN, a query, or a file path.
        message="An unexpected error occurred. Quote the correlation id when reporting it.",
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Install every handler. Called once by the application factory."""
    handlers: dict[type[Exception], Callable[[Request, Exception], Awaitable[JSONResponse]]] = {
        DomainError: domain_error_handler,
        ExternalServiceError: external_service_error_handler,
        RequestValidationError: validation_error_handler,
        Exception: unhandled_error_handler,
    }
    for exception_type, handler in handlers.items():
        app.add_exception_handler(exception_type, handler)

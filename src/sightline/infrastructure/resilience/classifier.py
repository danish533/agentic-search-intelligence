"""Error classification.

Turns raw transport failures and HTTP status codes into the typed taxonomy in
:mod:`sightline.infrastructure.resilience.errors`. This is the *only* module that knows which
status codes are worth retrying, so the policy cannot drift between call sites.

Deliberately generic HTTP: protocol-specific quirks stay with their adapter. DataForSEO, for
instance, returns HTTP 200 with an in-body ``status_code`` field, and interpreting that is the
DataForSEO adapter's job - it raises these same error types, but its own rules produce them.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Final

import httpx

from sightline.application.ports.errors import (
    AuthenticationError,
    BadRequestError,
    ExternalServiceError,
    RateLimitedError,
    ResourceNotFoundError,
    TransientNetworkError,
    UpstreamServerError,
)

#: 408 request timeout, 425 too early, 429 rate limited - all safe to repeat verbatim.
RETRYABLE_CLIENT_STATUS: Final[frozenset[int]] = frozenset({408, 425, 429})
AUTH_STATUS: Final[frozenset[int]] = frozenset({401, 403})
BAD_REQUEST_STATUS: Final[frozenset[int]] = frozenset({400, 422})

_MAX_BODY_EXCERPT_CHARS: Final = 300


def parse_retry_after(raw: str | None) -> float | None:
    """Parse a ``Retry-After`` header, which RFC 9110 allows in two forms.

    Accepts delta-seconds (``"120"``) and an HTTP-date
    (``"Wed, 21 Oct 2026 07:28:00 GMT"``). Returns ``None`` for absent or unparseable values -
    an unreadable hint must not break the retry path, it just falls back to computed backoff.
    """
    if raw is None:
        return None

    value = raw.strip()
    if not value:
        return None

    try:
        seconds = float(value)
    except ValueError:
        pass
    else:
        return max(0.0, seconds)

    try:
        deadline = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None

    if deadline.tzinfo is None:
        deadline = deadline.replace(tzinfo=UTC)
    return max(0.0, (deadline - datetime.now(UTC)).total_seconds())


def classify_status(
    status_code: int,
    *,
    service: str,
    endpoint: str | None = None,
    retry_after: str | None = None,
    body_excerpt: str | None = None,
) -> ExternalServiceError | None:
    """Classify an HTTP status. Returns ``None`` for a successful response.

    Unmapped 4xx codes are treated as non-retryable: a client error we do not specifically
    recognise is still a request *we* got wrong, and hammering the dependency with it is the
    wrong response.
    """
    if 200 <= status_code < 300:
        return None

    detail = (body_excerpt or "").strip()[:_MAX_BODY_EXCERPT_CHARS] or "no response body"

    if status_code == 429:
        return RateLimitedError(
            service=service,
            detail=detail,
            endpoint=endpoint,
            status_code=status_code,
            retry_after_seconds=parse_retry_after(retry_after),
        )

    if status_code in RETRYABLE_CLIENT_STATUS:
        return TransientNetworkError(
            service=service, detail=detail, endpoint=endpoint, status_code=status_code
        )

    if status_code in AUTH_STATUS:
        # Never echo the body of an auth failure: it can contain the submitted credential.
        return AuthenticationError(
            service=service,
            detail="authentication or authorisation failed",
            endpoint=endpoint,
            status_code=status_code,
        )

    if status_code == 404:
        return ResourceNotFoundError(
            service=service, detail=detail, endpoint=endpoint, status_code=status_code
        )

    if status_code in BAD_REQUEST_STATUS or 400 <= status_code < 500:
        return BadRequestError(
            service=service, detail=detail, endpoint=endpoint, status_code=status_code
        )

    return UpstreamServerError(
        service=service, detail=detail, endpoint=endpoint, status_code=status_code
    )


def classify_transport_exception(
    exc: BaseException,
    *,
    service: str,
    endpoint: str | None = None,
) -> ExternalServiceError | None:
    """Classify a transport-level exception.

    Returns ``None`` when the exception is **not** a transport failure. That is the important
    case: an unrecognised exception is a bug in our own code, and swallowing it into a tidy
    ``ExternalServiceError`` would hide it behind a retry loop. The caller re-raises instead.
    """
    if isinstance(exc, httpx.TimeoutException | asyncio.TimeoutError | TimeoutError):
        return TransientNetworkError(
            service=service, detail=f"timed out ({type(exc).__name__})", endpoint=endpoint
        )

    if isinstance(exc, httpx.RemoteProtocolError):
        return TransientNetworkError(
            service=service, detail="upstream closed the connection early", endpoint=endpoint
        )

    if isinstance(exc, httpx.TransportError):
        # Covers ConnectError, ReadError, WriteError, PoolTimeout, ProxyError.
        return TransientNetworkError(
            service=service, detail=f"{type(exc).__name__}: {exc}", endpoint=endpoint
        )

    if isinstance(exc, httpx.InvalidURL | httpx.UnsupportedProtocol | httpx.TooManyRedirects):
        return BadRequestError(
            service=service, detail=f"{type(exc).__name__}: {exc}", endpoint=endpoint
        )

    return None

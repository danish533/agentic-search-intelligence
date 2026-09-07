"""DataForSEO in-body status classification.

DataForSEO answers **HTTP 200 for application-level failures** and reports the real outcome in
a numeric ``status_code`` field, at two levels: once for the envelope and once per task. A
client that only checks the HTTP status therefore treats an out-of-credits error, an auth
failure and a successful response identically.

This module is the second half of error classification: :mod:`..resilience.classifier` handles
transport and HTTP status, and this handles the payload. Both produce the same error taxonomy,
so the retry executor makes one decision regardless of how the failure was expressed.

Codes are classified by **range** rather than by an exhaustive table. DataForSEO's code list
is long and versioned; ranges are stable, and an unrecognised code is far better classified by
its family than defaulted to "success".
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

from sightline.application.ports.errors import (
    AuthenticationError,
    BadRequestError,
    ExternalServiceError,
    RateLimitedError,
    UpstreamContractError,
    UpstreamServerError,
)

SERVICE_NAME: Final = "dataforseo"

#: The only status meaning "Ok."
STATUS_OK: Final = 20000

#: Task accepted but not yet complete - worth another attempt.
_IN_PROGRESS_RANGE: Final = range(30000, 40000)

#: 401xx authentication, 402xx payment/credits. Both are terminal: retrying an
#: out-of-credits or bad-credential call reproduces it exactly.
_AUTH_RANGE: Final = range(40100, 40300)

_CLIENT_RANGE: Final = range(40000, 50000)
_SERVER_RANGE: Final = range(50000, 60000)

#: Rate limiting is detected by message as well as code: DataForSEO expresses it through
#: more than one numeric code across API families, but the message is consistently explicit.
_RATE_LIMIT_MARKERS: Final[tuple[str, ...]] = ("rate limit", "too many requests", "limit exceeded")


def _is_rate_limited(status_code: int, message: str) -> bool:
    lowered = message.lower()
    return status_code == 42900 or any(marker in lowered for marker in _RATE_LIMIT_MARKERS)


def _classify_code(status_code: int, message: str, endpoint: str) -> ExternalServiceError | None:
    """Map one DataForSEO status code onto the shared error taxonomy."""
    if status_code == STATUS_OK:
        return None

    if _is_rate_limited(status_code, message):
        return RateLimitedError(
            service=SERVICE_NAME, detail=message, endpoint=endpoint, status_code=status_code
        )

    if status_code in _IN_PROGRESS_RANGE:
        return UpstreamServerError(
            service=SERVICE_NAME,
            detail=f"task not ready: {message}",
            endpoint=endpoint,
            status_code=status_code,
        )

    if status_code in _AUTH_RANGE:
        return AuthenticationError(
            service=SERVICE_NAME,
            detail=f"authentication, authorisation or account credit failure ({status_code})",
            endpoint=endpoint,
            status_code=status_code,
        )

    if status_code in _CLIENT_RANGE:
        return BadRequestError(
            service=SERVICE_NAME, detail=message, endpoint=endpoint, status_code=status_code
        )

    if status_code in _SERVER_RANGE:
        return UpstreamServerError(
            service=SERVICE_NAME, detail=message, endpoint=endpoint, status_code=status_code
        )

    return UpstreamContractError(
        service=SERVICE_NAME,
        detail=f"unrecognised status code {status_code}: {message}",
        endpoint=endpoint,
        status_code=status_code,
    )


def classify_payload(
    payload: Mapping[str, Any],
    *,
    endpoint: str,
) -> ExternalServiceError | None:
    """Classify a DataForSEO response body. Returns ``None`` when it is usable.

    Checks the envelope status first, then each task's status. A partially-failed multi-task
    response is reported by its first failing task, which is sufficient here because every
    call this system makes submits exactly one task.
    """
    # The payload is guaranteed to be a Mapping by the transport contract: both transports
    # construct TransportResponse.body, and HttpxTransport rejects a non-object JSON body as
    # UpstreamContractError before it reaches here. Re-checking would be provably dead code.
    envelope_code = payload.get("status_code")
    if not isinstance(envelope_code, int):
        return UpstreamContractError(
            service=SERVICE_NAME,
            detail="response is missing the required integer 'status_code' field",
            endpoint=endpoint,
        )

    envelope_error = _classify_code(envelope_code, str(payload.get("status_message", "")), endpoint)
    if envelope_error is not None:
        return envelope_error

    tasks = payload.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        return UpstreamContractError(
            service=SERVICE_NAME,
            detail="envelope reported success but carried no tasks",
            endpoint=endpoint,
        )

    for task in tasks:
        if not isinstance(task, Mapping):
            return UpstreamContractError(
                service=SERVICE_NAME, detail="malformed task entry", endpoint=endpoint
            )
        task_code = task.get("status_code")
        if not isinstance(task_code, int):
            return UpstreamContractError(
                service=SERVICE_NAME,
                detail="task is missing the required integer 'status_code' field",
                endpoint=endpoint,
            )
        task_error = _classify_code(task_code, str(task.get("status_message", "")), endpoint)
        if task_error is not None:
            return task_error

    return None

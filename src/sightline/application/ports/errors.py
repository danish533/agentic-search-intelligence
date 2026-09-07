"""External-dependency error taxonomy.

Spec S3.5 requires that retryable errors (network, timeout, rate limit) be distinguished from
non-retryable ones (bad request, auth failure) and "handled appropriately". That distinction
is encoded in the *type*, not in scattered ``if status == 429`` checks: the retry executor
decides purely on which branch of this hierarchy it caught, so classification lives in exactly
one place and cannot disagree with itself.

This lives in the **ports** package, not in infrastructure, because how a dependency fails is
part of its contract. The agent ring must be able to catch a failed retrieval and route around
it, and the dependency rule forbids it from importing an adapter to learn the exception type.
Adapters raise these; callers on either side of the port catch them.

::

    ExternalServiceError
    ├── RetryableError
    │   ├── TransientNetworkError   connection reset, DNS, timeout
    │   ├── RateLimitedError        429, may carry Retry-After
    │   └── UpstreamServerError     5xx
    ├── NonRetryableError
    │   ├── AuthenticationError     401, 403 - retrying cannot fix a credential
    │   ├── BadRequestError         400, 422 - our payload is wrong
    │   ├── ResourceNotFoundError   404
    │   └── UpstreamContractError   2xx whose body does not match the documented shape
    ├── RetryExhaustedError         retries spent; terminal by design
    └── CircuitOpenError            rejected without calling; terminal by design

``RetryExhaustedError`` and ``CircuitOpenError`` intentionally sit outside ``RetryableError``
so that an enclosing retry loop can never retry them a second time.
"""

from __future__ import annotations


class ExternalServiceError(Exception):
    """Base for every failure of a dependency outside this process."""

    def __init__(
        self,
        *,
        service: str,
        detail: str,
        endpoint: str | None = None,
        status_code: int | None = None,
    ) -> None:
        self.service = service
        self.detail = detail
        self.endpoint = endpoint
        self.status_code = status_code
        location = f"{service} {endpoint}" if endpoint else service
        status = f" [HTTP {status_code}]" if status_code is not None else ""
        super().__init__(f"{location}{status}: {detail}")

    def as_log_fields(self) -> dict[str, object]:
        """Structured fields for the observability layer - never the raw exception string."""
        return {
            "service": self.service,
            "endpoint": self.endpoint,
            "status_code": self.status_code,
            "error_type": type(self).__name__,
            "retryable": isinstance(self, RetryableError),
        }


class RetryableError(ExternalServiceError):
    """A failure that a later identical attempt could plausibly succeed at."""


class TransientNetworkError(RetryableError):
    """Connection failure, DNS failure, or timeout. No response was received."""


class RateLimitedError(RetryableError):
    """HTTP 429. Carries the server's ``Retry-After`` when one was supplied."""

    def __init__(
        self,
        *,
        service: str,
        detail: str,
        endpoint: str | None = None,
        status_code: int | None = 429,
        retry_after_seconds: float | None = None,
    ) -> None:
        self.retry_after_seconds = retry_after_seconds
        super().__init__(service=service, detail=detail, endpoint=endpoint, status_code=status_code)


class UpstreamServerError(RetryableError):
    """HTTP 5xx. The dependency accepted the request and failed to serve it."""


class NonRetryableError(ExternalServiceError):
    """A failure that repeating verbatim will reproduce exactly. Fail fast instead."""


class AuthenticationError(NonRetryableError):
    """HTTP 401/403. Credentials are missing, wrong, or lack permission."""


class BadRequestError(NonRetryableError):
    """HTTP 400/422. The request we constructed is invalid."""


class ResourceNotFoundError(NonRetryableError):
    """HTTP 404. The requested upstream resource does not exist."""


class UpstreamContractError(NonRetryableError):
    """A successful HTTP response whose body does not match the documented contract.

    Non-retryable on purpose: an identical request will return the identical unusable body.
    This is the error that turns a silent ``KeyError`` deep in a parser into a classified,
    logged, routable failure.
    """


class RetryExhaustedError(ExternalServiceError):
    """Every permitted attempt failed. Carries the final cause and the attempt count."""

    def __init__(
        self, *, service: str, endpoint: str | None, attempts: int, cause: Exception
    ) -> None:
        self.attempts = attempts
        self.cause = cause
        super().__init__(
            service=service,
            detail=f"exhausted {attempts} attempt(s); last failure: {cause}",
            endpoint=endpoint,
            status_code=getattr(cause, "status_code", None),
        )

    def as_log_fields(self) -> dict[str, object]:
        return {**super().as_log_fields(), "attempts": self.attempts}


class CircuitOpenError(NonRetryableError):
    """The circuit breaker rejected the call without attempting it.

    Non-retryable by classification: the entire point of an open circuit is to stop sending
    traffic to a dependency that is already known to be failing.
    """

    def __init__(
        self, *, service: str, endpoint: str | None = None, retry_in_seconds: float = 0.0
    ) -> None:
        self.retry_in_seconds = retry_in_seconds
        super().__init__(
            service=service,
            detail=f"circuit is open; next probe permitted in {retry_in_seconds:.1f}s",
            endpoint=endpoint,
        )

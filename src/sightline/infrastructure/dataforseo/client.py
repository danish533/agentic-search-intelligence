"""DataForSEO client.

The single place where a DataForSEO call is actually made, and the place where every
resilience and observability concern is applied. One method - :meth:`DataForSEOClient.execute`
- because every endpoint follows the identical shape: POST a one-element task array, classify
the HTTP status, classify the in-body status, retry what is retryable, and record the call.

Composition order matters and is deliberate::

    circuit breaker  ->  retry executor  ->  single attempt (transport + classification)

The breaker wraps the *whole* retry sequence rather than each attempt. An open circuit must
skip all three attempts, not fail three times faster; and a fully exhausted retry budget is
exactly the signal that the dependency is unhealthy, so it is what counts toward tripping it.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

from sightline.application.ports.errors import ExternalServiceError
from sightline.application.ports.search_data_provider import RawApiResult
from sightline.config.settings import DataForSEOSettings
from sightline.infrastructure.dataforseo.endpoints import Endpoint
from sightline.infrastructure.dataforseo.status import classify_payload
from sightline.infrastructure.dataforseo.transport import SERVICE_NAME, DataForSEOTransport
from sightline.infrastructure.resilience.circuit_breaker import CircuitBreaker
from sightline.infrastructure.resilience.classifier import (
    classify_status,
    classify_transport_exception,
)
from sightline.infrastructure.resilience.retry import RetryExecutor
from sightline.observability.logging import get_logger
from sightline.observability.metrics import ApiCallRecord, current_metrics

_logger = get_logger(__name__)

_MAX_BODY_EXCERPT_CHARS = 300


class DataForSEOClient:
    """Executes DataForSEO tasks with retries, circuit breaking and metrics."""

    def __init__(
        self,
        *,
        settings: DataForSEOSettings,
        transport: DataForSEOTransport,
        retry_executor: RetryExecutor,
        circuit_breaker: CircuitBreaker,
    ) -> None:
        self._settings = settings
        self._transport = transport
        self._retry = retry_executor
        self._breaker = circuit_breaker

    @property
    def mode_label(self) -> str:
        return self._transport.mode_label

    async def execute(self, endpoint: Endpoint, task: Mapping[str, Any]) -> RawApiResult:
        """Run one DataForSEO task and return its unparsed payload.

        Raises a classified :class:`ExternalServiceError` on failure. Callers do not inspect
        status codes - the taxonomy already tells them whether the failure was the
        dependency's, ours, or terminal.
        """
        started = time.perf_counter()

        try:
            outcome = await self._breaker.execute(
                lambda: self._retry.execute(
                    lambda: self._attempt(endpoint, task),
                    service=SERVICE_NAME,
                    endpoint=endpoint.path,
                ),
                endpoint=endpoint.path,
            )
        except ExternalServiceError as exc:
            duration_ms = (time.perf_counter() - started) * 1000
            attempts = int(getattr(exc, "attempts", 1))
            self._record_call(
                endpoint=endpoint,
                duration_ms=duration_ms,
                succeeded=False,
                attempts=attempts,
                status_code=exc.status_code,
                error_type=type(exc).__name__,
            )
            # Merged into a single mapping rather than splatted alongside explicit kwargs:
            # some error types (RetryExhaustedError) already contribute an "attempts" field,
            # and splatting both raises TypeError on duplicate keys - on the failure path
            # only, which is precisely where a logging call must not itself fail.
            _logger.error(  # noqa: TRY400 - re-raised; the node span logs the traceback once
                "dataforseo.call.failed",
                **{
                    **exc.as_log_fields(),
                    "duration_ms": round(duration_ms, 2),
                    "attempts": attempts,
                    "mode": self._transport.mode_label,
                },
            )
            raise

        duration_ms = (time.perf_counter() - started) * 1000
        self._record_call(
            endpoint=endpoint,
            duration_ms=duration_ms,
            succeeded=True,
            attempts=outcome.attempts,
            status_code=200,
            error_type=None,
        )
        _logger.info(
            "dataforseo.call.succeeded",
            endpoint=endpoint.path,
            kind=endpoint.kind.value,
            duration_ms=round(duration_ms, 2),
            attempts=outcome.attempts,
            retry_count=outcome.retry_count,
            mode=self._transport.mode_label,
        )

        return RawApiResult(
            endpoint=endpoint.path,
            payload=outcome.value,
            duration_ms=duration_ms,
            attempts=outcome.attempts,
            status_code=200,
            from_mock=self._transport.is_mock,
        )

    async def _attempt(self, endpoint: Endpoint, task: Mapping[str, Any]) -> Mapping[str, Any]:
        """One attempt: send, classify the HTTP status, then classify the in-body status."""
        try:
            response = await self._transport.post(endpoint.path, [task])
        except ExternalServiceError:
            # Already classified by the transport (e.g. a non-JSON body). Do not re-wrap.
            raise
        except Exception as exc:
            classified = classify_transport_exception(
                exc, service=SERVICE_NAME, endpoint=endpoint.path
            )
            if classified is None:
                # Not a transport failure - this is a bug in our own code. Let it propagate
                # rather than disguising it as a retryable dependency problem.
                raise
            raise classified from exc

        http_error = classify_status(
            response.status_code,
            service=SERVICE_NAME,
            endpoint=endpoint.path,
            retry_after=response.headers.get("retry-after"),
            body_excerpt=str(response.body)[:_MAX_BODY_EXCERPT_CHARS],
        )
        if http_error is not None:
            raise http_error

        # DataForSEO answers HTTP 200 for application-level failures, so a clean HTTP status
        # proves nothing on its own.
        payload_error = classify_payload(response.body, endpoint=endpoint.path)
        if payload_error is not None:
            raise payload_error

        return response.body

    def _record_call(
        self,
        *,
        endpoint: Endpoint,
        duration_ms: float,
        succeeded: bool,
        attempts: int,
        status_code: int | None,
        error_type: str | None,
    ) -> None:
        metrics = current_metrics()
        if metrics is None:
            return
        metrics.record_api_call(
            ApiCallRecord(
                endpoint=endpoint.path,
                duration_ms=duration_ms,
                succeeded=succeeded,
                attempts=attempts,
                status_code=status_code,
                error_type=error_type,
            )
        )

    async def aclose(self) -> None:
        await self._transport.aclose()

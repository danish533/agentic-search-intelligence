"""Circuit breaker (spec S3.5 bonus).

Retries protect a single call. They do nothing about a dependency that is comprehensively
down - there, retrying makes things worse: every one of the fan-out branches spends its full
attempt budget, the run takes ``branches x attempts x backoff`` seconds to fail, and the
struggling dependency receives more traffic than usual at its worst moment.

The breaker short-circuits that. After ``failure_threshold`` consecutive failures it opens and
rejects calls immediately, then admits a limited number of probes after ``reset_seconds`` to
discover whether the dependency has recovered.

::

    CLOSED ──failure_threshold consecutive failures──▶ OPEN
      ▲                                                 │
      │                                        reset_seconds elapsed
      │                                                 ▼
      └────half_open_successes probes succeed──── HALF_OPEN
                                                        │
                                    any probe fails ────┘ (back to OPEN)

Only *dependency* failures trip it. A 400 or a 401 means our request or our credentials are
wrong; the dependency is answering correctly and opening the circuit on that would be a
misdiagnosis that hides a configuration bug behind an availability symptom.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum

from sightline.application.ports.errors import (
    CircuitOpenError,
    RetryableError,
    RetryExhaustedError,
)
from sightline.config.settings import ResilienceSettings
from sightline.observability.logging import get_logger

_logger = get_logger(__name__)

#: Monotonic time source. Injected so tests can jump forward without sleeping.
Clock = Callable[[], float]


class CircuitState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass(frozen=True, slots=True)
class CircuitBreakerPolicy:
    """Thresholds governing the breaker's transitions."""

    failure_threshold: int
    reset_seconds: float
    half_open_successes: int

    @classmethod
    def from_settings(cls, settings: ResilienceSettings) -> CircuitBreakerPolicy:
        return cls(
            failure_threshold=settings.circuit_breaker_failure_threshold,
            reset_seconds=settings.circuit_breaker_reset_seconds,
            half_open_successes=settings.circuit_breaker_half_open_successes,
        )


def is_dependency_failure(exc: BaseException) -> bool:
    """Whether an exception indicates the dependency itself is unhealthy.

    ``RetryExhaustedError`` counts because it means a full retry budget was spent without
    success. ``NonRetryableError`` deliberately does not: see the module docstring.
    """
    return isinstance(exc, RetryableError | RetryExhaustedError)


class CircuitBreaker:
    """An async circuit breaker guarding one logical dependency.

    State is mutated only under an :class:`asyncio.Lock`, so concurrent fan-out branches
    cannot interleave a read-modify-write and, for example, admit five probes through a
    half-open circuit that should have admitted one.
    """

    def __init__(
        self,
        *,
        name: str,
        policy: CircuitBreakerPolicy,
        clock: Clock | None = None,
    ) -> None:
        self._name = name
        self._policy = policy
        self._clock: Clock = clock if clock is not None else time.monotonic
        self._lock = asyncio.Lock()

        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._opened_at: float | None = None
        self._half_open_successes = 0
        self._half_open_in_flight = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def state(self) -> CircuitState:
        return self._state

    def snapshot(self) -> dict[str, object]:
        """Current breaker state, for logs and the run's metrics summary."""
        return {
            "circuit": self._name,
            "state": self._state.value,
            "consecutive_failures": self._consecutive_failures,
        }

    async def execute[T](
        self,
        operation: Callable[[], Awaitable[T]],
        *,
        endpoint: str | None = None,
    ) -> T:
        """Run ``operation`` if the circuit permits it.

        Raises :class:`CircuitOpenError` without invoking ``operation`` when the circuit is
        open, or when a half-open circuit has already admitted its probe quota.
        """
        await self._admit(endpoint=endpoint)

        try:
            result = await operation()
        except Exception as exc:
            if is_dependency_failure(exc):
                await self._record_failure()
            else:
                await self._release_probe()
            raise
        else:
            await self._record_success()
            return result

    async def _admit(self, *, endpoint: str | None) -> None:
        async with self._lock:
            if self._state is CircuitState.OPEN:
                elapsed = self._clock() - (self._opened_at or 0.0)
                remaining = self._policy.reset_seconds - elapsed
                if remaining > 0:
                    raise CircuitOpenError(
                        service=self._name, endpoint=endpoint, retry_in_seconds=remaining
                    )
                self._transition_to_half_open()

            if self._state is CircuitState.HALF_OPEN:
                if self._half_open_in_flight >= self._policy.half_open_successes:
                    raise CircuitOpenError(
                        service=self._name,
                        endpoint=endpoint,
                        retry_in_seconds=self._policy.reset_seconds,
                    )
                self._half_open_in_flight += 1

    async def _record_success(self) -> None:
        async with self._lock:
            if self._state is CircuitState.HALF_OPEN:
                self._half_open_in_flight = max(0, self._half_open_in_flight - 1)
                self._half_open_successes += 1
                if self._half_open_successes >= self._policy.half_open_successes:
                    self._close()
                return
            self._consecutive_failures = 0

    async def _record_failure(self) -> None:
        async with self._lock:
            if self._state is CircuitState.HALF_OPEN:
                # A failed probe means the dependency has not recovered: re-open immediately
                # rather than spending the rest of the probe quota discovering the same thing.
                self._half_open_in_flight = max(0, self._half_open_in_flight - 1)
                self._open(reason="probe_failed")
                return

            self._consecutive_failures += 1
            # Only a CLOSED circuit may open. Calls already in flight when the threshold was
            # crossed keep failing and arriving here; without this guard each one re-opened an
            # already-open circuit, which (a) reset `opened_at` and so pushed the recovery
            # deadline further out every time, and (b) logged "circuit.opened" repeatedly,
            # making one outage look like six in an incident timeline.
            if (
                self._state is CircuitState.CLOSED
                and self._consecutive_failures >= self._policy.failure_threshold
            ):
                self._open(reason="failure_threshold_reached")

    async def _release_probe(self) -> None:
        """Release a half-open slot for a failure that does not count against the breaker."""
        async with self._lock:
            if self._state is CircuitState.HALF_OPEN:
                self._half_open_in_flight = max(0, self._half_open_in_flight - 1)

    # -- transitions ------------------------------------------------------------------
    # Called with the lock already held.

    def _open(self, *, reason: str) -> None:
        self._state = CircuitState.OPEN
        self._opened_at = self._clock()
        self._half_open_successes = 0
        self._half_open_in_flight = 0
        _logger.error(
            "circuit.opened",
            circuit=self._name,
            reason=reason,
            consecutive_failures=self._consecutive_failures,
            reset_seconds=self._policy.reset_seconds,
        )

    def _transition_to_half_open(self) -> None:
        self._state = CircuitState.HALF_OPEN
        self._half_open_successes = 0
        self._half_open_in_flight = 0
        _logger.warning("circuit.half_open", circuit=self._name)

    def _close(self) -> None:
        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._opened_at = None
        self._half_open_successes = 0
        self._half_open_in_flight = 0
        _logger.info("circuit.closed", circuit=self._name)

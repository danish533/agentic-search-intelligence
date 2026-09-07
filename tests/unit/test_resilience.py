"""Retry, backoff, error classification and circuit breaking (spec S3.5)."""

from __future__ import annotations

import asyncio
import random
from typing import Any

import httpx
import pytest

from sightline.application.ports.errors import (
    AuthenticationError,
    BadRequestError,
    CircuitOpenError,
    NonRetryableError,
    RateLimitedError,
    ResourceNotFoundError,
    RetryableError,
    RetryExhaustedError,
    TransientNetworkError,
    UpstreamServerError,
)
from sightline.config.settings import ResilienceSettings
from sightline.infrastructure.resilience.backoff import BackoffPolicy
from sightline.infrastructure.resilience.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerPolicy,
    CircuitState,
)
from sightline.infrastructure.resilience.classifier import (
    classify_status,
    classify_transport_exception,
    parse_retry_after,
)
from sightline.infrastructure.resilience.retry import RetryExecutor, RetryPolicy


def failing_then(*outcomes: Any) -> Any:
    """An operation that yields each outcome in turn, raising the ones that are exceptions."""
    iterator = iter(outcomes)

    async def operation() -> Any:
        value = next(iterator)
        if isinstance(value, Exception):
            raise value
        return value

    return operation


class TestErrorClassification:
    @pytest.mark.parametrize(
        ("status_code", "expected"),
        [
            (408, TransientNetworkError),
            (425, TransientNetworkError),
            (429, RateLimitedError),
            (500, UpstreamServerError),
            (502, UpstreamServerError),
            (503, UpstreamServerError),
        ],
    )
    def test_retryable_statuses(self, status_code: int, expected: type) -> None:
        error = classify_status(status_code, service="dfs")
        assert isinstance(error, expected)
        assert isinstance(error, RetryableError)

    @pytest.mark.parametrize(
        ("status_code", "expected"),
        [
            (400, BadRequestError),
            (401, AuthenticationError),
            (403, AuthenticationError),
            (404, ResourceNotFoundError),
            (422, BadRequestError),
            (451, BadRequestError),
        ],
    )
    def test_non_retryable_statuses(self, status_code: int, expected: type) -> None:
        error = classify_status(status_code, service="dfs")
        assert isinstance(error, expected)
        assert isinstance(error, NonRetryableError)

    @pytest.mark.parametrize("status_code", [200, 201, 204, 299])
    def test_success_is_not_an_error(self, status_code: int) -> None:
        assert classify_status(status_code, service="dfs") is None

    def test_auth_failure_never_echoes_the_response_body(self) -> None:
        """The body of a 401 can contain the credential that was submitted."""
        error = classify_status(401, service="dfs", body_excerpt="login=user password=hunter2")
        assert error is not None
        assert "hunter2" not in str(error)
        assert "password" not in str(error)

    @pytest.mark.parametrize(
        "exception",
        [
            httpx.ConnectTimeout("t"),
            httpx.ReadTimeout("t"),
            httpx.ConnectError("c"),
            httpx.RemoteProtocolError("p"),
            TimeoutError(),
        ],
    )
    def test_transport_failures_are_retryable(self, exception: Exception) -> None:
        error = classify_transport_exception(exception, service="dfs")
        assert isinstance(error, RetryableError)

    def test_a_non_transport_exception_is_not_classified(self) -> None:
        """Our own bugs must propagate, not be laundered into a retryable dependency error."""
        assert classify_transport_exception(ValueError("our bug"), service="dfs") is None

    @pytest.mark.parametrize(
        ("header", "expected"),
        [("120", 120.0), ("0", 0.0), ("not-a-date", None), (None, None), ("", None)],
    )
    def test_retry_after_parsing(self, header: str | None, expected: float | None) -> None:
        assert parse_retry_after(header) == expected


class TestBackoff:
    @pytest.fixture
    def policy(self) -> BackoffPolicy:
        return BackoffPolicy(
            initial_seconds=0.5, multiplier=2.0, max_seconds=8.0, jitter_ratio=0.25
        )

    def test_delay_grows_exponentially_and_is_capped(self, policy: BackoffPolicy) -> None:
        assert [policy.base_delay(n) for n in range(1, 8)] == [0.5, 1.0, 2.0, 4.0, 8.0, 8.0, 8.0]

    def test_jitter_stays_within_the_configured_ratio(self, policy: BackoffPolicy) -> None:
        delays = [policy.delay_for(3, rng=random.Random(seed)) for seed in range(500)]
        assert all(1.5 <= delay <= 2.5 for delay in delays)
        assert len(set(delays)) > 1, "jitter must actually vary the delay"

    def test_delay_is_never_negative(self, policy: BackoffPolicy) -> None:
        assert all(
            policy.delay_for(attempt, rng=random.Random(seed)) >= 0
            for attempt in range(1, 9)
            for seed in range(100)
        )

    def test_server_retry_after_is_honoured_and_never_undercut(self, policy: BackoffPolicy) -> None:
        """Jittering a Retry-After downward would retry sooner than the server instructed."""
        delays = [
            policy.delay_for(1, retry_after_seconds=3.0, rng=random.Random(seed))
            for seed in range(300)
        ]
        assert all(delay >= 3.0 for delay in delays)

    def test_retry_after_is_capped_so_a_wait_cannot_be_unbounded(
        self, policy: BackoffPolicy
    ) -> None:
        delay = policy.delay_for(1, retry_after_seconds=9999, rng=random.Random(1))
        assert delay <= 8.0 * (1 + 0.25)


class TestRetryExecutor:
    @pytest.fixture
    def executor(
        self, resilience_settings: ResilienceSettings, instant_sleeper: Any
    ) -> RetryExecutor:
        return RetryExecutor(
            RetryPolicy.from_settings(resilience_settings),
            sleeper=instant_sleeper,
            rng=random.Random(7),
        )

    async def test_transient_failure_is_retried_then_succeeds(
        self, executor: RetryExecutor, recorded_sleeps: list[float]
    ) -> None:
        outcome = await executor.execute(
            failing_then(
                RateLimitedError(service="dfs", detail="slow down"),
                RateLimitedError(service="dfs", detail="slow down"),
                {"ok": True},
            ),
            service="dfs",
        )

        assert outcome.value == {"ok": True}
        assert outcome.attempts == 3
        assert outcome.retry_count == 2
        assert len(recorded_sleeps) == 2

    async def test_a_server_supplied_retry_after_drives_the_first_delay(
        self, executor: RetryExecutor, recorded_sleeps: list[float]
    ) -> None:
        await executor.execute(
            failing_then(
                RateLimitedError(service="dfs", detail="429", retry_after_seconds=2.0),
                {"ok": True},
            ),
            service="dfs",
        )

        assert recorded_sleeps[0] >= 2.0

    async def test_non_retryable_failure_is_not_retried(
        self, executor: RetryExecutor, recorded_sleeps: list[float]
    ) -> None:
        with pytest.raises(AuthenticationError):
            await executor.execute(
                failing_then(
                    AuthenticationError(service="dfs", detail="bad creds", status_code=401),
                    {"never": "reached"},
                ),
                service="dfs",
            )

        assert recorded_sleeps == [], "a non-retryable error must not cost a backoff wait"

    async def test_exhausted_retries_raise_a_terminal_error_carrying_the_cause(
        self, executor: RetryExecutor
    ) -> None:
        with pytest.raises(RetryExhaustedError) as caught:
            await executor.execute(
                failing_then(*[UpstreamServerError(service="dfs", detail="down")] * 5),
                service="dfs",
            )

        assert caught.value.attempts == 3
        assert isinstance(caught.value.cause, UpstreamServerError)
        assert not isinstance(caught.value, RetryableError), (
            "an exhausted retry must be terminal, or an outer loop would retry it again"
        )


class TestCircuitBreaker:
    @pytest.fixture
    def clock(self) -> list[float]:
        return [1000.0]

    @pytest.fixture
    def breaker(self, clock: list[float]) -> CircuitBreaker:
        return CircuitBreaker(
            name="dfs",
            policy=CircuitBreakerPolicy(
                failure_threshold=3, reset_seconds=30.0, half_open_successes=1
            ),
            clock=lambda: clock[0],
        )

    @staticmethod
    async def _fail() -> None:
        raise UpstreamServerError(service="dfs", detail="down", status_code=503)

    @staticmethod
    async def _succeed() -> str:
        return "ok"

    async def test_opens_after_the_configured_consecutive_failures(
        self, breaker: CircuitBreaker
    ) -> None:
        for _ in range(2):
            with pytest.raises(UpstreamServerError):
                await breaker.execute(self._fail)
            assert breaker.state is CircuitState.CLOSED

        with pytest.raises(UpstreamServerError):
            await breaker.execute(self._fail)
        assert breaker.state is CircuitState.OPEN

    async def test_an_open_circuit_rejects_without_invoking_the_operation(
        self, breaker: CircuitBreaker
    ) -> None:
        for _ in range(3):
            with pytest.raises(UpstreamServerError):
                await breaker.execute(self._fail)

        invocations = 0

        async def counted() -> str:
            nonlocal invocations
            invocations += 1
            return "should not run"

        with pytest.raises(CircuitOpenError):
            await breaker.execute(counted)
        assert invocations == 0

    async def test_recovers_through_half_open_after_the_reset_window(
        self, breaker: CircuitBreaker, clock: list[float]
    ) -> None:
        for _ in range(3):
            with pytest.raises(UpstreamServerError):
                await breaker.execute(self._fail)

        clock[0] += 31.0
        assert await breaker.execute(self._succeed) == "ok"
        assert breaker.state is CircuitState.CLOSED

    async def test_a_failed_probe_reopens_immediately(
        self, breaker: CircuitBreaker, clock: list[float]
    ) -> None:
        for _ in range(3):
            with pytest.raises(UpstreamServerError):
                await breaker.execute(self._fail)

        clock[0] += 31.0
        with pytest.raises(UpstreamServerError):
            await breaker.execute(self._fail)
        assert breaker.state is CircuitState.OPEN, "must not linger in half-open"

    async def test_client_errors_do_not_trip_the_breaker(self, breaker: CircuitBreaker) -> None:
        """A 400 means our request is wrong; the dependency is answering correctly."""

        async def bad_request() -> None:
            raise BadRequestError(service="dfs", detail="malformed", status_code=400)

        for _ in range(10):
            with pytest.raises(BadRequestError):
                await breaker.execute(bad_request)

        assert breaker.state is CircuitState.CLOSED

    async def test_an_open_circuit_does_not_reopen_and_extend_its_own_window(
        self, breaker: CircuitBreaker, clock: list[float]
    ) -> None:
        """Calls already in flight when the threshold is crossed must not push the deadline.

        Each late failure used to call ``_open()`` again, resetting ``opened_at`` - so under
        concurrency the circuit stayed open longer than configured and the log showed one
        outage as many separate openings.
        """
        for _ in range(3):
            with pytest.raises(UpstreamServerError):
                await breaker.execute(self._fail)
        assert breaker.state is CircuitState.OPEN

        # More failures land while open (they bypass admission via a direct record).
        for _ in range(5):
            await breaker._record_failure()

        # The reset window is measured from the original opening, not the latest failure.
        clock[0] += 31.0
        assert await breaker.execute(self._succeed) == "ok"
        assert breaker.state is CircuitState.CLOSED

    async def test_only_one_probe_is_admitted_through_a_half_open_circuit(
        self, clock: list[float]
    ) -> None:
        """Concurrent fan-out must not stampede a recovering dependency."""
        breaker = CircuitBreaker(
            name="dfs",
            policy=CircuitBreakerPolicy(
                failure_threshold=1, reset_seconds=5.0, half_open_successes=1
            ),
            clock=lambda: clock[0],
        )
        with pytest.raises(UpstreamServerError):
            await breaker.execute(self._fail)
        clock[0] += 6.0

        admitted = 0

        async def probe() -> str:
            nonlocal admitted
            admitted += 1
            await asyncio.sleep(0)
            return "ok"

        results = await asyncio.gather(
            *(breaker.execute(probe) for _ in range(8)), return_exceptions=True
        )

        assert admitted == 1
        assert sum(1 for r in results if isinstance(r, CircuitOpenError)) == 7

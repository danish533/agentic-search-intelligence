"""The live HTTP transport and the client that wraps it.

Exercised through ``respx`` at the socket boundary, so the real request construction, real
classification, real retry and real circuit breaker all run - only the network is replaced.
"""

from __future__ import annotations

import base64
import random
from typing import Any

import httpx
import pytest
import respx
from pydantic import SecretStr

from sightline.application.ports.errors import (
    AuthenticationError,
    BadRequestError,
    RetryExhaustedError,
    UpstreamContractError,
)
from sightline.config.settings import DataForSEOMode, DataForSEOSettings, ResilienceSettings
from sightline.infrastructure.dataforseo.client import DataForSEOClient
from sightline.infrastructure.dataforseo.endpoints import ORGANIC_SERP
from sightline.infrastructure.dataforseo.transport import HttpxTransport
from sightline.infrastructure.resilience.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerPolicy,
)
from sightline.infrastructure.resilience.retry import RetryExecutor, RetryPolicy
from sightline.observability.metrics import RunMetrics, metrics_scope

BASE_URL = "https://api.dataforseo.com"
SERP_URL = f"{BASE_URL}{ORGANIC_SERP.path}"

OK_BODY = {
    "status_code": 20000,
    "status_message": "Ok.",
    "tasks": [{"status_code": 20000, "status_message": "Ok.", "result": [{"items": []}]}],
}


@pytest.fixture
def live_settings() -> DataForSEOSettings:
    return DataForSEOSettings(
        mode=DataForSEOMode.LIVE,
        login=SecretStr("test-login"),
        password=SecretStr("test-password"),
    )


@pytest.fixture
def live_client(live_settings: DataForSEOSettings, instant_sleeper: Any) -> DataForSEOClient:
    settings = ResilienceSettings(max_attempts=3)
    return DataForSEOClient(
        settings=live_settings,
        transport=HttpxTransport(live_settings),
        retry_executor=RetryExecutor(
            RetryPolicy.from_settings(settings),
            sleeper=instant_sleeper,
            rng=random.Random(1),
        ),
        circuit_breaker=CircuitBreaker(
            name="dataforseo", policy=CircuitBreakerPolicy.from_settings(settings)
        ),
    )


class TestRequestConstruction:
    @respx.mock
    async def test_credentials_are_sent_as_http_basic_auth(
        self, live_client: DataForSEOClient
    ) -> None:
        route = respx.post(SERP_URL).mock(return_value=httpx.Response(200, json=OK_BODY))

        await live_client.execute(ORGANIC_SERP, {"keyword": "seo"})

        header = route.calls.last.request.headers["authorization"]
        assert header.startswith("Basic ")
        decoded = base64.b64decode(header.removeprefix("Basic ")).decode()
        assert decoded == "test-login:test-password"

    @respx.mock
    async def test_the_task_is_sent_as_a_single_element_array(
        self, live_client: DataForSEOClient
    ) -> None:
        """DataForSEO expects an array of tasks, not a bare object."""
        import json

        route = respx.post(SERP_URL).mock(return_value=httpx.Response(200, json=OK_BODY))

        await live_client.execute(ORGANIC_SERP, {"keyword": "best seo tools", "depth": 10})

        payload = json.loads(route.calls.last.request.content)
        assert isinstance(payload, list)
        assert payload == [{"keyword": "best seo tools", "depth": 10}]

    @respx.mock
    async def test_the_result_reports_it_did_not_come_from_the_mock(
        self, live_client: DataForSEOClient
    ) -> None:
        respx.post(SERP_URL).mock(return_value=httpx.Response(200, json=OK_BODY))

        result = await live_client.execute(ORGANIC_SERP, {"keyword": "seo"})

        assert result.from_mock is False
        assert result.endpoint == ORGANIC_SERP.path
        assert result.duration_ms >= 0


class TestResponseHandling:
    @respx.mock
    async def test_a_non_json_body_is_a_contract_violation_not_a_transient_failure(
        self, live_client: DataForSEOClient, recorded_sleeps: list[float]
    ) -> None:
        """An identical retry returns the identical unparseable bytes."""
        respx.post(SERP_URL).mock(
            return_value=httpx.Response(200, content=b"<html>maintenance</html>")
        )

        with pytest.raises(UpstreamContractError):
            await live_client.execute(ORGANIC_SERP, {"keyword": "seo"})
        assert recorded_sleeps == []

    @respx.mock
    async def test_a_json_array_body_is_rejected(self, live_client: DataForSEOClient) -> None:
        respx.post(SERP_URL).mock(return_value=httpx.Response(200, json=[1, 2, 3]))

        with pytest.raises(UpstreamContractError):
            await live_client.execute(ORGANIC_SERP, {"keyword": "seo"})

    @respx.mock
    async def test_an_in_body_auth_failure_inside_http_200_is_caught(
        self, live_client: DataForSEOClient, recorded_sleeps: list[float]
    ) -> None:
        """DataForSEO answers 200 for application-level failures."""
        respx.post(SERP_URL).mock(
            return_value=httpx.Response(
                200, json={"status_code": 40100, "status_message": "Unauthorized.", "tasks": []}
            )
        )

        with pytest.raises(AuthenticationError):
            await live_client.execute(ORGANIC_SERP, {"keyword": "seo"})
        assert recorded_sleeps == [], "an auth failure must never be retried"


class TestRetryOverRealHttp:
    @respx.mock
    async def test_a_429_then_success_recovers_and_honours_retry_after(
        self, live_client: DataForSEOClient, recorded_sleeps: list[float]
    ) -> None:
        respx.post(SERP_URL).mock(
            side_effect=[
                httpx.Response(429, headers={"Retry-After": "2"}, json={"message": "slow down"}),
                httpx.Response(200, json=OK_BODY),
            ]
        )

        result = await live_client.execute(ORGANIC_SERP, {"keyword": "seo"})

        assert result.attempts == 2
        assert recorded_sleeps[0] >= 2.0

    @respx.mock
    async def test_a_connection_error_is_retried(self, live_client: DataForSEOClient) -> None:
        respx.post(SERP_URL).mock(
            side_effect=[
                httpx.ConnectError("connection reset"),
                httpx.Response(200, json=OK_BODY),
            ]
        )

        result = await live_client.execute(ORGANIC_SERP, {"keyword": "seo"})
        assert result.attempts == 2

    @respx.mock
    async def test_a_persistent_5xx_exhausts_the_budget_and_records_the_attempts(
        self, live_client: DataForSEOClient
    ) -> None:
        route = respx.post(SERP_URL).mock(return_value=httpx.Response(503, json={"e": 1}))
        metrics = RunMetrics()

        with metrics_scope(metrics), pytest.raises(RetryExhaustedError) as caught:
            await live_client.execute(ORGANIC_SERP, {"keyword": "seo"})

        assert caught.value.attempts == 3
        assert route.call_count == 3
        assert len(metrics.api_calls) == 1, "three attempts are one logical API call"
        assert metrics.total_retries == 2

    @respx.mock
    async def test_a_400_is_not_retried(
        self, live_client: DataForSEOClient, recorded_sleeps: list[float]
    ) -> None:
        route = respx.post(SERP_URL).mock(return_value=httpx.Response(400, json={"e": 1}))

        with pytest.raises(BadRequestError):
            await live_client.execute(ORGANIC_SERP, {"keyword": "seo"})

        assert route.call_count == 1
        assert recorded_sleeps == []


class TestTimeouts:
    def test_connect_and_read_timeouts_are_configured_separately(
        self, live_settings: DataForSEOSettings
    ) -> None:
        """Spec S3.5: sane timeouts on all external calls, and never unbounded."""
        transport = HttpxTransport(live_settings)
        timeout = transport._client.timeout

        assert timeout.connect == live_settings.connect_timeout_seconds
        assert timeout.read == live_settings.read_timeout_seconds
        assert timeout.connect is not None
        assert timeout.read is not None

    def test_the_sandbox_host_is_used_in_sandbox_mode(self) -> None:
        sandbox = DataForSEOSettings(
            mode=DataForSEOMode.SANDBOX,
            login=SecretStr("l"),
            password=SecretStr("p"),
        )
        assert sandbox.base_url == sandbox.sandbox_base_url
        assert HttpxTransport(sandbox).mode_label == "sandbox"

    def test_mock_mode_requires_no_credentials(self) -> None:
        assert DataForSEOSettings(mode=DataForSEOMode.MOCK).requires_credentials is False
        assert DataForSEOSettings(mode=DataForSEOMode.LIVE).requires_credentials is True

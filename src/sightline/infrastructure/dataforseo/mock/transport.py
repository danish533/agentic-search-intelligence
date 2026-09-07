"""Mock DataForSEO transport.

Substitutes for the network while leaving every layer above it untouched: the client still
classifies the in-body status, still retries, still trips the circuit breaker, still records
metrics. That is what makes ``DATAFORSEO_MODE=mock`` a faithful default rather than a bypass.

It also carries the **fault injection** seam. Spec S5 requires a test proving "one simulated
API failure with successful retry/fallback"; queueing faults here reproduces a 429 storm, a
5xx outage or a connection reset deterministically, at the transport boundary, exactly where a
real one would occur.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from sightline.config.settings import DataForSEOSettings
from sightline.infrastructure.dataforseo.mock import fixtures
from sightline.infrastructure.dataforseo.transport import TransportResponse
from sightline.observability.logging import get_logger

_logger = get_logger(__name__)

_SERP_PATH_MARKER: Final = "/serp/google/organic"
_LLM_PATH_MARKER: Final = "/ai_optimization/"
_KEYWORDS_PATH_MARKER: Final = "/keywords_data/"


@dataclass(frozen=True, slots=True)
class MockFault:
    """A scripted failure to return instead of a fixture.

    Exactly one of ``exception`` (transport-level: connection reset, timeout) or
    ``status_code``/``body`` (HTTP or application-level) describes the failure.
    """

    status_code: int = 200
    body: Mapping[str, Any] | None = None
    headers: Mapping[str, str] = field(default_factory=dict)
    exception: Exception | None = None

    @classmethod
    def rate_limited(cls, *, retry_after_seconds: float | None = None) -> MockFault:
        """An HTTP 429, optionally carrying a ``Retry-After`` header."""
        headers = (
            {"retry-after": str(retry_after_seconds)} if retry_after_seconds is not None else {}
        )
        return cls(
            status_code=429,
            body={"status_code": 42900, "status_message": "Rate limit exceeded."},
            headers=headers,
        )

    @classmethod
    def server_error(cls, status_code: int = 503) -> MockFault:
        return cls(
            status_code=status_code,
            body={"status_code": 50000, "status_message": "Internal Error."},
        )

    @classmethod
    def auth_failure(cls) -> MockFault:
        """An application-level auth failure: HTTP 200 with an in-body 401xx code."""
        return cls(
            status_code=200,
            body={"status_code": 40100, "status_message": "Unauthorized.", "tasks": []},
        )

    @classmethod
    def transport_error(cls, exception: Exception) -> MockFault:
        return cls(exception=exception)


class MockTransport:
    """Serves deterministic fixtures, with an optional queue of scripted faults."""

    def __init__(
        self,
        settings: DataForSEOSettings,
        *,
        faults: Sequence[MockFault] = (),
    ) -> None:
        self._settings = settings
        self._faults: deque[MockFault] = deque(faults)
        self._call_count = 0

    @property
    def mode_label(self) -> str:
        return "mock"

    @property
    def is_mock(self) -> bool:
        return True

    @property
    def call_count(self) -> int:
        """Total transport calls made, including those that returned a fault."""
        return self._call_count

    def queue_faults(self, *faults: MockFault) -> None:
        """Script the next ``len(faults)`` calls to fail. Consumed in order, then exhausted."""
        self._faults.extend(faults)

    def reset(self) -> None:
        self._faults.clear()
        self._call_count = 0

    async def post(
        self,
        path: str,
        payload: Sequence[Mapping[str, Any]],
    ) -> TransportResponse:
        self._call_count += 1

        if self._faults:
            fault = self._faults.popleft()
            _logger.warning(
                "dataforseo.mock.fault_injected",
                endpoint=path,
                status_code=fault.status_code,
                exception=type(fault.exception).__name__ if fault.exception else None,
                remaining_faults=len(self._faults),
            )
            if fault.exception is not None:
                raise fault.exception
            return TransportResponse(
                status_code=fault.status_code,
                body=fault.body or {},
                headers=fault.headers,
            )

        task = payload[0] if payload else {}
        return TransportResponse(status_code=200, body=self._fixture_for(path, task))

    def _fixture_for(self, path: str, task: Mapping[str, Any]) -> Mapping[str, Any]:
        """Route a request path to the fixture that models that endpoint's response."""
        location_code = int(task.get("location_code", self._settings.default_location_code))
        language_code = str(task.get("language_code", self._settings.default_language_code))

        if _KEYWORDS_PATH_MARKER in path:
            keywords = task.get("keywords") or []
            return fixtures.keyword_metrics(
                keywords=[str(keyword) for keyword in keywords],
                location_code=location_code,
                language_code=language_code,
            )

        if _LLM_PATH_MARKER in path:
            llm_name = path.split("/")[3] if len(path.split("/")) > 3 else "chat_gpt"
            return fixtures.llm_visibility(
                prompt=str(task.get("user_prompt", "")),
                llm_name=llm_name,
            )

        if _SERP_PATH_MARKER in path:
            keyword = str(task.get("keyword", ""))
            if task.get("load_async_ai_overview"):
                return fixtures.ai_overview(
                    keyword=keyword,
                    location_code=location_code,
                    language_code=language_code,
                )
            return fixtures.organic_serp(
                keyword=keyword,
                location_code=location_code,
                language_code=language_code,
                depth=int(task.get("depth", 10)),
            )

        # An unroutable path is a wiring bug, and must surface as one rather than as an
        # empty-but-successful response that the normalizer would silently accept.
        return {
            "status_code": 40400,
            "status_message": f"Mock transport has no fixture for path '{path}'.",
            "tasks": [],
        }

    async def aclose(self) -> None:
        """No-op: the mock holds no connections."""

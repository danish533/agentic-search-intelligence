"""DataForSEO transport.

The transport is the *only* thing that differs between ``mock``, ``sandbox`` and ``live``
modes. Authentication headers, payload construction, in-body status classification, retries,
backoff, the circuit breaker and metrics all live above it in
:class:`~sightline.infrastructure.dataforseo.client.DataForSEOClient`, so a run in mock mode
exercises the identical code path a live run does - which is what makes mock mode a credible
default rather than a bypass.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any, Protocol, Self

import httpx

from sightline.application.ports.errors import UpstreamContractError
from sightline.config.settings import DataForSEOSettings

SERVICE_NAME = "dataforseo"


@dataclass(frozen=True, slots=True)
class TransportResponse:
    """A raw HTTP outcome: status line, headers and decoded body, nothing interpreted.

    Headers are carried because the retry path needs ``Retry-After`` from a 429, and losing
    it here would silently downgrade rate-limit handling to blind exponential backoff.
    """

    status_code: int
    body: Mapping[str, Any]
    headers: Mapping[str, str] = field(default_factory=dict)


class DataForSEOTransport(Protocol):
    """Sends one request. Knows nothing about retries, classification or metrics."""

    @property
    def mode_label(self) -> str:
        """Human-readable transport mode, surfaced in logs and the run report."""
        ...

    @property
    def is_mock(self) -> bool: ...

    async def post(
        self,
        path: str,
        payload: Sequence[Mapping[str, Any]],
    ) -> TransportResponse:
        """POST a DataForSEO task array and return the decoded response."""
        ...

    async def aclose(self) -> None: ...


class HttpxTransport:
    """Real HTTP transport for ``sandbox`` and ``live`` modes.

    Timeouts are explicit and separate (spec S3.5: "sane timeouts on all external calls"): a
    connect timeout bounds how long we wait for a dead host, while a longer read timeout
    accommodates DataForSEO's live endpoints, which legitimately take seconds to answer.
    """

    def __init__(
        self, settings: DataForSEOSettings, *, client: httpx.AsyncClient | None = None
    ) -> None:
        self._settings = settings
        self._client = client if client is not None else self._build_client(settings)

    @staticmethod
    def _build_client(settings: DataForSEOSettings) -> httpx.AsyncClient:
        login = settings.login.get_secret_value() if settings.login else ""
        password = settings.password.get_secret_value() if settings.password else ""
        return httpx.AsyncClient(
            base_url=settings.base_url,
            auth=httpx.BasicAuth(login, password),
            timeout=httpx.Timeout(
                connect=settings.connect_timeout_seconds,
                read=settings.read_timeout_seconds,
                write=settings.read_timeout_seconds,
                pool=settings.connect_timeout_seconds,
            ),
            headers={"Content-Type": "application/json"},
        )

    @property
    def mode_label(self) -> str:
        return self._settings.mode.value

    @property
    def is_mock(self) -> bool:
        return False

    async def post(
        self,
        path: str,
        payload: Sequence[Mapping[str, Any]],
    ) -> TransportResponse:
        response = await self._client.post(path, json=list(payload))

        try:
            body = response.json()
        except ValueError as exc:
            # A non-JSON body is a contract violation, not a transport fault: an identical
            # retry returns the identical unparseable bytes.
            raise UpstreamContractError(
                service=SERVICE_NAME,
                detail=f"response body was not valid JSON: {exc}",
                endpoint=path,
                status_code=response.status_code,
            ) from exc

        if not isinstance(body, Mapping):
            raise UpstreamContractError(
                service=SERVICE_NAME,
                detail=f"expected a JSON object, got {type(body).__name__}",
                endpoint=path,
                status_code=response.status_code,
            )

        return TransportResponse(
            status_code=response.status_code,
            body=body,
            headers={key.lower(): value for key, value in response.headers.items()},
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()

"""The HTTP error envelope, including the path that runs outside the middleware.

The catch-all handler is registered on Starlette's ``ServerErrorMiddleware``, which sits
*outside* the correlation middleware. By the time it builds a 500 the context var has already
unwound - so the one response that tells a user to "quote the correlation id" was the one
response that did not carry one. These tests pin that shut.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from sightline.api.app import create_app
from sightline.composition.container import Container
from sightline.observability.correlation import CORRELATION_ID_HEADER


class StubContainer:
    """Satisfies the container protocol; no route under test uses it."""

    def __init__(self) -> None:
        self.register_profile = None
        self.get_profile = None
        self.run_pipeline = None
        self.list_queries = None
        self.list_recommendations = None
        self.recheck_query = None

    @property
    def environment(self) -> str:
        return "test"

    @property
    def llm_provider_name(self) -> str:
        return "fake"

    @property
    def llm_model(self) -> str:
        return "stub"

    @property
    def dataforseo_mode(self) -> str:
        return "mock"

    async def check_database(self) -> bool:
        return True


@pytest.fixture
def app_with_failing_route() -> FastAPI:
    app = create_app(container=StubContainer())  # type: ignore[arg-type]

    @app.get("/boom")
    async def boom() -> dict[str, Any]:
        raise RuntimeError("connection string postgresql://user:hunter2@db/app")

    return app


@pytest.fixture
async def client(app_with_failing_route: FastAPI) -> Any:
    transport = httpx.ASGITransport(app=app_with_failing_route, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


class TestUnhandledErrorEnvelope:
    async def test_an_unhandled_error_still_carries_a_correlation_id(
        self, client: httpx.AsyncClient
    ) -> None:
        """Regression: the 500 body told users to quote an id it did not include."""
        response = await client.get("/boom")
        body = response.json()

        assert response.status_code == 500
        assert body["correlation_id"], "the response asks for an id it must therefore provide"
        assert response.headers.get(CORRELATION_ID_HEADER)

    async def test_the_body_and_header_ids_match(self, client: httpx.AsyncClient) -> None:
        response = await client.get("/boom")
        assert response.json()["correlation_id"] == response.headers[CORRELATION_ID_HEADER]

    async def test_an_inbound_correlation_id_is_preserved_through_a_500(
        self, client: httpx.AsyncClient
    ) -> None:
        response = await client.get("/boom", headers={CORRELATION_ID_HEADER: "caller-trace-9"})
        assert response.json()["correlation_id"] == "caller-trace-9"

    async def test_no_internal_detail_leaks_into_the_response(
        self, client: httpx.AsyncClient
    ) -> None:
        """The raised message contains a DSN with a password. None of it may reach the client."""
        response = await client.get("/boom")
        rendered = response.text

        assert "hunter2" not in rendered
        assert "postgresql://" not in rendered
        assert "RuntimeError" not in rendered
        assert "Traceback" not in rendered

    async def test_the_envelope_shape_is_the_same_as_every_other_error(
        self, client: httpx.AsyncClient
    ) -> None:
        body = (await client.get("/boom")).json()
        assert set(body) == {"error", "message", "detail", "correlation_id"}


class TestHealthReportsDegradedRatherThanFailing:
    """A readiness probe must answer, not raise.

    ``check_database`` originally caught only ``SQLAlchemyError``. A refused connection raises
    ``ConnectionRefusedError``, which is an ``OSError`` - so the real failure escaped and
    /health returned 500, making the ``degraded`` / ``down`` states the endpoint models
    unreachable. An orchestrator polling it got an opaque error instead of a readiness signal.
    """

    async def test_a_refused_connection_is_not_a_sqlalchemy_error(self) -> None:
        """Pins the reason the original except clause was wrong."""
        from sqlalchemy.exc import SQLAlchemyError

        assert issubclass(ConnectionRefusedError, OSError)
        assert not issubclass(ConnectionRefusedError, SQLAlchemyError)

    async def test_check_database_returns_false_when_nothing_is_listening(self) -> None:
        from sightline.config.settings import DatabaseSettings
        from sightline.infrastructure.persistence.session import create_database_engine

        engine = create_database_engine(
            DatabaseSettings(url="postgresql+asyncpg://x:x@localhost:1/none")  # type: ignore[arg-type]
        )
        container = _ContainerWithEngine(engine)
        try:
            assert await container.check_database() is False
        finally:
            await engine.dispose()

    async def test_health_returns_200_degraded_when_the_database_is_down(self) -> None:
        app = create_app(container=_UnhealthyContainer())  # type: ignore[arg-type]
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/health")

        assert response.status_code == 200, "a probe reports state; it does not fail"
        body = response.json()
        assert body["status"] == "degraded"
        assert body["database"] == "down"


class _ContainerWithEngine:
    """Exercises the real ``check_database`` against a supplied engine.

    The concrete container is a frozen dataclass with many fields, none of which this check
    touches; borrowing the method is simpler than constructing one and keeps the test on the
    behaviour under examination.
    """

    def __init__(self, engine: Any) -> None:
        self._database = engine

    async def check_database(self) -> bool:
        return await Container.check_database(self)  # type: ignore[arg-type]


class _UnhealthyContainer(StubContainer):
    async def check_database(self) -> bool:
        return False

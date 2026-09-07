"""Shared test fixtures.

Two principles hold throughout the suite:

**Offline and deterministic.** Every unit test runs against the ``fake`` LLM provider and the
mock DataForSEO transport, with the sleeper and RNG injected. No test needs a key, a network,
or wall-clock time to pass (CLAUDE.md R7).

**Integration tests skip rather than fail** when PostgreSQL is unreachable, so ``make test``
works on a fresh clone. Run ``make db`` first to include them.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from sightline.agents.dependencies import NodeDependencies
from sightline.agents.engine import LangGraphPipelineEngine
from sightline.application.ports.unit_of_work import UnitOfWork, UnitOfWorkFactory
from sightline.composition.demo_responders import register_demo_responders
from sightline.config.settings import (
    DatabaseSettings,
    DataForSEOMode,
    DataForSEOSettings,
    LLMProviderName,
    LLMSettings,
    ObservabilitySettings,
    PipelineSettings,
    ResilienceSettings,
    Settings,
)
from sightline.domain.entities.profile import Profile
from sightline.infrastructure.dataforseo.client import DataForSEOClient
from sightline.infrastructure.dataforseo.mock.transport import MockFault, MockTransport
from sightline.infrastructure.dataforseo.provider import DataForSEOProvider
from sightline.infrastructure.llm.fake import FakeLLMProvider
from sightline.infrastructure.persistence.session import (
    create_database_engine,
    create_session_factory,
)
from sightline.infrastructure.persistence.unit_of_work import SqlAlchemyUnitOfWork
from sightline.infrastructure.resilience.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerPolicy,
)
from sightline.infrastructure.resilience.retry import RetryExecutor, RetryPolicy
from sightline.observability.logging import configure_logging
from sightline.tools.registry import ToolRegistry, default_tool_registry

#: A fixed seed everywhere jitter or fixture shaping is involved, so a failure is reproducible.
TEST_SEED = 20260906


@pytest.fixture(scope="session", autouse=True)
def _configure_logging() -> None:
    """Quiet, structured logging for the suite. Errors still surface."""
    configure_logging(ObservabilitySettings(log_format="console", log_level="ERROR"))


@pytest.fixture(autouse=True)
def _guard_against_live_providers(dataforseo_settings: DataForSEOSettings) -> None:
    """Fail loudly if a test would reach a real provider.

    A guard on top of the pinned fixtures. The suite must never need a key, spend credits, or
    vary with a developer's local ``.env`` (CLAUDE.md R7). Without this, setting
    ``LLM_PROVIDER=openai`` locally turned every run into a paid, networked, non-deterministic
    one - slowly, and without a single failing assertion to say so.
    """
    if dataforseo_settings.mode is not DataForSEOMode.MOCK:
        pytest.fail(f"test would use DataForSEO mode {dataforseo_settings.mode.value!r}")


@pytest.fixture
def resilience_settings() -> ResilienceSettings:
    return ResilienceSettings(max_attempts=3, backoff_initial_seconds=0.5, jitter_ratio=0.25)


@pytest.fixture
def dataforseo_settings() -> DataForSEOSettings:
    """Mock mode, pinned explicitly.

    ``DataForSEOSettings()`` reads ``.env``, so a developer whose local file says ``live``
    would silently point the entire suite at the real API - spending credits and making every
    assertion network-dependent. Constructor arguments override the environment.
    """
    return DataForSEOSettings(mode=DataForSEOMode.MOCK, login=None, password=None)


@pytest.fixture
def offline_llm_settings() -> LLMSettings:
    """The offline provider, pinned explicitly - never taken from the ambient environment."""
    return LLMSettings(
        provider=LLMProviderName.FAKE,
        openai_api_key=None,
        anthropic_api_key=None,
        google_api_key=None,
    )


@pytest.fixture
def pipeline_settings() -> PipelineSettings:
    return PipelineSettings(max_planned_queries=4, min_successful_retrievals=1)


@pytest.fixture
def recorded_sleeps() -> list[float]:
    """Collects the delays the retry executor would have slept for."""
    return []


@pytest.fixture
def instant_sleeper(recorded_sleeps: list[float]) -> Any:
    """A sleeper that records instead of sleeping, so retry timing is asserted, not waited on."""

    async def _sleep(seconds: float) -> None:
        recorded_sleeps.append(seconds)

    return _sleep


@pytest.fixture
def tool_registry() -> ToolRegistry:
    return default_tool_registry()


@pytest.fixture
def mock_transport(dataforseo_settings: DataForSEOSettings) -> MockTransport:
    return MockTransport(dataforseo_settings)


@pytest.fixture
def search_provider(
    dataforseo_settings: DataForSEOSettings,
    mock_transport: MockTransport,
    resilience_settings: ResilienceSettings,
    instant_sleeper: Any,
) -> DataForSEOProvider:
    client = DataForSEOClient(
        settings=dataforseo_settings,
        transport=mock_transport,
        retry_executor=RetryExecutor(
            RetryPolicy.from_settings(resilience_settings),
            sleeper=instant_sleeper,
            rng=random.Random(TEST_SEED),
        ),
        circuit_breaker=CircuitBreaker(
            name="dataforseo", policy=CircuitBreakerPolicy.from_settings(resilience_settings)
        ),
    )
    return DataForSEOProvider(client=client, settings=dataforseo_settings)


@pytest.fixture
def fake_llm(
    tool_registry: ToolRegistry, dataforseo_settings: DataForSEOSettings
) -> FakeLLMProvider:
    """The offline provider, with the demo responders the composition root also installs."""
    provider = FakeLLMProvider()
    register_demo_responders(provider, registry=tool_registry, settings=dataforseo_settings)
    return provider


@pytest.fixture
def node_dependencies(
    fake_llm: FakeLLMProvider,
    search_provider: DataForSEOProvider,
    tool_registry: ToolRegistry,
    pipeline_settings: PipelineSettings,
    dataforseo_settings: DataForSEOSettings,
) -> NodeDependencies:
    return NodeDependencies(
        llm=fake_llm,
        search_provider=search_provider,
        tools=tool_registry,
        pipeline=pipeline_settings,
        dataforseo=dataforseo_settings,
    )


@pytest.fixture
def pipeline_engine(node_dependencies: NodeDependencies) -> LangGraphPipelineEngine:
    return LangGraphPipelineEngine(node_dependencies)


@pytest.fixture
def profile() -> Profile:
    return Profile(
        name="Surfer SEO",
        domain="surferseo.com",
        industry="SEO Software",
        description="AI-powered SEO content optimization tool",
        competitors=("clearscope.io", "marketmuse.com", "frase.io"),
    )


@pytest.fixture
def server_error() -> MockFault:
    return MockFault.server_error()


@pytest.fixture
def rate_limited() -> MockFault:
    return MockFault.rate_limited(retry_after_seconds=0.1)


# ---------------------------------------------------------------------------------------
# Integration fixtures: PostgreSQL
# ---------------------------------------------------------------------------------------


#: Hosts a destructive test fixture may point at. Anything else is assumed to be shared.
_DISPOSABLE_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "db", "postgres"})

#: Database names a destructive test fixture may point at, plus any name ending in "_test".
_DISPOSABLE_DATABASES = frozenset({"sightline", "sightline_test"})


def pytest_sessionstart(session: pytest.Session) -> None:
    """Refuse to start if the target database is not obviously disposable.

    Checked here rather than in a fixture: ``pytest.exit`` from inside a fixture is converted
    into a setup error for every test that requests it, producing dozens of errors instead of
    one clear message. At session start it stops the run immediately, before anything can
    connect.
    """
    assert_database_is_disposable(DatabaseSettings())


def assert_database_is_disposable(settings: DatabaseSettings) -> None:
    """Refuse to run destructive fixtures against a database that is not obviously throwaway.

    The integration fixtures ``TRUNCATE`` before every test. Reading ``DATABASE_URL`` from the
    environment without checking it means ``DATABASE_URL=<staging> make test`` silently empties
    staging - and unlike a mis-set LLM key, which costs money, this is not recoverable.

    Convention #27 says test fixtures must not inherit the ambient environment for anything
    external. That was applied to the LLM and to DataForSEO but not to the database, which is
    the one dependency where the failure mode is destructive. This closes it.

    Uses ``pytest.exit`` rather than ``fail``: a run aimed at the wrong database should stop
    the session immediately, not report a failure and carry on to the next test.
    """
    # PostgresDsn is a MultiHostUrl in Pydantic v2, so hosts come from hosts(), not .host.
    # Every host must be disposable - a URL listing one safe replica and one shared one is
    # not safe.
    hosts = [str(entry.get("host") or "").lower() for entry in settings.url.hosts()]
    name = (settings.url.path or "").lstrip("/")
    hosts_ok = bool(hosts) and all(host in _DISPOSABLE_HOSTS for host in hosts)
    name_ok = name in _DISPOSABLE_DATABASES or name.endswith("_test")

    if not (hosts_ok and name_ok):
        host = ", ".join(hosts) or "<none>"
        pytest.exit(
            "\n\nREFUSING TO RUN: the test suite truncates its database before every test, "
            f"and DATABASE_URL points at host {host!r}, database {name!r}, which is not "
            "recognised as disposable.\n"
            f"  Allowed hosts     : {', '.join(sorted(_DISPOSABLE_HOSTS))}\n"
            f"  Allowed databases : {', '.join(sorted(_DISPOSABLE_DATABASES))}, "
            "or any name ending in '_test'\n"
            "  To run integration tests, point DATABASE_URL at a throwaway database "
            "(`make db` starts one).\n",
            returncode=2,
        )


@pytest.fixture(scope="session")
def database_settings() -> DatabaseSettings:
    """The database the integration tests use.

    Safety is enforced once at session start by :func:`pytest_sessionstart`, before any test
    or fixture can reach the database.
    """
    return DatabaseSettings()


@pytest.fixture(scope="session")
def database_available(database_settings: DatabaseSettings) -> bool:
    """Whether a usable, migrated database is reachable.

    Probes ``profiles`` rather than merely connecting: a running-but-unmigrated database would
    otherwise let every integration test fail with a confusing "relation does not exist".
    """

    async def _probe() -> bool:
        engine = create_database_engine(database_settings)
        try:
            async with engine.connect() as connection:
                await connection.execute(text("SELECT 1 FROM profiles LIMIT 1"))
        except Exception:  # any failure at all means the database is unusable here
            return False
        finally:
            await engine.dispose()
        return True

    return asyncio.run(_probe())


@pytest.fixture
def requires_database(database_available: bool) -> None:
    if not database_available:
        pytest.skip(
            "PostgreSQL is not reachable or not migrated. "
            "Run `make db && make migrate` to include integration tests."
        )


@pytest.fixture
async def database_engine(
    requires_database: None, database_settings: DatabaseSettings
) -> AsyncIterator[AsyncEngine]:
    engine = create_database_engine(database_settings)
    try:
        yield engine
    finally:
        await engine.dispose()


@pytest.fixture
async def clean_database(database_engine: AsyncEngine) -> AsyncIterator[None]:
    """Truncate before each test so ordering never matters and no test inherits state."""
    async with database_engine.begin() as connection:
        await connection.execute(text("TRUNCATE profiles CASCADE"))
    yield


@pytest.fixture
def unit_of_work_factory(database_engine: AsyncEngine, clean_database: None) -> UnitOfWorkFactory:
    session_factory = create_session_factory(database_engine)

    def factory() -> UnitOfWork:
        return SqlAlchemyUnitOfWork(session_factory)

    return factory


@pytest.fixture
def api_settings(
    database_settings: DatabaseSettings,
    offline_llm_settings: LLMSettings,
    pipeline_settings: PipelineSettings,
    dataforseo_settings: DataForSEOSettings,
    resilience_settings: ResilienceSettings,
) -> Settings:
    """Settings for an end-to-end application: offline LLM, mocked provider, real database."""
    return Settings(
        database=database_settings,
        llm=offline_llm_settings,
        dataforseo=dataforseo_settings,
        resilience=resilience_settings,
        pipeline=pipeline_settings,
        observability=ObservabilitySettings(log_level="ERROR", log_format="console"),
    )


@pytest.fixture
async def api_client(
    requires_database: None, clean_database: None, api_settings: Settings
) -> AsyncIterator[Any]:
    """A client bound to the real application, including its lifespan."""
    import httpx

    from sightline.composition.bootstrap import create_application

    app = create_application(api_settings)
    transport = httpx.ASGITransport(app=app)
    async with (
        httpx.AsyncClient(transport=transport, base_url="http://test", timeout=120) as client,
        app.router.lifespan_context(app),
    ):
        yield client


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def pytest_collection_modifyitems(items: list[pytest.Item]) -> Iterator[None] | None:
    """Tag everything under tests/integration and tests/e2e with the integration marker."""
    for item in items:
        path = str(item.path)
        if "/integration/" in path or "/e2e/" in path:
            item.add_marker(pytest.mark.integration)
    return None

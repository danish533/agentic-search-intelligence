"""Composition root.

The only module permitted to import from every layer (CLAUDE.md A3), and the only place any
concrete adapter is constructed. Everything else receives its collaborators through a port,
which is what the ``import-linter`` contracts enforce and what makes the whole system
testable by substitution.

Construction is eager and validated. A missing credential, an unreachable configuration, a
malformed graph - all surface here, when the process starts, rather than on a request an hour
later.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from sightline.agents.dependencies import NodeDependencies
from sightline.agents.engine import LangGraphPipelineEngine
from sightline.application.ports.unit_of_work import UnitOfWork, UnitOfWorkFactory
from sightline.application.use_cases.get_profile import GetProfileUseCase
from sightline.application.use_cases.list_queries import ListQueriesUseCase
from sightline.application.use_cases.list_recommendations import ListRecommendationsUseCase
from sightline.application.use_cases.recheck_query import RecheckQueryUseCase
from sightline.application.use_cases.register_profile import RegisterProfileUseCase
from sightline.application.use_cases.run_pipeline import RunPipelineUseCase
from sightline.composition.demo_responders import register_demo_responders
from sightline.config.settings import (
    DataForSEOMode,
    LLMProviderName,
    Settings,
    credential_is_present,
    get_settings,
)
from sightline.infrastructure.dataforseo.client import DataForSEOClient
from sightline.infrastructure.dataforseo.mock.transport import MockTransport
from sightline.infrastructure.dataforseo.provider import DataForSEOProvider
from sightline.infrastructure.dataforseo.transport import DataForSEOTransport, HttpxTransport
from sightline.infrastructure.llm.factory import build_llm_provider
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
from sightline.observability.logging import configure_logging, get_logger
from sightline.tools.registry import default_tool_registry

_logger = get_logger(__name__)


class MissingDataForSEOCredentialsError(RuntimeError):
    """Live or sandbox mode was selected without credentials."""

    def __init__(self, mode: DataForSEOMode) -> None:
        super().__init__(
            f"DATAFORSEO_MODE={mode.value} requires DATAFORSEO_LOGIN and "
            "DATAFORSEO_PASSWORD. Use DATAFORSEO_MODE=mock to run without credentials."
        )


@dataclass(frozen=True, slots=True)
class Container:
    """Everything the API layer needs, fully wired."""

    settings: Settings
    register_profile: RegisterProfileUseCase
    get_profile: GetProfileUseCase
    run_pipeline: RunPipelineUseCase
    list_queries: ListQueriesUseCase
    list_recommendations: ListRecommendationsUseCase
    recheck_query: RecheckQueryUseCase

    llm_provider_name: str
    llm_model: str
    dataforseo_mode: str

    _database: AsyncEngine
    _search_client: DataForSEOClient

    @property
    def environment(self) -> str:
        """Deployment environment name. Part of the ``UseCaseContainer`` contract."""
        return self.settings.app.environment.value

    async def check_database(self) -> bool:
        """Whether the database answers. Exposed so the health endpoint need not reach into
        the engine directly - the container owns that resource and answers for it."""
        try:
            async with self._database.connect() as connection:
                await connection.execute(text("SELECT 1"))
        except Exception as exc:
            # Deliberately broad. A refused connection raises ConnectionRefusedError, which is
            # an OSError and not a SQLAlchemyError - so catching the driver's exception family
            # alone let the real failure escape and turned /health into a 500, hiding the
            # "degraded" state the endpoint exists to report. A readiness probe answers up or
            # down; any failure is "down", and it must never raise.
            _logger.warning(
                "health.database_unreachable",
                error_type=type(exc).__name__,
                error=str(exc),
            )
            return False
        return True

    async def aclose(self) -> None:
        """Release the connection pool and HTTP client. Called on application shutdown."""
        await self._search_client.aclose()
        await self._database.dispose()
        _logger.info("container.closed")


def _build_transport(settings: Settings) -> DataForSEOTransport:
    if settings.dataforseo.mode is DataForSEOMode.MOCK:
        return MockTransport(settings.dataforseo)

    if not (
        credential_is_present(settings.dataforseo.login)
        and credential_is_present(settings.dataforseo.password)
    ):
        raise MissingDataForSEOCredentialsError(settings.dataforseo.mode)

    return HttpxTransport(settings.dataforseo)


def build_container(settings: Settings | None = None) -> Container:
    """Construct the application. Called once, by the application factory."""
    resolved = settings or get_settings()
    configure_logging(resolved.observability)

    database = create_database_engine(resolved.database)
    session_factory = create_session_factory(database)

    def unit_of_work() -> UnitOfWork:
        return SqlAlchemyUnitOfWork(session_factory)

    uow_factory: UnitOfWorkFactory = unit_of_work

    tools = default_tool_registry()
    llm = build_llm_provider(resolved.llm)

    # The offline provider gets realistic responders so that a run with no API key produces a
    # coherent report rather than schema-valid filler. Registered here because the composition
    # root is the only place allowed to know both the agent contracts and the fake adapter.
    if isinstance(llm, FakeLLMProvider):
        register_demo_responders(llm, registry=tools, settings=resolved.dataforseo)

    search_client = DataForSEOClient(
        settings=resolved.dataforseo,
        transport=_build_transport(resolved),
        retry_executor=RetryExecutor(RetryPolicy.from_settings(resolved.resilience)),
        circuit_breaker=CircuitBreaker(
            name="dataforseo",
            policy=CircuitBreakerPolicy.from_settings(resolved.resilience),
        ),
    )

    pipeline_engine = LangGraphPipelineEngine(
        NodeDependencies(
            llm=llm,
            search_provider=DataForSEOProvider(client=search_client, settings=resolved.dataforseo),
            tools=tools,
            pipeline=resolved.pipeline,
            dataforseo=resolved.dataforseo,
        )
    )

    _logger.info(
        "container.built",
        environment=resolved.app.environment.value,
        llm_provider=llm.name,
        llm_model=llm.model,
        dataforseo_mode=search_client.mode_label,
        tools=list(tools.names),
    )

    return Container(
        settings=resolved,
        register_profile=RegisterProfileUseCase(uow_factory),
        get_profile=GetProfileUseCase(uow_factory),
        run_pipeline=RunPipelineUseCase(uow_factory, pipeline_engine),
        list_queries=ListQueriesUseCase(uow_factory),
        list_recommendations=ListRecommendationsUseCase(uow_factory),
        recheck_query=RecheckQueryUseCase(uow_factory, pipeline_engine),
        llm_provider_name=llm.name,
        llm_model=llm.model,
        dataforseo_mode=search_client.mode_label,
        _database=database,
        _search_client=search_client,
    )


def is_fake_llm(settings: Settings) -> bool:
    """Whether the configured provider is the offline stub. Reported by the health endpoint."""
    return settings.llm.provider is LLMProviderName.FAKE

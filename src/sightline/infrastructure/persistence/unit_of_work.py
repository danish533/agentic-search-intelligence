"""SQLAlchemy unit of work.

A DAG run writes a run record, its queries, its insights and its recommendations. Those are
one business transaction: a run that persisted its queries but not its report is a worse
outcome than a run that persisted nothing, because it looks complete and is not.

Every repository here is bound to the same session, so all four writes flush together and
commit together.
"""

from __future__ import annotations

from types import TracebackType
from typing import Self

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from sightline.application.ports.repositories import (
    DiscoveredQueryRepository,
    InsightRepository,
    PipelineRunRepository,
    ProfileRepository,
    RecommendationRepository,
)
from sightline.infrastructure.persistence.repositories.discovered_query import (
    SqlAlchemyDiscoveredQueryRepository,
)
from sightline.infrastructure.persistence.repositories.insight import SqlAlchemyInsightRepository
from sightline.infrastructure.persistence.repositories.pipeline_run import (
    SqlAlchemyPipelineRunRepository,
)
from sightline.infrastructure.persistence.repositories.profile import SqlAlchemyProfileRepository
from sightline.infrastructure.persistence.repositories.recommendation import (
    SqlAlchemyRecommendationRepository,
)
from sightline.observability.logging import get_logger

_logger = get_logger(__name__)


class SqlAlchemyUnitOfWork:
    """Transactional scope exposing every repository over one session."""

    # Annotated to the *port* types, not the concrete ones. Protocol attributes are matched
    # invariantly, so without these the attributes would infer as SqlAlchemy* classes and this
    # adapter would silently fail to satisfy the UnitOfWork protocol - a mismatch that only
    # surfaces where the protocol is actually required, far from here.
    profiles: ProfileRepository
    runs: PipelineRunRepository
    queries: DiscoveredQueryRepository
    insights: InsightRepository
    recommendations: RecommendationRepository

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory
        self._session: AsyncSession | None = None

    @property
    def session(self) -> AsyncSession:
        """The active session. Only valid inside the context manager."""
        if self._session is None:
            raise RuntimeError(
                "UnitOfWork used outside its context manager; enter it with 'async with'."
            )
        return self._session

    async def __aenter__(self) -> Self:
        self._session = self._session_factory()
        self.profiles = SqlAlchemyProfileRepository(self._session)
        self.runs = SqlAlchemyPipelineRunRepository(self._session)
        self.queries = SqlAlchemyDiscoveredQueryRepository(self._session)
        self.insights = SqlAlchemyInsightRepository(self._session)
        self.recommendations = SqlAlchemyRecommendationRepository(self._session)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Roll back and close.

        The rollback is unconditional and safe: a committed transaction has nothing left to
        roll back, so this only ever discards work a use case did not explicitly commit.
        That is the point - a use case that raises halfway through must not leave a partial
        run behind, and forgetting to commit must never be the same as committing.
        """
        session = self._session
        if session is None:  # pragma: no cover - __aexit__ without __aenter__
            return
        try:
            if exc is not None:
                _logger.warning(
                    "unit_of_work.rolled_back",
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
            await session.rollback()
        finally:
            await session.close()
            self._session = None

    async def commit(self) -> None:
        await self.session.commit()

    async def rollback(self) -> None:
        await self.session.rollback()

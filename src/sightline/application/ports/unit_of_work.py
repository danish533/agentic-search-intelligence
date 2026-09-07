"""Unit-of-work port.

A DAG run writes a run record, many queries, many insights and many recommendations. Those
writes are one business transaction: a run that persists its queries but not its report is
worse than a run that persisted nothing. The unit of work makes that atomicity explicit and
gives every repository in a use case the same session.
"""

from __future__ import annotations

from collections.abc import Callable
from types import TracebackType
from typing import Protocol, Self

from sightline.application.ports.repositories import (
    DiscoveredQueryRepository,
    InsightRepository,
    PipelineRunRepository,
    ProfileRepository,
    RecommendationRepository,
)


class UnitOfWork(Protocol):
    """Transactional scope exposing every repository bound to a single session.

    Exiting the context without an explicit :meth:`commit` rolls back. Commit is deliberately
    never implicit: a use case must state that its work is complete.
    """

    profiles: ProfileRepository
    runs: PipelineRunRepository
    queries: DiscoveredQueryRepository
    insights: InsightRepository
    recommendations: RecommendationRepository

    async def __aenter__(self) -> Self: ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...

    async def commit(self) -> None: ...

    async def rollback(self) -> None: ...


#: Use cases receive a factory, not a live unit of work, so each use case owns its own
#: transaction boundary instead of inheriting one from its caller.
UnitOfWorkFactory = Callable[[], UnitOfWork]

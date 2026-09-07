"""Use-case container port.

The API ring needs somewhere to obtain its use cases, but it must not import the composition
root to get them - that would invert the dependency rule and drag the entire infrastructure
ring in transitively (which is exactly what ``import-linter`` refused).

This protocol is the inversion. The API depends on *what it needs* - six use cases and a few
readiness facts - and the composition root supplies a concrete object that satisfies it. The
API never learns that SQLAlchemy, LangGraph or DataForSEO exist.
"""

from __future__ import annotations

from typing import Protocol

from sightline.application.use_cases.get_profile import GetProfileUseCase
from sightline.application.use_cases.list_queries import ListQueriesUseCase
from sightline.application.use_cases.list_recommendations import ListRecommendationsUseCase
from sightline.application.use_cases.recheck_query import RecheckQueryUseCase
from sightline.application.use_cases.register_profile import RegisterProfileUseCase
from sightline.application.use_cases.run_pipeline import RunPipelineUseCase


class UseCaseContainer(Protocol):
    """Everything the HTTP layer may ask of the wired application.

    Declared as read-only properties rather than plain attributes. A Protocol *variable* must
    be settable, which would exclude any immutable implementation - and the concrete container
    is a frozen dataclass precisely because nothing should rebind a use case at runtime.
    """

    @property
    def register_profile(self) -> RegisterProfileUseCase: ...

    @property
    def get_profile(self) -> GetProfileUseCase: ...

    @property
    def run_pipeline(self) -> RunPipelineUseCase: ...

    @property
    def list_queries(self) -> ListQueriesUseCase: ...

    @property
    def list_recommendations(self) -> ListRecommendationsUseCase: ...

    @property
    def recheck_query(self) -> RecheckQueryUseCase: ...

    @property
    def environment(self) -> str:
        """Deployment environment name, for the health response."""
        ...

    @property
    def llm_provider_name(self) -> str: ...

    @property
    def llm_model(self) -> str: ...

    @property
    def dataforseo_mode(self) -> str: ...

    async def check_database(self) -> bool:
        """Whether the persistence layer answers. Reported by the health endpoint."""
        ...

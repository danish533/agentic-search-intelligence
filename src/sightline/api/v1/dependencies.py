"""FastAPI dependency providers.

Routes depend on use cases, never on repositories, engines or sessions. Each provider does
nothing but pull an already-wired use case off the container built at startup - so a request
performs no construction, and a route cannot reach past the application ring even by accident.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from sightline.application.ports.container import UseCaseContainer
from sightline.application.use_cases.get_profile import GetProfileUseCase
from sightline.application.use_cases.list_queries import ListQueriesUseCase
from sightline.application.use_cases.list_recommendations import ListRecommendationsUseCase
from sightline.application.use_cases.recheck_query import RecheckQueryUseCase
from sightline.application.use_cases.register_profile import RegisterProfileUseCase
from sightline.application.use_cases.run_pipeline import RunPipelineUseCase


def get_container(request: Request) -> UseCaseContainer:
    """Retrieve the wired container placed on application state by the composition root."""
    container = getattr(request.app.state, "container", None)
    if container is None:  # pragma: no cover - the factory always sets it
        raise RuntimeError("Application container was not initialised.")
    return container  # type: ignore[no-any-return]


ContainerDep = Annotated[UseCaseContainer, Depends(get_container)]


def get_register_profile(container: ContainerDep) -> RegisterProfileUseCase:
    return container.register_profile


def get_profile_reader(container: ContainerDep) -> GetProfileUseCase:
    return container.get_profile


def get_run_pipeline(container: ContainerDep) -> RunPipelineUseCase:
    return container.run_pipeline


def get_list_queries(container: ContainerDep) -> ListQueriesUseCase:
    return container.list_queries


def get_list_recommendations(container: ContainerDep) -> ListRecommendationsUseCase:
    return container.list_recommendations


def get_recheck_query(container: ContainerDep) -> RecheckQueryUseCase:
    return container.recheck_query


RegisterProfileDep = Annotated[RegisterProfileUseCase, Depends(get_register_profile)]
GetProfileDep = Annotated[GetProfileUseCase, Depends(get_profile_reader)]
RunPipelineDep = Annotated[RunPipelineUseCase, Depends(get_run_pipeline)]
ListQueriesDep = Annotated[ListQueriesUseCase, Depends(get_list_queries)]
ListRecommendationsDep = Annotated[ListRecommendationsUseCase, Depends(get_list_recommendations)]
RecheckQueryDep = Annotated[RecheckQueryUseCase, Depends(get_recheck_query)]

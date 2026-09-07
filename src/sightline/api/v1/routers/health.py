"""Health endpoint.

Not in the spec, but a service without one cannot be orchestrated: the compose stack, and any
scheduler in front of it, needs a readiness signal. It also reports which providers are
actually wired, which is the first question when a run behaves unexpectedly.
"""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict

from sightline.api.v1.dependencies import ContainerDep

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: Literal["ok", "degraded"]
    environment: str
    database: Literal["up", "down"]
    llm_provider: str
    llm_model: str
    dataforseo_mode: str


@router.get("/health", response_model=HealthResponse, summary="Service health")
async def health(container: ContainerDep) -> HealthResponse:
    """Report readiness and the active provider configuration."""
    reachable = await container.check_database()
    database: Literal["up", "down"] = "up" if reachable else "down"

    return HealthResponse(
        status="ok" if reachable else "degraded",
        environment=container.environment,
        database=database,
        llm_provider=container.llm_provider_name,
        llm_model=container.llm_model,
        dataforseo_mode=container.dataforseo_mode,
    )

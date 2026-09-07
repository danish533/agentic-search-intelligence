"""Pipeline trigger endpoint (spec S4.2)."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, status

from sightline.api.v1.dependencies import RunPipelineDep
from sightline.api.v1.schemas.common import ErrorResponse
from sightline.api.v1.schemas.run import RunResponse, RunTriggerRequest

router = APIRouter(tags=["pipeline"])


@router.post(
    "/profiles/{profile_uuid}/run",
    response_model=RunResponse,
    summary="Trigger the full agent DAG for a profile",
    responses={
        status.HTTP_404_NOT_FOUND: {"model": ErrorResponse},
        status.HTTP_502_BAD_GATEWAY: {"model": ErrorResponse},
        status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ErrorResponse},
    },
)
async def run_pipeline(
    profile_uuid: UUID,
    use_case: RunPipelineDep,
    payload: RunTriggerRequest | None = None,
) -> RunResponse:
    """Execute the DAG and return the run record with its report.

    Synchronous by design (permitted by spec S4.2): a run takes 10-30 seconds against live
    providers, and the caller receives the completed result rather than a job id to poll.

    Returns **200 even for a degraded run** - ``status`` will be ``partial`` with a
    ``degradation_reason``. A run that produced a partial answer is a successful request that
    reports incomplete coverage, not a failed one.
    """
    result = await use_case.execute(profile_uuid, question=payload.question if payload else None)
    return RunResponse.from_result(result)

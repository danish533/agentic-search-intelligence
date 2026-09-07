"""Content recommendation endpoint (spec S4.2)."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, status

from sightline.api.v1.dependencies import ListRecommendationsDep
from sightline.api.v1.schemas.common import ErrorResponse
from sightline.api.v1.schemas.recommendation import RecommendationListResponse

router = APIRouter(tags=["recommendations"])


@router.get(
    "/profiles/{profile_uuid}/recommendations",
    response_model=RecommendationListResponse,
    summary="List content recommendations from the latest run",
    responses={status.HTTP_404_NOT_FOUND: {"model": ErrorResponse}},
)
async def list_recommendations(
    profile_uuid: UUID,
    use_case: ListRecommendationsDep,
) -> RecommendationListResponse:
    """Return recommendations for the profile's most recent run, highest priority first."""
    return RecommendationListResponse.from_entities(await use_case.execute(profile_uuid))

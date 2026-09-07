"""Business profile endpoints (spec S4.1)."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, status

from sightline.api.v1.dependencies import GetProfileDep, RegisterProfileDep
from sightline.api.v1.schemas.common import ErrorResponse
from sightline.api.v1.schemas.profile import (
    ProfileCreatedResponse,
    ProfileCreateRequest,
    ProfileDetailResponse,
)
from sightline.application.use_cases.register_profile import RegisterProfileInput

router = APIRouter(tags=["profiles"])


@router.post(
    "/profiles",
    status_code=status.HTTP_201_CREATED,
    response_model=ProfileCreatedResponse,
    summary="Register a brand profile",
    responses={
        status.HTTP_409_CONFLICT: {
            "model": ErrorResponse,
            "description": "A profile for this domain already exists.",
        },
        status.HTTP_422_UNPROCESSABLE_CONTENT: {"model": ErrorResponse},
    },
)
async def create_profile(
    payload: ProfileCreateRequest,
    use_case: RegisterProfileDep,
) -> ProfileCreatedResponse:
    """Create a profile. This is the entry point for the DAG pipeline."""
    profile = await use_case.execute(
        RegisterProfileInput(
            name=payload.name,
            domain=payload.domain,
            industry=payload.industry,
            description=payload.description,
            competitors=tuple(payload.competitors),
        )
    )
    return ProfileCreatedResponse.from_entity(profile)


@router.get(
    "/profiles/{profile_uuid}",
    response_model=ProfileDetailResponse,
    summary="Retrieve a profile with its run statistics",
    responses={status.HTTP_404_NOT_FOUND: {"model": ErrorResponse}},
)
async def get_profile(
    profile_uuid: UUID,
    use_case: GetProfileDep,
) -> ProfileDetailResponse:
    """Return the profile plus total runs, latest run status, and mean opportunity score."""
    return ProfileDetailResponse.from_summary(await use_case.execute(profile_uuid))

"""Discovered-query endpoints (spec S4.2)."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Query, status

from sightline.api.v1.dependencies import ListQueriesDep, RecheckQueryDep
from sightline.api.v1.schemas.common import ErrorResponse
from sightline.api.v1.schemas.query import QueryListResponse, RecheckResponse
from sightline.application.dto.pagination import (
    DEFAULT_PAGE,
    DEFAULT_PER_PAGE,
    MAX_PER_PAGE,
    PageRequest,
)
from sightline.application.dto.query_filters import QueryFilters
from sightline.domain.value_objects.enums import VisibilityStatus

router = APIRouter(tags=["queries"])


@router.get(
    "/profiles/{profile_uuid}/queries",
    response_model=QueryListResponse,
    summary="List a profile's discovered queries, highest opportunity first",
    responses={status.HTTP_404_NOT_FOUND: {"model": ErrorResponse}},
)
async def list_queries(
    profile_uuid: UUID,
    use_case: ListQueriesDep,
    min_score: Annotated[
        float | None,
        Query(ge=0.0, le=1.0, description="Only queries scoring at or above this value."),
    ] = None,
    status_filter: Annotated[
        VisibilityStatus | None,
        Query(alias="status", description="Filter by visibility status."),
    ] = None,
    page: Annotated[int, Query(ge=1)] = DEFAULT_PAGE,
    per_page: Annotated[int, Query(ge=1, le=MAX_PER_PAGE)] = DEFAULT_PER_PAGE,
) -> QueryListResponse:
    """Return the sub-queries from the profile's most recent run.

    A profile with no runs yet returns an empty page rather than 404: the resource exists, it
    simply has nothing in it.
    """
    page_result = await use_case.execute(
        profile_uuid,
        filters=QueryFilters(min_score=min_score, visibility_status=status_filter),
        page_request=PageRequest(page=page, per_page=per_page),
    )
    return QueryListResponse.from_page(page_result)


@router.post(
    "/queries/{query_uuid}/recheck",
    response_model=RecheckResponse,
    summary="Re-measure a single query",
    responses={
        status.HTTP_404_NOT_FOUND: {"model": ErrorResponse},
        status.HTTP_502_BAD_GATEWAY: {"model": ErrorResponse},
    },
)
async def recheck_query(
    query_uuid: UUID,
    use_case: RecheckQueryDep,
) -> RecheckResponse:
    """Re-run retrieval, normalization and analysis for one query.

    Useful after publishing content, or after a previous run took a fallback path. The
    response carries the previous measurement alongside the new one so the delta is visible.
    """
    return RecheckResponse.from_result(await use_case.execute(query_uuid))

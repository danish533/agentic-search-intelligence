"""Discovered-query response schemas (spec S4.2)."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from sightline.api.v1.schemas.common import PaginationMeta
from sightline.application.dto.pagination import Page
from sightline.application.dto.results import RecheckResult
from sightline.domain.entities.discovered_query import DiscoveredQuery
from sightline.domain.value_objects.enums import RunStatus, VisibilityStatus


class QueryResponse(BaseModel):
    """One discovered query, carrying exactly the fields the spec enumerates."""

    model_config = ConfigDict(frozen=True)

    query_uuid: UUID
    query_text: str
    estimated_search_volume: int
    competitive_difficulty: int = Field(ge=0, le=100)
    opportunity_score: float = Field(ge=0.0, le=1.0)
    domain_visible: bool
    visibility_position: int | None
    visibility_status: VisibilityStatus = Field(
        description="Three-state visibility. 'unknown' means retrieval never established it, "
        "which the boolean above cannot express on its own."
    )
    discovered_at: datetime

    @classmethod
    def from_entity(cls, query: DiscoveredQuery) -> QueryResponse:
        return cls(
            query_uuid=query.uuid,
            query_text=query.query_text,
            estimated_search_volume=int(query.estimated_search_volume),
            competitive_difficulty=int(query.competitive_difficulty),
            opportunity_score=query.opportunity_score.value,
            domain_visible=query.domain_visible,
            visibility_position=query.visibility_position,
            visibility_status=query.visibility_status,
            discovered_at=query.discovered_at,
        )


class QueryListResponse(BaseModel):
    """Paginated query listing."""

    model_config = ConfigDict(frozen=True)

    items: list[QueryResponse]
    pagination: PaginationMeta

    @classmethod
    def from_page(cls, page: Page[DiscoveredQuery]) -> QueryListResponse:
        return cls(
            items=[QueryResponse.from_entity(query) for query in page.items],
            pagination=PaginationMeta.from_page(page),
        )


class RecheckInsight(BaseModel):
    model_config = ConfigDict(frozen=True)

    headline: str
    detail: str
    relevance_score: float


class RecheckResponse(BaseModel):
    """Response for ``POST /api/v1/queries/{query_uuid}/recheck``.

    Carries the previous measurement alongside the new one: the reason to recheck a query is
    to find out what changed, and a response holding only the new value cannot answer that.
    """

    model_config = ConfigDict(frozen=True)

    status: RunStatus
    query: QueryResponse
    previous_opportunity_score: float
    previous_visibility_status: str
    previous_visibility_position: int | None = None
    opportunity_score_delta: float = 0.0
    domains_gained: list[str] = Field(
        default_factory=list,
        description="Domains present now that were absent at the previous measurement.",
    )
    domains_lost: list[str] = Field(
        default_factory=list,
        description="Domains present at the previous measurement that are absent now.",
    )
    insights: list[RecheckInsight]
    total_tokens: int
    error_message: str | None = None

    @classmethod
    def from_result(cls, result: RecheckResult) -> RecheckResponse:
        insights: Sequence[RecheckInsight] = [
            RecheckInsight(
                headline=insight.headline,
                detail=insight.detail,
                relevance_score=insight.relevance_score.value,
            )
            for insight in result.insights
        ]
        return cls(
            status=result.status,
            query=QueryResponse.from_entity(result.query),
            previous_opportunity_score=result.previous_opportunity_score,
            previous_visibility_status=result.previous_visibility_status,
            previous_visibility_position=result.previous_visibility_position,
            domains_gained=list(result.domains_gained),
            domains_lost=list(result.domains_lost),
            opportunity_score_delta=round(
                result.query.opportunity_score.value - result.previous_opportunity_score, 4
            ),
            insights=list(insights),
            total_tokens=result.total_tokens,
            error_message=result.error_message,
        )

"""Recommendation response schemas (spec S4.2)."""

from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from sightline.domain.entities.recommendation import Recommendation
from sightline.domain.value_objects.enums import ContentType, Priority


class RecommendationResponse(BaseModel):
    """One content recommendation, with exactly the fields the spec enumerates."""

    model_config = ConfigDict(frozen=True)

    recommendation_uuid: UUID
    target_query_uuid: UUID
    content_type: ContentType
    title: str
    rationale: str
    target_keywords: list[str]
    priority: Priority

    @classmethod
    def from_entity(cls, recommendation: Recommendation) -> RecommendationResponse:
        return cls(
            recommendation_uuid=recommendation.uuid,
            target_query_uuid=recommendation.target_query_uuid,
            content_type=recommendation.content_type,
            title=recommendation.title,
            rationale=recommendation.rationale,
            target_keywords=list(recommendation.target_keywords),
            priority=recommendation.priority,
        )


class RecommendationListResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    items: list[RecommendationResponse]
    total_items: int

    @classmethod
    def from_entities(cls, recommendations: Sequence[Recommendation]) -> RecommendationListResponse:
        items = [RecommendationResponse.from_entity(item) for item in recommendations]
        return cls(items=items, total_items=len(items))

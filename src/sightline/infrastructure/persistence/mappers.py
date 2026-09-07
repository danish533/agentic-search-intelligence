"""Translation between persistence models and domain entities.

Explicit functions, not an ORM-maps-the-domain shortcut. Keeping the two model families
separate (CLAUDE.md A5) is what stops SQLAlchemy vocabulary - lazy loading, identity map,
session lifetime - from leaking into domain logic, and what lets the domain enforce its
invariants on every read rather than trusting whatever the database happens to hold.

Reconstruction runs through the entity constructors, so a row that violates a domain invariant
fails loudly at the boundary instead of flowing onward as a valid-looking object.
"""

from __future__ import annotations

from sightline.domain.entities.discovered_query import DiscoveredQuery
from sightline.domain.entities.insight import Insight
from sightline.domain.entities.pipeline_run import PipelineRun
from sightline.domain.entities.profile import Profile
from sightline.domain.entities.recommendation import Recommendation
from sightline.domain.value_objects.scores import (
    CompetitiveDifficulty,
    OpportunityScore,
    RelevanceScore,
    SearchVolume,
)
from sightline.infrastructure.persistence.models.discovered_query import DiscoveredQueryModel
from sightline.infrastructure.persistence.models.insight import InsightModel
from sightline.infrastructure.persistence.models.pipeline_run import PipelineRunModel
from sightline.infrastructure.persistence.models.profile import ProfileModel
from sightline.infrastructure.persistence.models.recommendation import RecommendationModel


def profile_to_domain(model: ProfileModel) -> Profile:
    return Profile(
        name=model.name,
        domain=model.domain,
        industry=model.industry,
        description=model.description,
        competitors=tuple(model.competitors),
        uuid=model.uuid,
        created_at=model.created_at,
    )


def profile_to_model(entity: Profile) -> ProfileModel:
    return ProfileModel(
        uuid=entity.uuid,
        name=entity.name,
        domain=entity.domain,
        industry=entity.industry,
        description=entity.description,
        competitors=list(entity.competitors),
        created_at=entity.created_at,
    )


def run_to_domain(model: PipelineRunModel) -> PipelineRun:
    return PipelineRun(
        profile_uuid=model.profile_uuid,
        correlation_id=model.correlation_id,
        question=model.question,
        uuid=model.uuid,
        status=model.status,
        started_at=model.started_at,
        completed_at=model.completed_at,
        planned_retrieval_count=model.planned_retrieval_count,
        successful_retrieval_count=model.successful_retrieval_count,
        normalized_record_count=model.normalized_record_count,
        total_tokens=model.total_tokens,
        degradation_reason=model.degradation_reason,
        error_message=model.error_message,
        report=model.report,
        metrics=model.metrics,
    )


def run_to_model(entity: PipelineRun) -> PipelineRunModel:
    return PipelineRunModel(
        uuid=entity.uuid,
        profile_uuid=entity.profile_uuid,
        correlation_id=entity.correlation_id,
        question=entity.question,
        status=entity.status,
        started_at=entity.started_at,
        completed_at=entity.completed_at,
        planned_retrieval_count=entity.planned_retrieval_count,
        successful_retrieval_count=entity.successful_retrieval_count,
        normalized_record_count=entity.normalized_record_count,
        total_tokens=entity.total_tokens,
        degradation_reason=entity.degradation_reason,
        error_message=entity.error_message,
        report=dict(entity.report) if entity.report is not None else None,
        metrics=dict(entity.metrics) if entity.metrics is not None else None,
    )


def apply_run_changes(model: PipelineRunModel, entity: PipelineRun) -> None:
    """Copy a mutated run aggregate onto its already-persistent row.

    Used by ``save``: updating the attached instance keeps the change inside the session's
    unit of work, whereas merging a freshly built model would issue a redundant SELECT.
    """
    model.status = entity.status
    model.completed_at = entity.completed_at
    model.planned_retrieval_count = entity.planned_retrieval_count
    model.successful_retrieval_count = entity.successful_retrieval_count
    model.normalized_record_count = entity.normalized_record_count
    model.total_tokens = entity.total_tokens
    model.degradation_reason = entity.degradation_reason
    model.error_message = entity.error_message
    model.report = dict(entity.report) if entity.report is not None else None
    model.metrics = dict(entity.metrics) if entity.metrics is not None else None


def query_to_domain(model: DiscoveredQueryModel) -> DiscoveredQuery:
    return DiscoveredQuery(
        profile_uuid=model.profile_uuid,
        run_uuid=model.run_uuid,
        query_text=model.query_text,
        estimated_search_volume=SearchVolume(model.estimated_search_volume),
        competitive_difficulty=CompetitiveDifficulty(model.competitive_difficulty),
        opportunity_score=OpportunityScore(model.opportunity_score),
        visibility_status=model.visibility_status,
        visibility_position=model.visibility_position,
        evidence=model.evidence,
        uuid=model.uuid,
        discovered_at=model.discovered_at,
    )


def query_to_model(entity: DiscoveredQuery) -> DiscoveredQueryModel:
    return DiscoveredQueryModel(
        uuid=entity.uuid,
        profile_uuid=entity.profile_uuid,
        run_uuid=entity.run_uuid,
        query_text=entity.query_text,
        estimated_search_volume=int(entity.estimated_search_volume),
        competitive_difficulty=int(entity.competitive_difficulty),
        opportunity_score=float(entity.opportunity_score),
        visibility_status=entity.visibility_status,
        visibility_position=entity.visibility_position,
        evidence=dict(entity.evidence) if entity.evidence is not None else None,
        discovered_at=entity.discovered_at,
    )


def apply_query_changes(model: DiscoveredQueryModel, entity: DiscoveredQuery) -> None:
    """Copy re-measured values onto an existing query row - the recheck endpoint's write."""
    model.estimated_search_volume = int(entity.estimated_search_volume)
    model.competitive_difficulty = int(entity.competitive_difficulty)
    model.opportunity_score = float(entity.opportunity_score)
    model.visibility_status = entity.visibility_status
    model.visibility_position = entity.visibility_position
    model.evidence = dict(entity.evidence) if entity.evidence is not None else None
    model.discovered_at = entity.discovered_at


def insight_to_domain(model: InsightModel) -> Insight:
    return Insight(
        run_uuid=model.run_uuid,
        headline=model.headline,
        detail=model.detail,
        relevance_score=RelevanceScore(model.relevance_score),
        supporting_query_uuids=tuple(model.supporting_query_uuids),
        uuid=model.uuid,
    )


def insight_to_model(entity: Insight, *, position: int = 0) -> InsightModel:
    return InsightModel(
        uuid=entity.uuid,
        run_uuid=entity.run_uuid,
        headline=entity.headline,
        detail=entity.detail,
        relevance_score=float(entity.relevance_score),
        supporting_query_uuids=list(entity.supporting_query_uuids),
        position=position,
    )


def recommendation_to_domain(model: RecommendationModel) -> Recommendation:
    return Recommendation(
        run_uuid=model.run_uuid,
        target_query_uuid=model.target_query_uuid,
        content_type=model.content_type,
        title=model.title,
        rationale=model.rationale,
        target_keywords=tuple(model.target_keywords),
        priority=model.priority,
        uuid=model.uuid,
    )


def recommendation_to_model(entity: Recommendation, *, position: int = 0) -> RecommendationModel:
    return RecommendationModel(
        uuid=entity.uuid,
        run_uuid=entity.run_uuid,
        target_query_uuid=entity.target_query_uuid,
        content_type=entity.content_type,
        title=entity.title,
        rationale=entity.rationale,
        target_keywords=list(entity.target_keywords),
        priority=entity.priority,
        position=position,
    )

"""Pipeline-run request and response schemas (spec S4.2)."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from sightline.application.dto.results import RunResult
from sightline.domain.value_objects.enums import DegradationReason, RunStatus


class RunTriggerRequest(BaseModel):
    """Body of ``POST /api/v1/profiles/{profile_uuid}/run``.

    Every field is optional, so an empty body is valid - which keeps the spec's bodyless
    trigger working while still allowing the natural-language question of spec S2.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    question: str | None = Field(
        default=None,
        max_length=1000,
        description="The research question to answer. When omitted, one is derived from the "
        "profile.",
        examples=["How does Surfer SEO show up in AI answers for 'best SEO content tools'?"],
    )


class InsightResponse(BaseModel):
    """One finding, with the relevance score spec S4.2 asks for."""

    model_config = ConfigDict(frozen=True)

    insight_uuid: UUID
    headline: str
    detail: str
    relevance_score: float = Field(ge=0.0, le=1.0)
    supporting_query_uuids: list[UUID]


class ReportResponse(BaseModel):
    """The dual-form report: machine-readable JSON and a human-readable summary."""

    model_config = ConfigDict(frozen=True)

    headline: str
    summary: str = Field(description="Human-readable markdown summary.")
    structured: dict[str, Any] = Field(description="The machine-readable report body.")
    generated_at: datetime


class RunResponse(BaseModel):
    """200 response for a completed (or degraded) pipeline run."""

    model_config = ConfigDict(frozen=True)

    pipeline_run_uuid: UUID
    profile_uuid: UUID
    correlation_id: str = Field(
        description="Trace identifier. Every log line for this run carries it."
    )
    status: RunStatus
    question: str

    planned_retrieval_calls: int = Field(
        description="Retrieval calls the Query Planner decided were needed."
    )
    successful_retrieval_calls: int
    normalized_record_count: int = Field(
        description="Records successfully extracted and normalized."
    )

    top_insights: list[InsightResponse]
    report: ReportResponse
    total_tokens: int

    degradation_reason: DegradationReason | None = Field(
        default=None,
        description="Set when status is 'partial': why the run could not complete fully.",
    )
    error_message: str | None = None
    duration_ms: float | None = None
    metrics: dict[str, Any] = Field(
        default_factory=dict,
        description="Per-node latency, success rates, API call counts and retry totals.",
    )

    @classmethod
    def from_result(cls, result: RunResult, *, top_insights: int = 5) -> RunResponse:
        run, outcome = result.run, result.outcome
        ranked = sorted(
            outcome.insights, key=lambda insight: insight.relevance_score.value, reverse=True
        )
        return cls(
            pipeline_run_uuid=run.uuid,
            profile_uuid=run.profile_uuid,
            correlation_id=run.correlation_id,
            status=run.status,
            question=run.question,
            planned_retrieval_calls=run.planned_retrieval_count,
            successful_retrieval_calls=run.successful_retrieval_count,
            normalized_record_count=run.normalized_record_count,
            top_insights=[
                InsightResponse(
                    insight_uuid=insight.uuid,
                    headline=insight.headline,
                    detail=insight.detail,
                    relevance_score=insight.relevance_score.value,
                    supporting_query_uuids=list(insight.supporting_query_uuids),
                )
                for insight in ranked[:top_insights]
            ],
            report=ReportResponse(
                headline=outcome.report.headline,
                summary=outcome.report.summary,
                structured=dict(outcome.report.structured),
                generated_at=outcome.report.generated_at,
            ),
            total_tokens=run.total_tokens,
            degradation_reason=run.degradation_reason,
            error_message=run.error_message,
            duration_ms=run.duration_ms,
            metrics=dict(outcome.metrics_summary),
        )

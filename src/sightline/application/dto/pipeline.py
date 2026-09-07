"""DTOs crossing the application <-> agent boundary.

The application ring may not import the agent ring (CLAUDE.md A1), so the DAG is reached
through the :class:`~sightline.application.ports.pipeline_engine.PipelineEngine` port. These
are the types that flow across that port: plain dataclasses, no framework, no LangGraph
vocabulary leaking inward.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from sightline.domain.entities.discovered_query import DiscoveredQuery
from sightline.domain.entities.insight import Insight
from sightline.domain.entities.profile import Profile
from sightline.domain.entities.recommendation import Recommendation
from sightline.domain.value_objects.enums import DegradationReason, RunStatus


@dataclass(frozen=True, slots=True)
class PipelineRequest:
    """Everything the DAG needs to execute a full run."""

    profile: Profile
    question: str
    run_uuid: UUID
    correlation_id: str


@dataclass(frozen=True, slots=True)
class RecheckRequest:
    """Input to the recheck subgraph: retrieval -> normalization -> analysis for one query."""

    profile: Profile
    query: DiscoveredQuery
    run_uuid: UUID
    correlation_id: str


@dataclass(frozen=True, slots=True)
class ReportDocument:
    """The Report agent's output: the dual JSON + human-readable form spec S4.2 requires."""

    headline: str
    summary: str
    structured: Mapping[str, Any] = field(default_factory=dict)
    generated_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(frozen=True, slots=True)
class PipelineOutcome:
    """Result of a full DAG run, including a degraded one.

    A degraded run still returns a populated outcome - that is what "graceful degradation"
    (spec S3.5) means here: ``status`` is ``PARTIAL`` and ``degradation_reason`` is set, but
    the caller still receives a report rather than an exception.
    """

    status: RunStatus
    report: ReportDocument
    queries: tuple[DiscoveredQuery, ...] = ()
    insights: tuple[Insight, ...] = ()
    recommendations: tuple[Recommendation, ...] = ()
    planned_retrieval_count: int = 0
    successful_retrieval_count: int = 0
    normalized_record_count: int = 0
    total_tokens: int = 0
    metrics_summary: Mapping[str, Any] = field(default_factory=dict)
    degradation_reason: DegradationReason | None = None
    error_message: str | None = None


@dataclass(frozen=True, slots=True)
class RecheckOutcome:
    """Result of re-running the retrieval subgraph for a single query (spec S4.2)."""

    status: RunStatus
    query: DiscoveredQuery
    insights: tuple[Insight, ...] = ()
    total_tokens: int = 0
    metrics_summary: Mapping[str, Any] = field(default_factory=dict)
    degradation_reason: DegradationReason | None = None
    error_message: str | None = None

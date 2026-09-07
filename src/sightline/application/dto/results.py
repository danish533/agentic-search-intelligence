"""Use-case result DTOs.

Plain dataclasses returned to the API ring. Deliberately separate from both the domain
entities and the HTTP schemas (CLAUDE.md A5), so a change to a response shape never reaches
back into the domain, and a change to an entity never silently alters the wire contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from sightline.application.dto.pipeline import PipelineOutcome
from sightline.domain.entities.discovered_query import DiscoveredQuery
from sightline.domain.entities.insight import Insight
from sightline.domain.entities.pipeline_run import PipelineRun
from sightline.domain.entities.profile import Profile
from sightline.domain.value_objects.enums import RunStatus


@dataclass(frozen=True, slots=True)
class ProfileSummary:
    """A profile plus the run statistics spec S4.1 requires on retrieval."""

    profile: Profile
    total_runs: int
    latest_run_uuid: UUID | None = None
    latest_run_status: RunStatus | None = None
    latest_run_at: str | None = None
    average_opportunity_score: float | None = None


@dataclass(frozen=True, slots=True)
class RunResult:
    """A completed pipeline run: the persisted record plus the DAG's own output."""

    run: PipelineRun
    outcome: PipelineOutcome


@dataclass(frozen=True, slots=True)
class RecheckResult:
    """The outcome of re-measuring a single query.

    Carries what *moved*, not only the new numbers. A score going from 0.74 to 0.81 does not
    tell a marketing team anything actionable; "clearscope.io dropped out and you gained an AI
    Overview citation" does. That comparison is only possible because the previous run stored
    its evidence.
    """

    query: DiscoveredQuery
    previous_opportunity_score: float
    previous_visibility_status: str
    status: RunStatus
    previous_visibility_position: int | None = None
    domains_gained: tuple[str, ...] = ()
    domains_lost: tuple[str, ...] = ()
    insights: tuple[Insight, ...] = ()
    total_tokens: int = 0
    error_message: str | None = None

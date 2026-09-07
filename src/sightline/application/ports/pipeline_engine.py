"""Pipeline engine port.

The DAG lives in the agent ring, which sits *outside* the application ring, so a use case may
not import it (CLAUDE.md A1, enforced by import-linter). This port inverts that dependency:
the application declares what it needs of an orchestrator, and the agent ring supplies it.

The practical payoff is testing - a use case can be exercised against a stub engine with no
LangGraph, no LLM and no network - and the architectural payoff is that swapping the
orchestration technology touches one adapter.
"""

from __future__ import annotations

from typing import Protocol

from sightline.application.dto.pipeline import (
    PipelineOutcome,
    PipelineRequest,
    RecheckOutcome,
    RecheckRequest,
)


class PipelineEngine(Protocol):
    """Executes the agent DAG."""

    async def run(self, request: PipelineRequest) -> PipelineOutcome:
        """Execute the full graph: plan -> retrieve -> normalize -> analyse -> report.

        Never raises for a data-level failure. A run that could not complete comes back as a
        ``PARTIAL`` or ``FAILED`` outcome carrying a degradation reason, because the API must
        answer with a run record rather than a stack trace (spec S3.5).
        """
        ...

    async def recheck(self, request: RecheckRequest) -> RecheckOutcome:
        """Re-execute the retrieval -> normalization -> analysis subgraph for one query."""
        ...

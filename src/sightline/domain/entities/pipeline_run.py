"""The PipelineRun aggregate: one execution of the DAG.

This is the only mutable entity in the domain, because it is the only one with a lifecycle.
Transitions are guarded so an illegal state change raises rather than silently corrupting the
run record that the API reports on.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from sightline.domain.errors import InvalidRunTransitionError
from sightline.domain.value_objects.enums import DegradationReason, RunStatus


@dataclass(slots=True)
class PipelineRun:
    """Execution record for a single DAG run, including its degradation outcome."""

    profile_uuid: UUID
    correlation_id: str
    question: str
    uuid: UUID = field(default_factory=uuid4)
    status: RunStatus = RunStatus.RUNNING
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None
    planned_retrieval_count: int = 0
    successful_retrieval_count: int = 0
    normalized_record_count: int = 0
    total_tokens: int = 0
    degradation_reason: DegradationReason | None = None
    error_message: str | None = None

    #: The Report agent's output and the run's metrics summary, held as already-serialised
    #: mappings. A run owns its outcome, so this belongs on the aggregate rather than in a
    #: side table; the shape is owned by the Report agent, so the domain does not model it.
    report: Mapping[str, Any] | None = None
    metrics: Mapping[str, Any] | None = None

    @property
    def is_terminal(self) -> bool:
        return self.status.is_terminal

    @property
    def duration_ms(self) -> float | None:
        """Wall-clock duration, or ``None`` while the run is still in flight."""
        if self.completed_at is None:
            return None
        return (self.completed_at - self.started_at).total_seconds() * 1000

    def record_plan(self, planned_retrieval_count: int) -> None:
        """Record how many retrieval calls the Query Planner decided were needed."""
        self._require_running(RunStatus.RUNNING)
        self.planned_retrieval_count = planned_retrieval_count

    def record_retrieval_outcome(self, successful: int) -> None:
        self._require_running(RunStatus.RUNNING)
        self.successful_retrieval_count = successful

    def record_normalized(self, normalized_record_count: int) -> None:
        self._require_running(RunStatus.RUNNING)
        self.normalized_record_count = normalized_record_count

    def add_tokens(self, tokens: int) -> None:
        """Accumulate token usage reported by the LLM provider across all nodes."""
        if tokens > 0:
            self.total_tokens += tokens

    def attach_output(
        self,
        *,
        report: Mapping[str, Any] | None = None,
        metrics: Mapping[str, Any] | None = None,
    ) -> None:
        """Attach the run's report and metrics summary before it reaches a terminal state."""
        self._require_running(RunStatus.RUNNING)
        if report is not None:
            self.report = report
        if metrics is not None:
            self.metrics = metrics

    def mark_completed(self) -> None:
        """The full happy path ran: every required node produced usable output."""
        self._require_running(RunStatus.COMPLETED)
        self.status = RunStatus.COMPLETED
        self.completed_at = datetime.now(UTC)

    def mark_partial(self, reason: DegradationReason, detail: str | None = None) -> None:
        """The graph degraded but still produced a report (spec S3.5 graceful degradation)."""
        self._require_running(RunStatus.PARTIAL)
        self.status = RunStatus.PARTIAL
        self.degradation_reason = reason
        self.error_message = detail
        self.completed_at = datetime.now(UTC)

    def mark_failed(self, error_message: str) -> None:
        """No usable output at all - reserved for failures the DAG cannot degrade around."""
        self._require_running(RunStatus.FAILED)
        self.status = RunStatus.FAILED
        self.error_message = error_message
        self.completed_at = datetime.now(UTC)

    def _require_running(self, requested: RunStatus) -> None:
        if self.status is not RunStatus.RUNNING:
            raise InvalidRunTransitionError(self.status.value, requested.value)

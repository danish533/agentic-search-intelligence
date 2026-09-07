"""In-run metrics collection.

Spec S3.6 asks for per-node latency, success/failure rate and API call counts, and states
that "a simple in-memory or logged summary at the end of a run is sufficient". This module is
that collector: it accumulates one record per node execution and per external call, then
renders a summary the Report agent and the API response can both carry.

It is deliberately not a Prometheus client. The README documents the production swap - this
type is the seam that makes it a one-adapter change rather than a rewrite.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class NodeExecutionRecord:
    """One completed execution of one DAG node."""

    node_name: str
    duration_ms: float
    succeeded: bool
    retry_count: int = 0
    error_type: str | None = None


@dataclass(frozen=True, slots=True)
class ApiCallRecord:
    """One logical external API call, counted after its retries have settled."""

    endpoint: str
    duration_ms: float
    succeeded: bool
    attempts: int = 1
    status_code: int | None = None
    error_type: str | None = None


@dataclass(slots=True)
class RunMetrics:
    """Mutable per-run accumulator. One instance per DAG run, never shared across runs."""

    node_executions: list[NodeExecutionRecord] = field(default_factory=list)
    api_calls: list[ApiCallRecord] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def record_node(self, record: NodeExecutionRecord) -> None:
        self.node_executions.append(record)

    def record_api_call(self, record: ApiCallRecord) -> None:
        self.api_calls.append(record)

    def record_tokens(self, *, prompt_tokens: int, completion_tokens: int) -> None:
        self.prompt_tokens += max(0, prompt_tokens)
        self.completion_tokens += max(0, completion_tokens)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def total_retries(self) -> int:
        """Retries actually performed, counted once.

        Only external-call attempts are summed. A node's ``retry_count`` is a *per-node
        attribution* of those same retries - the retrieval node reports the attempts its tool
        calls consumed, so that the per-node log required by spec S3.6 carries them - and
        adding both would report every retry twice.
        """
        return sum(max(0, record.attempts - 1) for record in self.api_calls)

    def node_retry_counts(self) -> dict[str, int]:
        """Retries attributed to each node. Sums to :attr:`total_retries` for a single-source
        pipeline; kept separate so the two views cannot be confused for independent totals."""
        counts: dict[str, int] = defaultdict(int)
        for record in self.node_executions:
            if record.retry_count:
                counts[record.node_name] += record.retry_count
        return dict(sorted(counts.items()))

    def node_latency_ms(self) -> dict[str, float]:
        """Total wall-clock time per node name, summed across fan-out branches."""
        latency: dict[str, float] = defaultdict(float)
        for record in self.node_executions:
            latency[record.node_name] += record.duration_ms
        return {name: round(value, 2) for name, value in sorted(latency.items())}

    def node_success_rate(self) -> float:
        if not self.node_executions:
            return 0.0
        successes = sum(1 for record in self.node_executions if record.succeeded)
        return round(successes / len(self.node_executions), 4)

    def api_success_rate(self) -> float:
        if not self.api_calls:
            return 0.0
        successes = sum(1 for record in self.api_calls if record.succeeded)
        return round(successes / len(self.api_calls), 4)

    def failed_nodes(self) -> list[str]:
        return [record.node_name for record in self.node_executions if not record.succeeded]

    def summary(self) -> dict[str, Any]:
        """Render the end-of-run summary emitted to logs and returned by the API."""
        return {
            "node_count": len(self.node_executions),
            "node_latency_ms": self.node_latency_ms(),
            "node_success_rate": self.node_success_rate(),
            "failed_nodes": self.failed_nodes(),
            "api_call_count": len(self.api_calls),
            "api_success_rate": self.api_success_rate(),
            "total_retries": self.total_retries,
            "node_retry_counts": self.node_retry_counts(),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


#: The metrics collector for the run executing in this context. Set once per run by the
#: pipeline engine and read by deep call sites - the DataForSEO client in particular - so that
#: a collector does not have to be threaded through node -> tool -> provider -> client
#: signatures that have no other reason to know about it. Same rationale as the correlation
#: context, and it propagates across ``asyncio.gather`` the same way.
_current_metrics: ContextVar[RunMetrics | None] = ContextVar("run_metrics", default=None)


def current_metrics() -> RunMetrics | None:
    """The active run's collector, or ``None`` outside a run (a bare tool call, a test)."""
    return _current_metrics.get()


@contextmanager
def metrics_scope(metrics: RunMetrics) -> Iterator[RunMetrics]:
    """Bind ``metrics`` as the active collector for the duration of the block."""
    token = _current_metrics.set(metrics)
    try:
        yield metrics
    finally:
        _current_metrics.reset(token)

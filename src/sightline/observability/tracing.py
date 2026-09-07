"""Node-level tracing.

Every DAG node executes inside a :func:`node_span`. The span is the single place that emits
the per-node observability record spec S3.6 requires - node name, correlation id, duration,
success/failure, retry count, redacted payloads - so no node has to remember to log, and no
two nodes can log it differently.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from sightline.observability.correlation import correlation_scope
from sightline.observability.logging import get_logger
from sightline.observability.metrics import NodeExecutionRecord, RunMetrics

_logger = get_logger(__name__)


@dataclass(slots=True)
class NodeSpan:
    """Handle for the currently executing node. Yielded by :func:`node_span`."""

    node_name: str
    retry_count: int = 0
    outputs: dict[str, Any] = field(default_factory=dict)

    def record_retry(self) -> None:
        self.retry_count += 1

    def set_output(self, **fields: Any) -> None:
        """Attach a summary of what the node produced. Redacted by the logging processor."""
        self.outputs.update(fields)


@asynccontextmanager
async def node_span(
    node_name: str,
    *,
    metrics: RunMetrics | None = None,
    inputs: dict[str, Any] | None = None,
    log_payloads: bool = True,
) -> AsyncIterator[NodeSpan]:
    """Time, trace and record a single node execution.

    The span re-raises whatever the node raised after recording it: this is a tracing
    boundary, not an error-handling boundary. Deciding how a failure routes is the graph's
    job (CLAUDE.md A8), so swallowing here would hide a failure the DAG needs to see.
    """
    span = NodeSpan(node_name=node_name)
    started = time.perf_counter()

    with correlation_scope(node_name=node_name):
        _logger.info(
            "node.started",
            **({"inputs": inputs} if log_payloads and inputs else {}),
        )
        try:
            yield span
        except Exception as exc:
            duration_ms = (time.perf_counter() - started) * 1000
            if metrics is not None:
                metrics.record_node(
                    NodeExecutionRecord(
                        node_name=node_name,
                        duration_ms=duration_ms,
                        succeeded=False,
                        retry_count=span.retry_count,
                        error_type=type(exc).__name__,
                    )
                )
            _logger.error(
                "node.failed",
                status="failed",
                duration_ms=round(duration_ms, 2),
                retry_count=span.retry_count,
                error_type=type(exc).__name__,
                error=str(exc),
                exc_info=True,
            )
            raise
        else:
            duration_ms = (time.perf_counter() - started) * 1000
            if metrics is not None:
                metrics.record_node(
                    NodeExecutionRecord(
                        node_name=node_name,
                        duration_ms=duration_ms,
                        succeeded=True,
                        retry_count=span.retry_count,
                    )
                )
            _logger.info(
                "node.completed",
                status="ok",
                duration_ms=round(duration_ms, 2),
                retry_count=span.retry_count,
                **({"outputs": span.outputs} if log_payloads and span.outputs else {}),
            )

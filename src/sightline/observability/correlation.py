"""Correlation context.

Spec S3.6 requires that "a single request can be followed node-by-node". That is implemented
with a correlation id generated once per request and carried implicitly through every node,
tool call and HTTP request via :mod:`contextvars` - so no function signature has to thread a
trace id through it, and nothing can forget to pass one on.

``contextvars`` propagate correctly across ``await`` boundaries *and* into tasks spawned by
``asyncio.gather``, which is what makes the parallel retrieval fan-out traceable.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass

CORRELATION_ID_HEADER = "X-Correlation-ID"

_correlation_id: ContextVar[str | None] = ContextVar("correlation_id", default=None)
_run_uuid: ContextVar[str | None] = ContextVar("run_uuid", default=None)
_node_name: ContextVar[str | None] = ContextVar("node_name", default=None)


@dataclass(frozen=True, slots=True)
class CorrelationContext:
    """The trace identifiers attached to the current logical operation."""

    correlation_id: str | None
    run_uuid: str | None
    node_name: str | None

    def as_log_fields(self) -> dict[str, str]:
        """Render as log fields, omitting anything not yet set."""
        fields = {
            "correlation_id": self.correlation_id,
            "run_uuid": self.run_uuid,
            "node": self.node_name,
        }
        return {key: value for key, value in fields.items() if value is not None}


def new_correlation_id() -> str:
    """Generate a fresh correlation id.

    Hex without dashes so it stays greppable in log aggregators that tokenise on ``-``.
    """
    return uuid.uuid4().hex


def current_context() -> CorrelationContext:
    return CorrelationContext(
        correlation_id=_correlation_id.get(),
        run_uuid=_run_uuid.get(),
        node_name=_node_name.get(),
    )


def current_correlation_id() -> str | None:
    return _correlation_id.get()


@contextmanager
def correlation_scope(
    *,
    correlation_id: str | None = None,
    run_uuid: str | None = None,
    node_name: str | None = None,
) -> Iterator[CorrelationContext]:
    """Bind trace identifiers for the duration of the block, then restore the prior values.

    Only the arguments actually supplied are rebound, so an inner node scope inherits the
    request's correlation id instead of clearing it.
    """
    tokens: list[tuple[ContextVar[str | None], Token[str | None]]] = []
    if correlation_id is not None:
        tokens.append((_correlation_id, _correlation_id.set(correlation_id)))
    if run_uuid is not None:
        tokens.append((_run_uuid, _run_uuid.set(run_uuid)))
    if node_name is not None:
        tokens.append((_node_name, _node_name.set(node_name)))
    try:
        yield current_context()
    finally:
        for variable, token in reversed(tokens):
            variable.reset(token)

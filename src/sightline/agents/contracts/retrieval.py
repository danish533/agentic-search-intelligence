"""Retrieval node output contract.

Deliberately holds **raw, unparsed** provider payloads. The retrieval agent's sole
responsibility is to fetch (spec S3.2); a retrieval result that had already been parsed would
mean retrieval and extraction were the same component, which is the merge this assessment
grades against first.

A dataclass rather than a Pydantic model: this never crosses an LLM boundary, so schema
validation would buy nothing and the raw payloads would be needlessly re-validated.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sightline.agents.contracts.plan import PlannedRetrieval
from sightline.application.ports.search_data_provider import RawApiResult
from sightline.tools.validation import ToolCallRejected


@dataclass(frozen=True, slots=True)
class RetrievalOutcome:
    """What one retrieval branch produced for one sub-query."""

    sub_query: PlannedRetrieval
    results: tuple[RawApiResult, ...] = ()
    rejected_calls: tuple[ToolCallRejected, ...] = ()
    error: str | None = None
    error_type: str | None = None
    attempts: int = 1
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tool_names: tuple[str, ...] = field(default_factory=tuple)

    @property
    def query_text(self) -> str:
        return self.sub_query.query_text

    @property
    def succeeded(self) -> bool:
        """A branch succeeded if it came back with at least one usable payload."""
        return bool(self.results) and self.error is None

    @property
    def retry_count(self) -> int:
        return max(0, self.attempts - 1)

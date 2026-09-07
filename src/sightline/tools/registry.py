"""Tool registry.

Holds the tools available to the retrieval agent and is the single source of truth for what
the model may call. The specs shown to the model and the schemas the validation gate enforces
are both read from here, so "advertised" and "accepted" cannot diverge.

Duplicate names are rejected at construction. Two tools answering to one name is not a
runtime edge case to handle gracefully - it is a wiring bug, and it should fail at startup
rather than resolve arbitrarily on the first call.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from typing import Any

from sightline.application.ports.llm_provider import ToolSpec
from sightline.domain.value_objects.enums import RetrievalKind
from sightline.tools.dataforseo.ai_overview import FETCH_AI_OVERVIEW
from sightline.tools.dataforseo.keyword_metrics import FETCH_KEYWORD_METRICS
from sightline.tools.dataforseo.llm_visibility import FETCH_LLM_VISIBILITY
from sightline.tools.dataforseo.serp import FETCH_ORGANIC_SERP
from sightline.tools.definition import Tool


class DuplicateToolError(Exception):
    """Two tools were registered under the same name."""

    def __init__(self, name: str) -> None:
        super().__init__(f"A tool named '{name}' is already registered.")


class UnknownToolError(KeyError):
    """A tool was requested by a name that is not registered."""

    def __init__(self, name: str, available: Sequence[str]) -> None:
        self.name = name
        self.available = tuple(available)
        super().__init__(f"Unknown tool '{name}'. Available: {', '.join(available)}.")


class ToolRegistry:
    """An immutable, name-indexed collection of tools."""

    def __init__(self, tools: Sequence[Tool[Any]]) -> None:
        indexed: dict[str, Tool[Any]] = {}
        for tool in tools:
            if tool.name in indexed:
                raise DuplicateToolError(tool.name)
            indexed[tool.name] = tool
        self._tools = indexed

    def __len__(self) -> int:
        return len(self._tools)

    def __iter__(self) -> Iterator[Tool[Any]]:
        return iter(self._tools.values())

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._tools

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._tools)

    def get(self, name: str) -> Tool[Any] | None:
        """Look up a tool, returning ``None`` when absent.

        This is what the validation gate uses: an unrecognised name from the model is expected
        input to be rejected as data, not an exception to be raised.
        """
        return self._tools.get(name)

    def require(self, name: str) -> Tool[Any]:
        """Look up a tool, raising :class:`UnknownToolError` when absent.

        For internal call sites that know the name is valid; never for model-supplied names.
        """
        tool = self._tools.get(name)
        if tool is None:
            raise UnknownToolError(name, self.names)
        return tool

    def for_kind(self, kind: RetrievalKind) -> Tool[Any] | None:
        """The tool serving one logical retrieval capability."""
        return next((tool for tool in self._tools.values() if tool.kind is kind), None)

    def subset(self, kinds: Iterable[RetrievalKind]) -> ToolRegistry:
        """A narrowed registry exposing only the tools for the given kinds.

        The retrieval node uses this to offer the model exactly the tools the Query Planner
        decided were needed, rather than the full catalogue every time - fewer irrelevant
        options measurably reduces mis-selection.
        """
        wanted = set(kinds)
        return ToolRegistry([tool for tool in self._tools.values() if tool.kind in wanted])

    def specs(self) -> tuple[ToolSpec, ...]:
        """The tool list handed to the LLM provider."""
        return tuple(tool.to_spec() for tool in self._tools.values())


def default_tool_registry() -> ToolRegistry:
    """The four DataForSEO tools - one per logical API call, per spec S3.4."""
    return ToolRegistry(
        [
            FETCH_ORGANIC_SERP,
            FETCH_AI_OVERVIEW,
            FETCH_LLM_VISIBILITY,
            FETCH_KEYWORD_METRICS,
        ]
    )

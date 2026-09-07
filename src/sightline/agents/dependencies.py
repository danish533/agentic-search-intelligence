"""Dependencies injected into DAG nodes.

Nodes receive their collaborators as an explicit bundle bound at graph-build time, rather than
importing them or reading a global. That is what keeps every node unit-testable in isolation
(CLAUDE.md R1): a test constructs this with a fake LLM and a stub provider and calls the node
function directly, with no graph, no network and no database.

Only ports appear here - never a concrete adapter - so the agent ring stays independent of
infrastructure and the import contract holds.
"""

from __future__ import annotations

from dataclasses import dataclass

from sightline.application.ports.llm_provider import LLMProvider
from sightline.application.ports.search_data_provider import SearchDataProvider
from sightline.config.settings import DataForSEOSettings, PipelineSettings
from sightline.tools.registry import ToolRegistry


@dataclass(frozen=True, slots=True)
class NodeDependencies:
    """Everything the DAG's nodes need from outside themselves."""

    llm: LLMProvider
    search_provider: SearchDataProvider
    tools: ToolRegistry
    pipeline: PipelineSettings
    dataforseo: DataForSEOSettings

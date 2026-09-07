"""The Tool type.

A tool binds four things that must never drift apart: the name the model calls, the
description it selects on, the schema its arguments are validated against, and the executor
that performs the real call. Declaring them together in one frozen object means a tool cannot
be half-registered - shown to the model but unexecutable, or executable under a schema
different from the one advertised.

Tools are pure declarations. They hold no provider, no client and no credentials; the provider
is passed in at execution time. That keeps the registry constructible without any I/O
dependency, which is what lets the tool layer be unit-tested with a stub provider.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from pydantic import BaseModel

from sightline.application.ports.llm_provider import ToolSpec
from sightline.application.ports.search_data_provider import RawApiResult, SearchDataProvider
from sightline.domain.value_objects.enums import RetrievalKind

#: Performs the real API call for a tool, given validated arguments.
type ToolExecutor[ArgsT: BaseModel] = Callable[[SearchDataProvider, ArgsT], Awaitable[RawApiResult]]


@dataclass(frozen=True, slots=True)
class Tool[ArgsT: BaseModel]:
    """One logical API call, exposed to the LLM."""

    name: str
    description: str
    kind: RetrievalKind
    args_schema: type[ArgsT]
    executor: ToolExecutor[ArgsT]

    def to_spec(self) -> ToolSpec:
        """Render as the provider-agnostic spec the LLM port accepts."""
        return ToolSpec(
            name=self.name,
            description=self.description,
            args_schema=self.args_schema,
        )

    async def execute(self, provider: SearchDataProvider, args: ArgsT) -> RawApiResult:
        """Perform the call.

        ``args`` must already be a validated instance of :attr:`args_schema` - this method is
        reachable only through :mod:`sightline.tools.validation`, which is the gate spec S3.3
        requires between the model's proposal and the real request.
        """
        return await self.executor(provider, args)

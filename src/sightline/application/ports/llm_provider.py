"""LLM provider port.

Agents depend on this protocol and never on a vendor SDK, which is what makes
``LLM_PROVIDER=openai|anthropic|gemini|fake`` a configuration change rather than a code change
- and what lets the whole test suite run offline against the deterministic ``fake`` adapter.

Two capabilities, deliberately separated:

``complete_structured``
    Schema-constrained generation. Used by the planner, analysis and report agents, which need
    a typed object back, not prose.

``propose_tool_calls``
    Tool selection. Used only by the retrieval agent. It returns the calls the model *wants*
    to make; it never executes them. Execution happens only after the argument-validation gate
    in :mod:`sightline.tools.validation` (spec S3.3).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from pydantic import BaseModel


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Token accounting for one LLM call. Zero when a provider does not report usage."""

    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
        )


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """A tool offered to the model: its name, what it is for, and its argument schema.

    ``args_schema`` is a Pydantic model - the same model the validation gate later enforces -
    so the schema the model is shown and the schema its arguments are checked against cannot
    drift apart (spec S3.3).
    """

    name: str
    description: str
    args_schema: type[BaseModel]


@dataclass(frozen=True, slots=True)
class ProposedToolCall:
    """A tool invocation the model requested. Unvalidated by construction.

    Arguments are ``Mapping[str, Any]`` on purpose: this is raw model output and may be
    malformed, partial, or reference a tool that does not exist. Typing it as anything
    stronger here would be a lie that the validation gate exists to expose.
    """

    tool_name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    call_id: str = ""
    raw_arguments: str | None = None
    """Set by an adapter when the model's argument payload could not be decoded at all.

    Providers often emit tool arguments as a JSON string; when that string is not valid JSON
    there is nothing to put in ``arguments``. Recording the raw text here lets the validation
    gate report *malformed payload* specifically, rather than mislabelling it as a set of
    missing fields, and gives the log something to diagnose from.
    """


@dataclass(frozen=True, slots=True)
class StructuredCompletion[T: BaseModel]:
    """A schema-validated model response plus its token cost."""

    value: T
    usage: TokenUsage = field(default_factory=TokenUsage)


@dataclass(frozen=True, slots=True)
class ToolCallCompletion:
    """The tool calls a model proposed, plus its token cost."""

    calls: tuple[ProposedToolCall, ...] = ()
    usage: TokenUsage = field(default_factory=TokenUsage)


class LLMProvider(Protocol):
    """The only LLM surface the agent ring is allowed to know about."""

    @property
    def name(self) -> str:
        """Provider identifier, for logs and the run report."""
        ...

    @property
    def model(self) -> str:
        """Model identifier, for logs and the run report."""
        ...

    async def complete_structured[T: BaseModel](
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: type[T],
    ) -> StructuredCompletion[T]:
        """Generate an instance of ``schema``.

        Raises :class:`~sightline.application.ports.llm_provider.LLMError` if the provider
        could not produce a schema-valid object after its own internal retries.
        """
        ...

    async def propose_tool_calls(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        tools: Sequence[ToolSpec],
    ) -> ToolCallCompletion:
        """Ask the model which tools to call, and with which arguments. Never executes them."""
        ...


class LLMError(Exception):
    """The LLM provider could not produce a usable response."""

    def __init__(self, provider: str, detail: str) -> None:
        self.provider = provider
        self.detail = detail
        super().__init__(f"LLM provider '{provider}' failed: {detail}")

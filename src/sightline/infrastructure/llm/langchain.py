"""LangChain-backed LLM provider.

One adapter serves OpenAI, Anthropic and Gemini, because at this boundary they differ only in
which ``BaseChatModel`` is constructed - selecting that is the factory's job, and it is the
only vendor-aware code in the system. Three near-identical adapters would have been three
places to fix the same bug.

The two port capabilities map onto two distinct LangChain mechanisms:

``complete_structured``
    ``with_structured_output(schema, include_raw=True)``. ``include_raw`` matters: without it
    the token usage is discarded, and spec S4.2 asks for total tokens in the run response.

``propose_tool_calls``
    ``bind_tools(...)`` followed by reading ``tool_calls`` **and** ``invalid_tool_calls``. The
    second list is the one that matters for spec S3.3: it holds calls whose argument payload
    would not parse, and forwarding them as proposals with ``raw_arguments`` set lets the
    validation gate reject them explicitly instead of them vanishing silently.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

from sightline.application.ports.llm_provider import (
    LLMError,
    ProposedToolCall,
    StructuredCompletion,
    TokenUsage,
    ToolCallCompletion,
    ToolSpec,
)
from sightline.observability.logging import get_logger

_logger = get_logger(__name__)


def _to_openai_tool(spec: ToolSpec) -> dict[str, Any]:
    """Render a tool spec as the function-calling schema LangChain normalises from.

    Built from the Pydantic model's own JSON Schema so the model is shown exactly the contract
    the validation gate will enforce - the same object, not a hand-maintained copy.
    """
    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description,
            "parameters": spec.args_schema.model_json_schema(),
        },
    }


def _usage_from(message: BaseMessage) -> TokenUsage:
    """Extract token usage, tolerating providers that report it differently or not at all."""
    metadata = getattr(message, "usage_metadata", None)
    if isinstance(metadata, Mapping):
        return TokenUsage(
            prompt_tokens=int(metadata.get("input_tokens", 0) or 0),
            completion_tokens=int(metadata.get("output_tokens", 0) or 0),
        )

    response_metadata = getattr(message, "response_metadata", {}) or {}
    raw = response_metadata.get("token_usage") or response_metadata.get("usage") or {}
    if isinstance(raw, Mapping):
        return TokenUsage(
            prompt_tokens=int(raw.get("prompt_tokens", raw.get("input_tokens", 0)) or 0),
            completion_tokens=int(raw.get("completion_tokens", raw.get("output_tokens", 0)) or 0),
        )
    return TokenUsage()


class LangChainLLMProvider:
    """``LLMProvider`` implemented over any LangChain chat model."""

    def __init__(self, *, model: BaseChatModel, provider_name: str, model_name: str) -> None:
        self._model = model
        self._provider_name = provider_name
        self._model_name = model_name

    @property
    def name(self) -> str:
        return self._provider_name

    @property
    def model(self) -> str:
        return self._model_name

    def _messages(self, system_prompt: str, user_prompt: str) -> list[BaseMessage]:
        return [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]

    async def complete_structured[T: BaseModel](
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: type[T],
    ) -> StructuredCompletion[T]:
        structured = self._model.with_structured_output(schema, include_raw=True)

        try:
            response = await structured.ainvoke(self._messages(system_prompt, user_prompt))
        except Exception as exc:
            raise LLMError(self._provider_name, f"structured completion failed: {exc}") from exc

        if not isinstance(response, Mapping):
            raise LLMError(
                self._provider_name,
                f"expected a raw/parsed mapping, got {type(response).__name__}",
            )

        parsing_error = response.get("parsing_error")
        parsed = response.get("parsed")
        if parsing_error is not None or parsed is None:
            raise LLMError(
                self._provider_name,
                f"model output did not satisfy {schema.__name__}: {parsing_error}",
            )
        if not isinstance(parsed, schema):
            raise LLMError(
                self._provider_name,
                f"expected {schema.__name__}, got {type(parsed).__name__}",
            )

        raw = response.get("raw")
        usage = _usage_from(raw) if isinstance(raw, BaseMessage) else TokenUsage()
        return StructuredCompletion(value=parsed, usage=usage)

    async def propose_tool_calls(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        tools: Sequence[ToolSpec],
    ) -> ToolCallCompletion:
        if not tools:
            raise LLMError(self._provider_name, "propose_tool_calls requires at least one tool")

        bound = self._model.bind_tools([_to_openai_tool(spec) for spec in tools])

        try:
            message = await bound.ainvoke(self._messages(system_prompt, user_prompt))
        except Exception as exc:
            raise LLMError(self._provider_name, f"tool-call proposal failed: {exc}") from exc

        if not isinstance(message, AIMessage):
            raise LLMError(
                self._provider_name, f"expected an AIMessage, got {type(message).__name__}"
            )

        proposals: list[ProposedToolCall] = [
            ProposedToolCall(
                tool_name=str(call.get("name", "")),
                arguments=call.get("args") or {},
                call_id=str(call.get("id") or ""),
            )
            for call in message.tool_calls
        ]

        # Forwarded rather than dropped: an unparseable argument payload is exactly the
        # "malformed tool-call arguments" case spec S3.3 requires be handled gracefully, and
        # it can only be handled if it reaches the gate.
        invalid = getattr(message, "invalid_tool_calls", []) or []
        proposals.extend(
            ProposedToolCall(
                tool_name=str(call.get("name") or ""),
                arguments={},
                call_id=str(call.get("id") or ""),
                raw_arguments=str(call.get("args") or ""),
            )
            for call in invalid
        )

        if invalid:
            _logger.warning(
                "llm.tool_calls.unparseable",
                provider=self._provider_name,
                count=len(invalid),
            )

        return ToolCallCompletion(calls=tuple(proposals), usage=_usage_from(message))

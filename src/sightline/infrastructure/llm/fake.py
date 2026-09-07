"""Deterministic offline LLM provider.

This is what makes ``make test`` free, fast, offline and reproducible, and it is the only way
to test the failure paths spec S5 requires: you cannot reliably make a real model emit a
missing required field or a hallucinated tool name on demand, but you can script one here.

Two layers of behaviour:

**Registered responders** - the composition root registers realistic responses for the agent
contracts it knows about. This is where a demo run gets a sensible plan and a coherent report.

**Schema synthesis** - the fallback. Given any Pydantic model, it constructs a
minimally-valid instance by walking the model's fields and honouring their constraints. That
means a new agent contract works against the fake the moment it is written, with no fixture to
maintain, and the fake can never return something the caller's own schema would reject.

Everything is seeded from the prompt, so the same input always produces the same output.
"""

from __future__ import annotations

import enum
import hashlib
import random
import types
import typing
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal, get_args, get_origin
from uuid import UUID

from annotated_types import Ge, Gt, Le, Lt, MaxLen, MinLen
from pydantic import BaseModel
from pydantic.fields import FieldInfo

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

#: Filler used for unconstrained string fields. Recognisable in output as synthetic.
_FILLER_WORDS = (
    "search",
    "visibility",
    "content",
    "ranking",
    "coverage",
    "insight",
    "audience",
    "analysis",
)

type StructuredResponder = Callable[[str, str], BaseModel]
type ToolCallResponder = Callable[[Sequence[ToolSpec], str, str], Sequence[ProposedToolCall]]


def compact_hint(text: str, *, max_chars: int = 72) -> str:
    """Reduce a multi-line prompt to one short, sentence-like line.

    Without this, string fields are filled with the raw prompt: a synthesized ``query_text``
    became the entire planner prompt, which then went to the search provider as a 200-character
    keyword. Schema-valid, and complete nonsense - the kind of wrong that no validator catches.
    """
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.endswith(":") or stripped[0] in "-#{[":
            continue
        return stripped[:max_chars]
    return ""


def _seeded(*parts: str) -> random.Random:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))  # noqa: S311 - test fixture, not security


def _length_constraint(metadata: Sequence[Any], kind: type, attribute: str) -> int | None:
    """Read an integer length bound (MinLen/MaxLen) out of a field's constraint metadata."""
    for item in metadata:
        if isinstance(item, kind):
            value = getattr(item, attribute, None)
            if isinstance(value, int):
                return value
    return None


def _numeric_constraint(metadata: Sequence[Any], kind: type, attribute: str) -> float | None:
    """Read a numeric bound (Ge/Gt/Le/Lt) out of a field's constraint metadata."""
    for item in metadata:
        if isinstance(item, kind):
            value = getattr(item, attribute, None)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)
    return None


def _synthesize_str(info: FieldInfo, hint: str, rng: random.Random) -> str:
    """A string satisfying any length bounds, preferring the caller's hint text."""
    minimum = _length_constraint(info.metadata, MinLen, "min_length") or 0
    maximum = _length_constraint(info.metadata, MaxLen, "max_length") or 200

    candidate = hint.strip() or " ".join(rng.sample(_FILLER_WORDS, k=3))
    while len(candidate) < minimum:
        candidate = f"{candidate} {rng.choice(_FILLER_WORDS)}".strip()
    return candidate[:maximum]


def _synthesize_number(info: FieldInfo, *, as_int: bool) -> float:
    """A number inside any declared bounds, biased low so it stays plausible."""
    lower = _numeric_constraint(info.metadata, Ge, "ge")
    if lower is None:
        exclusive = _numeric_constraint(info.metadata, Gt, "gt")
        lower = exclusive + 1 if exclusive is not None else 1
    upper = _numeric_constraint(info.metadata, Le, "le")
    if upper is None:
        exclusive_upper = _numeric_constraint(info.metadata, Lt, "lt")
        upper = exclusive_upper - 1 if exclusive_upper is not None else lower + 9

    value = min(max(lower, 1 if lower <= 1 else lower), upper)
    return int(value) if as_int else float(value)


def _synthesize(annotation: Any, info: FieldInfo, hint: str, rng: random.Random) -> Any:
    """Build one field value satisfying ``annotation`` and ``info``'s constraints."""
    origin = get_origin(annotation)

    if origin is Literal:
        return get_args(annotation)[0]

    if origin in (types.UnionType, typing.Union):
        non_null = [arg for arg in get_args(annotation) if arg is not type(None)]
        return _synthesize(non_null[0], info, hint, rng) if non_null else None

    if origin in (list, Sequence, tuple):
        item_annotation = (get_args(annotation) or (str,))[0]
        count = max(1, _length_constraint(info.metadata, MinLen, "min_length") or 1)
        # The container's own bounds constrain how many items there are, never what each item
        # looks like. Re-deriving FieldInfo from the item annotation both drops the container's
        # metadata and unwraps any Annotated[...] constraints the item declares for itself.
        item_info = FieldInfo.from_annotation(item_annotation)
        return [
            _synthesize(
                item_info.annotation,
                item_info,
                hint if index == 0 else f"{hint} {index + 1}",
                rng,
            )
            for index in range(count)
        ]

    if origin in (dict, Mapping):
        return {}

    if isinstance(annotation, type):
        if issubclass(annotation, BaseModel):
            return synthesize_model(annotation, hint=hint, seed=str(annotation.__name__))
        if issubclass(annotation, enum.Enum):
            return next(iter(annotation))
        if annotation is bool:
            return True
        if annotation is int:
            return int(_synthesize_number(info, as_int=True))
        if annotation is float:
            return float(_synthesize_number(info, as_int=False))
        if annotation is str:
            return _synthesize_str(info, hint, rng)
        if annotation is UUID:
            return UUID(int=rng.getrandbits(128))
        if annotation is datetime:
            return datetime.now(UTC)

    return _synthesize_str(info, hint, rng)


def synthesize_model[T: BaseModel](schema: type[T], *, hint: str = "", seed: str = "") -> T:
    """Construct a minimally-valid instance of ``schema``.

    Only required fields are populated; optional ones are left for Pydantic to default, which
    keeps synthetic tool calls close to what a well-behaved model would actually emit.
    """
    rng = _seeded(schema.__name__, seed, hint)
    values: dict[str, Any] = {}

    for name, info in schema.model_fields.items():
        if not info.is_required():
            continue
        values[name] = _synthesize(info.annotation, info, hint, rng)

    return schema.model_validate(values)


@dataclass(slots=True)
class FakeLLMProvider:
    """An ``LLMProvider`` that never leaves the process."""

    usage_per_call: TokenUsage = field(
        default_factory=lambda: TokenUsage(prompt_tokens=180, completion_tokens=120)
    )
    _structured: dict[type[BaseModel], StructuredResponder] = field(default_factory=dict)
    _tool_responder: ToolCallResponder | None = None
    _scripted_calls: list[Sequence[ProposedToolCall]] = field(default_factory=list)
    call_log: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return "fake"

    @property
    def model(self) -> str:
        return "deterministic-stub"

    def register_structured[T: BaseModel](
        self, schema: type[T], responder: Callable[[str, str], T]
    ) -> None:
        """Register a realistic response for one contract, overriding schema synthesis."""
        self._structured[schema] = typing.cast("StructuredResponder", responder)

    def register_tool_calls(self, responder: ToolCallResponder) -> None:
        self._tool_responder = responder

    def script_tool_calls(self, *batches: Sequence[ProposedToolCall]) -> None:
        """Queue exact tool-call batches, consumed one per ``propose_tool_calls``.

        The mechanism tests use to reproduce a hallucinated tool name or a malformed argument
        payload deterministically.
        """
        self._scripted_calls.extend(batches)

    async def complete_structured[T: BaseModel](
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: type[T],
    ) -> StructuredCompletion[T]:
        self.call_log.append(f"complete_structured:{schema.__name__}")

        responder = self._structured.get(schema)
        if responder is not None:
            produced = responder(system_prompt, user_prompt)
            if not isinstance(produced, schema):
                raise LLMError(
                    self.name,
                    f"registered responder returned {type(produced).__name__}, "
                    f"expected {schema.__name__}",
                )
            return StructuredCompletion(value=produced, usage=self.usage_per_call)

        value = synthesize_model(schema, hint=compact_hint(user_prompt), seed=system_prompt[:60])
        return StructuredCompletion(value=value, usage=self.usage_per_call)

    async def propose_tool_calls(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        tools: Sequence[ToolSpec],
    ) -> ToolCallCompletion:
        self.call_log.append(f"propose_tool_calls:{len(tools)}")

        if not tools:
            raise LLMError(self.name, "propose_tool_calls requires at least one tool")

        if self._scripted_calls:
            return ToolCallCompletion(
                calls=tuple(self._scripted_calls.pop(0)), usage=self.usage_per_call
            )

        if self._tool_responder is not None:
            return ToolCallCompletion(
                calls=tuple(self._tool_responder(tools, system_prompt, user_prompt)),
                usage=self.usage_per_call,
            )

        # Default: propose one well-formed call per offered tool, with arguments synthesized
        # from each tool's own schema - so the gate downstream sees valid input unless a test
        # deliberately scripts otherwise.
        calls = tuple(
            ProposedToolCall(
                tool_name=spec.name,
                arguments=synthesize_model(
                    spec.args_schema, hint=compact_hint(user_prompt), seed=spec.name
                ).model_dump(),
                call_id=f"fake-{index}",
            )
            for index, spec in enumerate(tools)
        )
        return ToolCallCompletion(calls=calls, usage=self.usage_per_call)

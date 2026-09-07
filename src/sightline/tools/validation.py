"""The tool-call validation gate.

Spec S3.3, verbatim: *"your code should validate the LLM's tool call arguments before
executing the real API call"* and *"handle malformed or partial tool-call arguments gracefully
(e.g. missing required field) rather than crashing"*.

This module is that gate, and it is the only path from a model-proposed call to a real HTTP
request. Nothing downstream of it ever sees an unvalidated argument, and nothing upstream of
it can reach the network.

**Rejection is a value, not an exception** (CLAUDE.md A8). A model proposing a bad call is
ordinary, expected traffic - not an exceptional condition. Returning a typed rejection lets
the retrieval node choose what to do: drop that one call and proceed with the others, or feed
:meth:`ToolCallRejected.repair_hint` back to the model for a corrective retry. Raising would
force one policy on every caller and would let a single malformed argument abort a fan-out in
which every other branch was fine.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ValidationError

from sightline.application.ports.llm_provider import ProposedToolCall
from sightline.observability.logging import get_logger
from sightline.tools.definition import Tool
from sightline.tools.registry import ToolRegistry

_logger = get_logger(__name__)

_MAX_RAW_EXCERPT_CHARS = 200


class RejectionReason(StrEnum):
    """Why a proposed tool call was refused. Emitted in logs and in the repair hint."""

    UNKNOWN_TOOL = "unknown_tool"
    MALFORMED_ARGUMENTS = "malformed_arguments"
    MISSING_REQUIRED_FIELD = "missing_required_field"
    UNEXPECTED_FIELD = "unexpected_field"
    INVALID_FIELD_VALUE = "invalid_field_value"


@dataclass(frozen=True, slots=True)
class FieldIssue:
    """One field-level problem, in terms the model can act on."""

    field: str
    problem: str

    def __str__(self) -> str:
        return f"{self.field}: {self.problem}"


@dataclass(frozen=True, slots=True)
class ValidatedToolCall:
    """A call that passed the gate. Holds a parsed, schema-conformant argument model."""

    tool: Tool[Any]
    arguments: BaseModel
    call_id: str

    @property
    def tool_name(self) -> str:
        return self.tool.name


@dataclass(frozen=True, slots=True)
class ToolCallRejected:
    """A call that failed the gate, with enough detail to correct it."""

    tool_name: str
    call_id: str
    reason: RejectionReason
    message: str
    issues: tuple[FieldIssue, ...] = ()

    def repair_hint(self) -> str:
        """A correction message suitable for feeding back to the model.

        Phrased as an instruction rather than a stack trace, because its audience is a
        language model deciding what to emit next.
        """
        if self.reason is RejectionReason.UNKNOWN_TOOL:
            return (
                f"The tool '{self.tool_name}' does not exist. {self.message} "
                "Call one of the available tools instead."
            )
        if self.reason is RejectionReason.MALFORMED_ARGUMENTS:
            return (
                f"The arguments for '{self.tool_name}' were not a valid JSON object. "
                "Re-issue the call with a well-formed JSON object of arguments."
            )
        detail = "; ".join(str(issue) for issue in self.issues) or self.message
        return (
            f"The arguments for '{self.tool_name}' were rejected: {detail}. "
            "Re-issue the call with corrected arguments."
        )

    def as_log_fields(self) -> dict[str, object]:
        return {
            "tool_name": self.tool_name,
            "call_id": self.call_id,
            "reason": self.reason.value,
            "issues": [str(issue) for issue in self.issues],
        }


#: The gate's outcome. Callers branch on the type rather than on a success flag.
type ToolCallValidation = ValidatedToolCall | ToolCallRejected


def _overall_reason(issues: Sequence[FieldIssue], raw_types: Sequence[str]) -> RejectionReason:
    """Pick the single reason that best characterises a multi-error rejection.

    Missing fields outrank the rest: they are the most actionable thing to tell the model, and
    a missing field frequently causes the downstream type errors reported alongside it.
    """
    if "missing" in raw_types:
        return RejectionReason.MISSING_REQUIRED_FIELD
    if "extra_forbidden" in raw_types:
        return RejectionReason.UNEXPECTED_FIELD
    return RejectionReason.INVALID_FIELD_VALUE


def _issues_from(error: ValidationError) -> tuple[tuple[FieldIssue, ...], tuple[str, ...]]:
    issues: list[FieldIssue] = []
    raw_types: list[str] = []
    for detail in error.errors():
        location = ".".join(str(part) for part in detail["loc"]) or "<root>"
        issues.append(FieldIssue(field=location, problem=detail["msg"]))
        raw_types.append(str(detail["type"]))
    return tuple(issues), tuple(raw_types)


def validate_tool_call(
    proposed: ProposedToolCall,
    *,
    registry: ToolRegistry,
) -> ToolCallValidation:
    """Validate one proposed call against the registry and its schema.

    Checks, in order: the tool exists, the argument payload decoded at all, and the arguments
    satisfy the tool's Pydantic schema (including ``extra="forbid"``, which catches invented
    arguments rather than silently discarding them).
    """
    tool = registry.get(proposed.tool_name)
    if tool is None:
        rejection = ToolCallRejected(
            tool_name=proposed.tool_name,
            call_id=proposed.call_id,
            reason=RejectionReason.UNKNOWN_TOOL,
            message=f"Available tools: {', '.join(registry.names)}.",
        )
        _logger.warning("tool_call.rejected", **rejection.as_log_fields())
        return rejection

    if proposed.raw_arguments is not None and not proposed.arguments:
        rejection = ToolCallRejected(
            tool_name=proposed.tool_name,
            call_id=proposed.call_id,
            reason=RejectionReason.MALFORMED_ARGUMENTS,
            message=(
                "argument payload could not be decoded as JSON: "
                f"{proposed.raw_arguments[:_MAX_RAW_EXCERPT_CHARS]!r}"
            ),
        )
        _logger.warning("tool_call.rejected", **rejection.as_log_fields())
        return rejection

    try:
        arguments = tool.args_schema.model_validate(dict(proposed.arguments))
    except ValidationError as error:
        issues, raw_types = _issues_from(error)
        rejection = ToolCallRejected(
            tool_name=proposed.tool_name,
            call_id=proposed.call_id,
            reason=_overall_reason(issues, raw_types),
            message=f"{len(issues)} argument error(s).",
            issues=issues,
        )
        _logger.warning("tool_call.rejected", **rejection.as_log_fields())
        return rejection

    _logger.debug(
        "tool_call.validated",
        tool_name=tool.name,
        call_id=proposed.call_id,
        arguments=arguments.model_dump(),
    )
    return ValidatedToolCall(tool=tool, arguments=arguments, call_id=proposed.call_id)


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """The outcome of validating a batch of proposed calls."""

    accepted: tuple[ValidatedToolCall, ...] = ()
    rejected: tuple[ToolCallRejected, ...] = ()

    @property
    def has_accepted(self) -> bool:
        return bool(self.accepted)

    @property
    def total(self) -> int:
        return len(self.accepted) + len(self.rejected)

    def repair_hints(self) -> tuple[str, ...]:
        return tuple(rejection.repair_hint() for rejection in self.rejected)


def validate_tool_calls(
    proposed: Sequence[ProposedToolCall],
    *,
    registry: ToolRegistry,
) -> ValidationReport:
    """Validate a batch, partitioning it into accepted and rejected calls.

    Partitioning rather than failing fast is the point: a model that proposes four calls and
    gets one wrong should still have the other three executed. Aborting the batch would
    discard good work and turn a recoverable mistake into a failed run.
    """
    accepted: list[ValidatedToolCall] = []
    rejected: list[ToolCallRejected] = []

    for call in proposed:
        outcome = validate_tool_call(call, registry=registry)
        if isinstance(outcome, ValidatedToolCall):
            accepted.append(outcome)
        else:
            rejected.append(outcome)

    return ValidationReport(accepted=tuple(accepted), rejected=tuple(rejected))

"""Retrieval node.

Sole responsibility: fetch (spec S3.2). It selects tools, validates their arguments, executes
them, and hands back **raw, unparsed** payloads. It never inspects what came back.

This is where spec S3.3 is enforced end to end::

    LLM proposes  ->  validation gate  ->  repair retry  ->  deterministic fallback  ->  execute

Each stage exists because the one before it can fail. The model may hallucinate a tool or
malform an argument, so the gate rejects it. A rejection is actionable, so the model gets one
corrective attempt carrying the gate's own repair hints. And if it still cannot produce a
valid call, the node falls back to constructing the calls itself from the planner's declared
capabilities - because a model that cannot format arguments is not a reason to return no data
for a query we already know how to measure.

The fallback calls go through the identical validation gate. There is no code path to the
network that skips it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from sightline.agents.contracts.plan import PlannedRetrieval
from sightline.agents.contracts.retrieval import RetrievalOutcome
from sightline.agents.dependencies import NodeDependencies
from sightline.agents.graph.names import NodeName
from sightline.agents.graph.state import PipelineState, RetrievalBranchState
from sightline.agents.prompts import retrieval as prompts
from sightline.application.ports.errors import ExternalServiceError
from sightline.application.ports.llm_provider import LLMError
from sightline.application.ports.search_data_provider import RawApiResult
from sightline.config.settings import DataForSEOSettings
from sightline.domain.entities.profile import Profile
from sightline.domain.value_objects.enums import RetrievalKind
from sightline.observability.logging import get_logger
from sightline.observability.metrics import current_metrics
from sightline.observability.tracing import node_span
from sightline.tools.registry import ToolRegistry
from sightline.tools.validation import ToolCallRejected, ValidatedToolCall, validate_tool_calls

_logger = get_logger(__name__)

#: Arguments the node builds itself when the model cannot produce a valid call. Keyed by the
#: capability the planner asked for, so the fallback covers exactly what was requested.
#:
#: Public because the offline demo responder reuses it to emit the same well-formed calls a
#: competent model would. One definition, so the two cannot drift apart.
DEFAULT_TOOL_ARGUMENTS: dict[RetrievalKind, Callable[[str, DataForSEOSettings], dict[str, Any]]] = {
    RetrievalKind.ORGANIC_SERP: lambda query, settings: {
        "keyword": query,
        "location_code": settings.default_location_code,
        "language_code": settings.default_language_code,
    },
    RetrievalKind.AI_OVERVIEW: lambda query, settings: {
        "keyword": query,
        "location_code": settings.default_location_code,
        "language_code": settings.default_language_code,
    },
    RetrievalKind.LLM_VISIBILITY: lambda query, _settings: {
        # Phrased as a question: the endpoint measures how an assistant answers a buyer, and a
        # bare keyword produces a definition rather than a recommendation.
        "prompt": f"Which options are best for: {query}?",
    },
    RetrievalKind.KEYWORD_METRICS: lambda query, settings: {
        "keywords": [query],
        "location_code": settings.default_location_code,
        "language_code": settings.default_language_code,
    },
}


def _build_fallback_calls(
    sub_query: PlannedRetrieval,
    registry: ToolRegistry,
    settings: DataForSEOSettings,
) -> list[ValidatedToolCall]:
    """Construct tool calls directly from the planner's declared capabilities.

    Still validated through each tool's own schema, so this path cannot smuggle an argument
    the gate would have rejected.
    """
    calls: list[ValidatedToolCall] = []
    for index, kind in enumerate(sub_query.retrieval_kinds):
        tool = registry.for_kind(kind)
        builder = DEFAULT_TOOL_ARGUMENTS.get(kind)
        if tool is None or builder is None:
            continue
        try:
            arguments = tool.args_schema.model_validate(builder(sub_query.query_text, settings))
        except ValueError as exc:  # pragma: no cover - would be a template bug
            # exception(), not error(): this is swallowed rather than re-raised, so without
            # the traceback there would be no record of where a bad template came from.
            _logger.exception("retrieval.fallback_args_invalid", kind=kind.value, error=str(exc))
            continue
        calls.append(ValidatedToolCall(tool=tool, arguments=arguments, call_id=f"fallback-{index}"))
    return calls


async def _select_tool_calls(
    *,
    profile: Profile,
    sub_query: PlannedRetrieval,
    registry: ToolRegistry,
    deps: NodeDependencies,
) -> tuple[list[ValidatedToolCall], list[ToolCallRejected], int, int, bool]:
    """Ask the model which tools to call, validate, and repair or fall back as needed."""
    system_prompt = prompts.SYSTEM_PROMPT
    user_prompt = prompts.build_user_prompt(profile, sub_query)
    prompt_tokens = completion_tokens = 0
    all_rejected: list[ToolCallRejected] = []

    for attempt in (1, 2):
        completion = await deps.llm.propose_tool_calls(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            tools=registry.specs(),
        )
        prompt_tokens += completion.usage.prompt_tokens
        completion_tokens += completion.usage.completion_tokens

        report = validate_tool_calls(completion.calls, registry=registry)
        all_rejected.extend(report.rejected)

        if report.has_accepted:
            return list(report.accepted), all_rejected, prompt_tokens, completion_tokens, False

        if attempt == 1 and report.rejected:
            # One corrective attempt, carrying the gate's own repair hints. The model is told
            # exactly which field was wrong and why, which is far more likely to succeed than
            # simply asking again.
            _logger.warning(
                "retrieval.repair_attempt",
                query=sub_query.query_text,
                rejected=len(report.rejected),
            )
            hints = "\n".join(f"- {hint}" for hint in report.repair_hints())
            user_prompt = f"{user_prompt}\n\nYour previous call was rejected:\n{hints}"

    return [], all_rejected, prompt_tokens, completion_tokens, True


async def perform_retrieval(
    *,
    profile: Profile,
    sub_query: PlannedRetrieval,
    deps: NodeDependencies,
) -> RetrievalOutcome:
    """Measure one sub-query. Shared by the fan-out branch and the recheck subgraph.

    Never raises for a dependency failure: a branch that could not be measured comes back as
    an outcome carrying its error, so the graph can route on how many branches succeeded
    rather than losing the whole run to one bad query.
    """
    registry = deps.tools.subset(sub_query.retrieval_kinds)
    if not len(registry):
        return RetrievalOutcome(
            sub_query=sub_query,
            error="no tool serves the requested retrieval kinds",
            error_type="ConfigurationError",
        )

    try:
        accepted, rejected, prompt_tokens, completion_tokens, exhausted = await _select_tool_calls(
            profile=profile, sub_query=sub_query, registry=registry, deps=deps
        )
    except LLMError as exc:
        _logger.warning("retrieval.llm_failed", query=sub_query.query_text, error=str(exc))
        accepted, rejected, prompt_tokens, completion_tokens, exhausted = [], [], 0, 0, True

    used_fallback = False
    if not accepted and exhausted:
        accepted = _build_fallback_calls(sub_query, registry, deps.dataforseo)
        used_fallback = bool(accepted)
        if used_fallback:
            _logger.warning(
                "retrieval.deterministic_fallback",
                query=sub_query.query_text,
                calls=len(accepted),
            )

    results: list[RawApiResult] = []
    tool_names: list[str] = []
    attempts = 1
    failure: ExternalServiceError | None = None

    for call in accepted:
        try:
            result = await call.tool.execute(deps.search_provider, call.arguments)
        except ExternalServiceError as exc:
            # One failing capability must not discard the ones that succeeded: a SERP result
            # is still useful when the keyword-metrics call was rate limited.
            failure = exc
            _logger.warning("retrieval.tool_failed", tool=call.tool_name, **exc.as_log_fields())
            continue
        results.append(result)
        tool_names.append(call.tool_name)
        attempts = max(attempts, result.attempts)

    error = None if results else (str(failure) if failure else "no usable tool call produced")
    return RetrievalOutcome(
        sub_query=sub_query,
        results=tuple(results),
        rejected_calls=tuple(rejected),
        error=error,
        error_type=type(failure).__name__ if failure and not results else None,
        attempts=attempts,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        tool_names=tuple(tool_names) + (("<deterministic-fallback>",) if used_fallback else ()),
    )


async def retrieval_node(state: RetrievalBranchState, *, deps: NodeDependencies) -> PipelineState:
    """One fan-out branch. Returns a single-element list that the reducer accumulates."""
    profile = state["profile"]
    sub_query = state["sub_query"]

    async with node_span(
        NodeName.RETRIEVAL,
        metrics=current_metrics(),
        inputs={
            "query": sub_query.query_text,
            "kinds": [kind.value for kind in sub_query.retrieval_kinds],
            "branch": state["branch_index"],
        },
    ) as span:
        outcome = await perform_retrieval(profile=profile, sub_query=sub_query, deps=deps)
        for _ in range(outcome.retry_count):
            span.record_retry()
        span.set_output(
            succeeded=outcome.succeeded,
            payloads=len(outcome.results),
            tools=list(outcome.tool_names),
            rejected_calls=len(outcome.rejected_calls),
            error=outcome.error,
        )

    return PipelineState(
        retrieval_outcomes=[outcome],
        errors=[f"retrieval[{sub_query.query_text}]: {outcome.error}"] if outcome.error else [],
        prompt_tokens=outcome.prompt_tokens,
        completion_tokens=outcome.completion_tokens,
    )


async def single_retrieval_node(state: PipelineState, *, deps: NodeDependencies) -> PipelineState:
    """Recheck-subgraph entry: measure the one sub-query in the state's plan.

    Delegates to the same :func:`perform_retrieval` the fan-out branch uses, which is what
    makes ``POST /queries/{uuid}/recheck`` a genuine re-entry into the pipeline rather than a
    parallel implementation that can drift.
    """
    plan = state.get("plan")
    sub_queries: Sequence[PlannedRetrieval] = plan.sub_queries if plan else ()
    if not sub_queries:
        return PipelineState(errors=["recheck: no sub-query supplied"])

    branch = RetrievalBranchState(
        profile=state["profile"],
        sub_query=sub_queries[0],
        run_uuid=state["run_uuid"],
        branch_index=0,
    )
    return await retrieval_node(branch, deps=deps)

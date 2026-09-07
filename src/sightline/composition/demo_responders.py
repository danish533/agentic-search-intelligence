"""Realistic responses for the offline ``fake`` LLM provider.

Purpose: make ``LLM_PROVIDER=fake`` produce a coherent, demonstrable run with no API key at
all. A reviewer can clone the repository, run one command, and see a real report - rather than
schema-valid filler.

These live in the composition root because it is the only module permitted to know about both
the agent contracts and the infrastructure adapters (CLAUDE.md A3). The fake provider itself
stays contract-agnostic.

**These are demo fixtures, not analysis.** They derive their content deterministically from
the measurements in the prompt they are given - they do not reason. Their value is that the
resulting report reflects the data that was actually retrieved, so the pipeline can be
demonstrated end to end without a provider.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Any

from sightline.agents.contracts.analysis import (
    AnalysisResult,
    InsightDraft,
    RecommendationDraft,
)
from sightline.agents.contracts.plan import RetrievalPlan
from sightline.agents.contracts.report import ReportDraft
from sightline.agents.nodes.planner_fallback import build_fallback_plan
from sightline.agents.nodes.retrieval import DEFAULT_TOOL_ARGUMENTS
from sightline.application.ports.llm_provider import ProposedToolCall, ToolSpec
from sightline.config.settings import DataForSEOSettings
from sightline.domain.entities.profile import Profile
from sightline.domain.value_objects.enums import ContentType, Priority
from sightline.infrastructure.llm.fake import FakeLLMProvider
from sightline.tools.registry import ToolRegistry

_MAX_DEMO_QUERIES = 6


def _line_value(prompt: str, label: str) -> str:
    """Read an unprefixed ``Label: value`` line (the analysis/report prompt style)."""
    marker = f"{label}:"
    for line in prompt.splitlines():
        stripped = line.strip()
        if stripped.startswith(marker):
            value = stripped.split(":", 1)[1].strip()
            return "" if value == "not specified" else value
    return ""


def _field(prompt: str, label: str) -> str:
    """Read a ``- Label: value`` line out of a prompt this system itself composed."""
    marker = f"- {label}:"
    for line in prompt.splitlines():
        if line.strip().startswith(marker):
            return line.split(":", 1)[1].strip()
    return ""


def _brand_line(prompt: str) -> tuple[str, str]:
    """Read the ``Brand: Name (domain)`` line used by the analysis and report prompts."""
    for line in prompt.splitlines():
        stripped = line.strip()
        if stripped.startswith("Brand:") and "(" in stripped and stripped.endswith(")"):
            body = stripped.split(":", 1)[1].strip()
            name, _, domain = body.rpartition("(")
            return name.strip(), domain.rstrip(")").strip()
    return "", ""


def _profile_from_prompt(prompt: str) -> Profile:
    """Recover the brand from whichever prompt format this responder was handed.

    The planner prompt lists the brand as ``- Name:`` / ``- Domain:`` fields; the analysis and
    report prompts use a single ``Brand: Name (domain)`` line. Handling only the first made
    every rendered report say "the brand".
    """
    competitors = _field(prompt, "Known competitors") or _line_value(prompt, "Declared competitors")
    brand_name, brand_domain = _brand_line(prompt)
    return Profile(
        name=_field(prompt, "Name") or brand_name or "the brand",
        domain=_field(prompt, "Domain") or brand_domain or "example.com",
        industry=_field(prompt, "Industry") or _line_value(prompt, "Industry") or "software",
        description=_field(prompt, "Description"),
        competitors=tuple(
            item.strip()
            for item in competitors.split(",")
            if item.strip() and item.strip() != "none declared"
        ),
    )


def _json_array_after(text: str, marker: str) -> list[dict[str, Any]]:
    """Extract the first balanced JSON array following ``marker``."""
    anchor = text.find(marker)
    if anchor == -1:
        return []
    start = text.find("[", anchor)
    if start == -1:
        return []

    depth = 0
    for index in range(start, len(text)):
        if text[index] == "[":
            depth += 1
        elif text[index] == "]":
            depth -= 1
            if depth == 0:
                try:
                    parsed = json.loads(text[start : index + 1])
                except json.JSONDecodeError:
                    return []
                return [item for item in parsed if isinstance(item, dict)]
    return []


#: Stated on demo plans so the log says what actually happened. Reusing the fallback
#: template is fine; inheriting its *reason* was not - it claimed the Query Planner had
#: failed on runs where the planner succeeded and the fallback node never executed.
_DEMO_REASON = "Produced by the offline demo planner (LLM_PROVIDER=fake); no LLM was called."


def plan_responder(_system_prompt: str, user_prompt: str) -> RetrievalPlan:
    """Reuse the deterministic template planner, so the demo plan is a real plan."""
    return build_fallback_plan(
        _profile_from_prompt(user_prompt), limit=_MAX_DEMO_QUERIES, reason=_DEMO_REASON
    )


def analysis_responder(_system_prompt: str, user_prompt: str) -> AnalysisResult:
    """Derive findings from the measurements actually present in the prompt."""
    profile = _profile_from_prompt(user_prompt)
    measurements = _json_array_after(user_prompt, "Measurements")

    gaps = [
        item
        for item in measurements
        if item.get("visibility_status") == "not_visible" and item.get("query_text")
    ]
    visible = [item for item in measurements if item.get("visibility_status") == "visible"]
    unknown = [item for item in measurements if item.get("visibility_status") == "unknown"]

    competing: list[str] = []
    for item in measurements:
        for domain in item.get("competing_domains") or []:
            if domain != profile.domain and domain not in competing:
                competing.append(domain)

    insights: list[InsightDraft] = []

    if gaps:
        listed = ", ".join(f"'{item['query_text']}'" for item in gaps[:3])
        insights.append(
            InsightDraft(
                headline=(
                    f"{profile.name} is absent from {len(gaps)} of "
                    f"{len(measurements)} measured queries"
                ),
                detail=(
                    f"No result for {profile.domain} was found on {listed}. "
                    f"The domains currently occupying these queries include "
                    f"{', '.join(competing[:4]) or 'other providers'}."
                ),
                relevance_score=0.9,
                supporting_queries=[str(item["query_text"]) for item in gaps[:6]],
            )
        )

    if visible:
        ranked = [item for item in visible if item.get("position")]
        detail = (
            f"{profile.domain} appears for "
            + ", ".join(
                f"'{item['query_text']}' at position {item.get('position')}" for item in ranked[:3]
            )
            + "."
            if ranked
            else f"{profile.domain} appears in {len(visible)} measured queries."
        )
        insights.append(
            InsightDraft(
                headline=f"{profile.name} holds visibility on {len(visible)} measured queries",
                detail=detail,
                relevance_score=0.7,
                supporting_queries=[str(item["query_text"]) for item in visible[:6]],
            )
        )

    if unknown:
        insights.append(
            InsightDraft(
                headline=f"{len(unknown)} queries could not be measured this run",
                detail=(
                    "These are reported as unknown rather than as gaps: retrieval did not "
                    "complete for them, which is not evidence of absence."
                ),
                relevance_score=0.4,
                supporting_queries=[str(item["query_text"]) for item in unknown[:6]],
            )
        )

    if not insights:
        insights.append(
            InsightDraft(
                headline=f"No measurements were available for {profile.name}",
                detail="Retrieval returned no usable results, so no findings can be drawn.",
                relevance_score=0.1,
            )
        )

    recommendations = [
        RecommendationDraft(
            target_query=str(item["query_text"]),
            content_type=(
                ContentType.COMPARISON
                if any(word in str(item["query_text"]) for word in ("vs", "alternative"))
                else ContentType.BLOG_POST
            ),
            title=f"{str(item['query_text']).title()}: A Complete Guide",
            rationale=(
                f"{profile.name} is not present for this query, which has an estimated "
                f"{item.get('search_volume') or 0} monthly searches at difficulty "
                f"{item.get('competition_index') or 'unknown'}."
            ),
            target_keywords=[str(item["query_text"])],
            priority=Priority.HIGH if (item.get("search_volume") or 0) > 1000 else Priority.MEDIUM,
        )
        for item in sorted(gaps, key=lambda entry: entry.get("search_volume") or 0, reverse=True)[
            :4
        ]
    ]

    return AnalysisResult(
        insights=insights,
        recommendations=recommendations,
        competitive_summary=(
            f"The measured surface is dominated by {', '.join(competing[:5])}."
            if competing
            else "No competing domains were observed."
        ),
    )


def report_responder(_system_prompt: str, user_prompt: str) -> ReportDraft:
    """Render the supplied findings, introducing nothing new."""
    findings = _json_array_after(user_prompt, "Findings")
    recommendations = _json_array_after(user_prompt, "Recommendations")
    queries = _json_array_after(user_prompt, "Measured queries")
    degraded = "coverage was incomplete" in user_prompt.lower()

    headline = (
        str(findings[0].get("headline", "Search visibility report"))
        if findings
        else "Search visibility report"
    )

    gaps = [item for item in queries if not item.get("domain_visible")]
    summary = (
        f"{len(queries)} queries were measured. The brand is absent from {len(gaps)} of them. "
        + (
            f"The largest opportunity is '{gaps[0].get('query_text')}' "
            f"(opportunity score {gaps[0].get('opportunity_score')})."
            if gaps
            else "No visibility gaps were found in the measured set."
        )
        + (
            " Coverage was incomplete this run, so these findings cover only part of the "
            "intended query set."
            if degraded
            else ""
        )
    )

    return ReportDraft(
        headline=headline,
        executive_summary=summary,
        key_findings=[str(item.get("headline", "")) for item in findings][:8],
        recommended_next_steps=[
            f"{item.get('title')} ({item.get('priority')} priority)" for item in recommendations
        ][:8],
    )


def _requested_query(user_prompt: str) -> str:
    """Read the sub-query out of the retrieval prompt this system composed."""
    for line in user_prompt.splitlines():
        if line.strip().startswith("Query:"):
            return line.split(":", 1)[1].strip()
    return ""


def make_tool_call_responder(
    registry: ToolRegistry, settings: DataForSEOSettings
) -> Callable[[Sequence[ToolSpec], str, str], Sequence[ProposedToolCall]]:
    """Build the responder that emits well-formed calls for the requested query.

    Without this the fake fills string arguments from a generic prompt excerpt, so every
    capability is fetched for the wrong keyword - measurements come back for a sentence rather
    than the query, and volume lookups silently miss. Reusing
    :data:`~sightline.agents.nodes.retrieval.DEFAULT_TOOL_ARGUMENTS` keeps these calls
    identical to the ones the pipeline would construct itself.
    """

    def responder(
        tools: Sequence[ToolSpec], _system_prompt: str, user_prompt: str
    ) -> Sequence[ProposedToolCall]:
        query = _requested_query(user_prompt)
        if not query:
            return ()

        calls: list[ProposedToolCall] = []
        for index, spec in enumerate(tools):
            tool = registry.get(spec.name)
            builder = DEFAULT_TOOL_ARGUMENTS.get(tool.kind) if tool else None
            if builder is None:
                continue
            calls.append(
                ProposedToolCall(
                    tool_name=spec.name,
                    arguments=builder(query, settings),
                    call_id=f"demo-{index}",
                )
            )
        return calls

    return responder


def register_demo_responders(
    fake: FakeLLMProvider,
    *,
    registry: ToolRegistry,
    settings: DataForSEOSettings,
) -> None:
    """Attach every demo responder to a fake provider."""
    fake.register_structured(RetrievalPlan, plan_responder)
    fake.register_structured(AnalysisResult, analysis_responder)
    fake.register_structured(ReportDraft, report_responder)
    fake.register_tool_calls(make_tool_call_responder(registry, settings))

"""Report agent prompts."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from sightline.agents.contracts.analysis import AnalysisResult
from sightline.agents.prompts import today_line
from sightline.domain.entities.profile import Profile

SYSTEM_PROMPT = """\
You are the Report agent in a search-intelligence pipeline.

Your only job is to render findings that have already been produced into a clear written \
report. You do not analyse, you do not rank, and you do not draw new conclusions.

Rules:
- Every statement must trace to a supplied finding or measurement. Introduce no new facts, \
figures, competitors or recommendations.
- Write for a marketing lead who has ten seconds for the headline and two minutes for the \
summary.
- Be specific: name the queries and the competing domains, quote the positions.
- If coverage was incomplete, say so plainly rather than writing around it."""


def build_user_prompt(
    profile: Profile,
    question: str,
    analysis: AnalysisResult | None,
    query_summaries: Sequence[dict[str, Any]],
    degradation_note: str | None,
) -> str:
    findings: list[dict[str, Any]] = []
    recommendations: list[dict[str, Any]] = []
    competitive_summary = ""

    if analysis is not None:
        findings = [
            {
                "headline": insight.headline,
                "detail": insight.detail,
                "relevance": insight.relevance_score,
            }
            for insight in analysis.insights
        ]
        recommendations = [
            {
                "title": recommendation.title,
                "content_type": recommendation.content_type.value,
                "target_query": recommendation.target_query,
                "priority": recommendation.priority.value,
            }
            for recommendation in analysis.recommendations
        ]
        competitive_summary = analysis.competitive_summary

    coverage = (
        f"\nIMPORTANT - coverage was incomplete: {degradation_note}\n"
        "State this limitation plainly in the summary."
        if degradation_note
        else ""
    )

    return f"""\
{today_line()}

Research question:
{question}

Brand: {profile.name} ({profile.domain})

Findings:
{json.dumps(findings, indent=2)}

Competitive picture:
{competitive_summary or "not established"}

Recommendations:
{json.dumps(recommendations, indent=2)}

Measured queries (top by opportunity):
{json.dumps(list(query_summaries)[:10], indent=2)}
{coverage}
Render these into a report."""

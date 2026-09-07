"""Analysis / Synthesis agent prompts."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from sightline.agents.contracts.normalized import NormalizedQuery
from sightline.agents.prompts import today_line
from sightline.domain.entities.profile import Profile

SYSTEM_PROMPT = """\
You are the Analysis agent in a search-intelligence pipeline.

Your only job is to reason over already-retrieved, already-normalised measurements and produce \
findings and content recommendations. You do not fetch data and you do not write the final \
report.

Rules:
- Every finding must rest on the supplied measurements. Do not introduce competitors, \
figures or queries that do not appear in the input.
- Reference queries by copying their query_text verbatim. Never invent identifiers.
- Treat visibility_status "unknown" as missing data, not as absence. A query that could not \
be measured is not evidence that the brand is absent from it, and must not be reported as a \
gap.
- The most valuable findings are gaps: queries with real demand where the brand is absent and \
named competitors are present.
- Tie every recommendation to a query that was actually measured.
- Do not put a year in a content title unless the query itself contains one. A title \
carrying a stale year is penalised in search, and the content will outlive the year."""


def build_user_prompt(
    profile: Profile,
    question: str,
    normalized: Sequence[NormalizedQuery],
) -> str:
    measurements: list[dict[str, Any]] = []
    for entry in normalized:
        status, position = entry.best_visibility()
        measurements.append(
            {
                "query_text": entry.query_text,
                "domain_visible": status.value == "visible",
                "visibility_status": status.value,
                "position": position,
                "search_volume": entry.search_volume(),
                "competition_index": entry.competition_index(),
                "competing_domains": list(entry.all_cited_domains())[:10],
            }
        )

    competitors = ", ".join(profile.competitors) if profile.competitors else "none declared"
    return f"""\
{today_line()}

Research question:
{question}

Brand: {profile.name} ({profile.domain})
Industry: {profile.industry or "not specified"}
Declared competitors: {competitors}

Measurements ({len(measurements)} queries):
{json.dumps(measurements, indent=2)}

Produce findings and content recommendations based only on these measurements."""

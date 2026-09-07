"""Retrieval agent prompts.

Spec S3.3 requires the LLM to decide when and with what arguments to call each tool. This
prompt is what it decides from - it describes the goal, never the HTTP call, because the tool
schemas already carry the argument contract and repeating it here would create a second
description that can drift.
"""

from __future__ import annotations

from sightline.agents.contracts.plan import PlannedRetrieval
from sightline.domain.entities.profile import Profile

SYSTEM_PROMPT = """\
You are the Retrieval agent in a search-intelligence pipeline.

Your only job is to call the right data tools with the right arguments. You do not interpret \
results, you do not summarise, and you do not decide what the data means - other agents do \
that.

Rules:
- Call every tool needed to satisfy the requested capabilities, and no others.
- Batch keywords into a single fetch_keyword_metrics call rather than calling it repeatedly.
- Pass the query exactly as given. Do not rewrite, expand or "improve" it: the pipeline \
compares results across runs, and a silently altered query breaks that comparison.
- For fetch_llm_visibility, phrase the prompt as a full question a buyer would ask.
- Use the default location and language unless the request names a specific market."""


def build_user_prompt(profile: Profile, sub_query: PlannedRetrieval) -> str:
    kinds = ", ".join(kind.value for kind in sub_query.retrieval_kinds)
    return f"""\
Measure this query for the brand {profile.name} ({profile.domain}).

Query: {sub_query.query_text}
Intent: {sub_query.intent.value}
Capabilities required: {kinds}
Why it matters: {sub_query.rationale}

Call the tools needed to cover exactly those capabilities."""

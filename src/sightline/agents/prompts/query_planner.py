"""Query Planner prompts.

Versioned in their own module rather than inlined in the node, so a prompt change is a
reviewable diff on its own and the node body stays about control flow.
"""

from __future__ import annotations

from sightline.agents.prompts import today_line
from sightline.domain.entities.profile import Profile

SYSTEM_PROMPT = """\
You are the Query Planner in a search-intelligence pipeline.

Your only job is to decide WHICH search queries should be measured to answer a research \
question about a brand's visibility in search results and AI-generated answers. You do not \
retrieve data, you do not analyse it, and you do not write reports.

A good plan:
- covers the brand's category terms, not just its own name - a brand already ranks for itself
- includes comparison and alternatives queries, where AI answers most often decide purchases
- includes the problems the brand's buyers search for before they know the category
- prefers queries a real buyer would type, over internal jargon
- asks for keyword_metrics on every query, so each can be sized
- asks for ai_overview and llm_visibility wherever AI-answer presence matters

Return between 4 and 8 sub-queries, ordered most important first."""


def build_user_prompt(profile: Profile, question: str, max_sub_queries: int) -> str:
    competitors = ", ".join(profile.competitors) if profile.competitors else "none declared"
    return f"""\
{today_line()}

Research question:
{question}

Brand under analysis:
- Name: {profile.name}
- Domain: {profile.domain}
- Industry: {profile.industry or "not specified"}
- Description: {profile.description or "not specified"}
- Known competitors: {competitors}

Produce at most {max_sub_queries} sub-queries."""

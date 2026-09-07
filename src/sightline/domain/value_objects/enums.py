"""Domain enumerations.

``StrEnum`` is used throughout so that values serialise to the exact literals the assessment
spec prints (e.g. ``"not_visible"``, ``"blog_post"``) without a translation table.
"""

from __future__ import annotations

from enum import StrEnum


class RunStatus(StrEnum):
    """Lifecycle of a single DAG execution.

    ``PARTIAL`` is load-bearing, not decorative: it is the state the graph lands in when the
    degradation path produced a usable report from incomplete retrieval (CLAUDE.md R5).
    """

    RUNNING = "running"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"

    @property
    def is_terminal(self) -> bool:
        return self is not RunStatus.RUNNING


class VisibilityStatus(StrEnum):
    """Whether the profile's domain was found in the answer set for a query.

    ``UNKNOWN`` is distinct from ``NOT_VISIBLE``: it means retrieval never produced a usable
    answer (fallback path taken), so absence of evidence must not be reported as evidence of
    absence.
    """

    VISIBLE = "visible"
    NOT_VISIBLE = "not_visible"
    UNKNOWN = "unknown"


class RetrievalKind(StrEnum):
    """The logical DataForSEO capability a planned retrieval call requires.

    The Query Planner emits these; the Retrieval node maps each to exactly one tool
    (spec S3.4: one tool per logical API call).
    """

    ORGANIC_SERP = "organic_serp"
    AI_OVERVIEW = "ai_overview"
    LLM_VISIBILITY = "llm_visibility"
    KEYWORD_METRICS = "keyword_metrics"


class QueryIntent(StrEnum):
    """What a searcher is trying to accomplish with a query.

    Drives which retrieval capabilities are worth spending on: a comparison query is where AI
    answers most often decide a purchase, while a navigational one measures brand defence.
    """

    INFORMATIONAL = "informational"
    COMMERCIAL = "commercial"
    COMPARISON = "comparison"
    NAVIGATIONAL = "navigational"


class ContentType(StrEnum):
    """Deliverable type proposed by a recommendation."""

    BLOG_POST = "blog_post"
    LANDING_PAGE = "landing_page"
    FAQ = "faq"
    COMPARISON = "comparison"
    CASE_STUDY = "case_study"


class Priority(StrEnum):
    """Recommendation urgency."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class DegradationReason(StrEnum):
    """Why a run degraded to ``PARTIAL``. Surfaced in the report's error flag."""

    NO_PLAN_PRODUCED = "no_plan_produced"
    ALL_RETRIEVALS_FAILED = "all_retrievals_failed"
    INSUFFICIENT_RETRIEVALS = "insufficient_retrievals"
    NORMALIZATION_EMPTY = "normalization_empty"
    ANALYSIS_UNAVAILABLE = "analysis_unavailable"
    UPSTREAM_CIRCUIT_OPEN = "upstream_circuit_open"

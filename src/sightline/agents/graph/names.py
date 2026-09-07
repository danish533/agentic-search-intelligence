"""Canonical node names.

Spec S3.1 requires "clearly named nodes and edges". An enum rather than string literals so a
typo is a startup failure instead of an unreachable node discovered at runtime, and so the
node set is enumerable - which is what lets the README diagram and the metrics summary be
generated from the graph rather than maintained alongside it.
"""

from __future__ import annotations

from enum import StrEnum


class NodeName(StrEnum):
    """Every node in the pipeline graph."""

    INGEST = "ingest"
    QUERY_PLANNER = "query_planner"
    PLANNER_FALLBACK = "planner_fallback"
    RETRIEVAL = "retrieval"
    NORMALIZATION = "normalization"
    ANALYSIS = "analysis"
    DEGRADATION = "degradation"
    REPORT = "report"

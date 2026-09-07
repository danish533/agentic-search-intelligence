"""Extraction / Normalization node.

Sole responsibility: parse raw provider payloads into a clean, typed schema (spec S3.2). It
performs no I/O and draws no conclusions - it records what was observed and stops.

Parsing is defensive throughout. Every payload is treated as untrusted shape: a missing key, a
null where a list was expected, or an item type that did not exist when this was written must
degrade that one observation, never abort the run. The alternative - a `KeyError` deep in a
parser - would fail the whole pipeline over one unexpected field.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any
from urllib.parse import urlparse
from uuid import UUID

from sightline.agents.contracts.normalized import NormalizedObservation, NormalizedQuery
from sightline.agents.contracts.retrieval import RetrievalOutcome
from sightline.agents.dependencies import NodeDependencies
from sightline.agents.graph.names import NodeName
from sightline.agents.graph.state import PipelineState
from sightline.application.ports.search_data_provider import RawApiResult
from sightline.domain.entities.discovered_query import DiscoveredQuery
from sightline.domain.entities.profile import Profile, normalize_domain
from sightline.domain.services.opportunity_scoring import calculate_opportunity_score
from sightline.domain.value_objects.enums import RetrievalKind, VisibilityStatus
from sightline.domain.value_objects.scores import CompetitiveDifficulty, SearchVolume
from sightline.observability.logging import get_logger
from sightline.observability.metrics import current_metrics
from sightline.observability.tracing import node_span

_logger = get_logger(__name__)

#: Used when a query's difficulty could not be measured. Neutral by design: assuming 0 would
#: inflate every unmeasured query to the top of the opportunity ranking, and assuming 100
#: would bury them. Neither is knowledge we have.
_UNKNOWN_DIFFICULTY = 50


def _domain_of(url: str | None) -> str:
    if not url:
        return ""
    host = urlparse(url).netloc or url
    return normalize_domain(host)


def _same_domain(candidate: str, target: str) -> bool:
    """Whether ``candidate`` is the target domain or a subdomain of it.

    ``blog.surferseo.com`` is Surfer SEO's visibility; ``notsurferseo.com`` is not. A plain
    substring test would count both.
    """
    left, right = normalize_domain(candidate), normalize_domain(target)
    return bool(left) and (left == right or left.endswith(f".{right}"))


def _result_blocks(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Every ``result`` entry across every task, tolerating nulls at each level."""
    blocks: list[Mapping[str, Any]] = []
    for task in payload.get("tasks") or []:
        if not isinstance(task, Mapping):
            continue
        for block in task.get("result") or []:
            if isinstance(block, Mapping):
                blocks.append(block)
    return blocks


def _items_of(block: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    for item in block.get("items") or []:
        if isinstance(item, Mapping):
            yield item


def _observe(
    *,
    query_text: str,
    kind: RetrievalKind,
    result: RawApiResult,
    ranked: Sequence[tuple[str, int | None]],
    profile: Profile,
    search_volume: int | None = None,
    competition_index: int | None = None,
    looked_for_visibility: bool = True,
) -> NormalizedObservation:
    """Assemble one observation from an ordered list of (domain, position) pairs."""
    status = VisibilityStatus.UNKNOWN
    position: int | None = None

    if looked_for_visibility:
        status = VisibilityStatus.NOT_VISIBLE
        for domain, rank in ranked:
            if _same_domain(domain, profile.domain):
                status = VisibilityStatus.VISIBLE
                position = rank
                break

    competitor_positions: dict[str, int] = {}
    for competitor in profile.competitors:
        for domain, rank in ranked:
            if _same_domain(domain, competitor) and rank is not None:
                competitor_positions.setdefault(competitor, rank)
                break

    return NormalizedObservation(
        query_text=query_text,
        kind=kind,
        source_endpoint=result.endpoint,
        visibility_status=status,
        position=position,
        cited_domains=tuple(domain for domain, _ in ranked if domain),
        competitor_positions=competitor_positions,
        search_volume=search_volume,
        competition_index=competition_index,
        from_mock=result.from_mock,
    )


def parse_organic_serp(
    result: RawApiResult, *, query_text: str, profile: Profile
) -> NormalizedObservation:
    ranked: list[tuple[str, int | None]] = []
    for block in _result_blocks(result.payload):
        for item in _items_of(block):
            if item.get("type") != "organic":
                continue
            domain = str(item.get("domain") or _domain_of(item.get("url")))
            rank = item.get("rank_absolute")
            ranked.append((domain, int(rank) if isinstance(rank, int) else None))

    return _observe(
        query_text=query_text,
        kind=RetrievalKind.ORGANIC_SERP,
        result=result,
        ranked=ranked,
        profile=profile,
    )


def parse_ai_overview(
    result: RawApiResult, *, query_text: str, profile: Profile
) -> NormalizedObservation:
    cited: list[tuple[str, int | None]] = []
    for block in _result_blocks(result.payload):
        for item in _items_of(block):
            if item.get("type") != "ai_overview":
                continue
            for reference in item.get("references") or []:
                if isinstance(reference, Mapping):
                    domain = str(reference.get("domain") or _domain_of(reference.get("url")))
                    if domain:
                        # No rank: a citation in an AI Overview has no ordinal position, and
                        # inventing one would be a fabricated measurement.
                        cited.append((domain, None))

    return _observe(
        query_text=query_text,
        kind=RetrievalKind.AI_OVERVIEW,
        result=result,
        ranked=cited,
        profile=profile,
        # An AI Overview that was not generated is missing data, not proven absence.
        looked_for_visibility=bool(cited),
    )


def parse_llm_visibility(
    result: RawApiResult, *, query_text: str, profile: Profile
) -> NormalizedObservation:
    cited: list[tuple[str, int | None]] = []
    for block in _result_blocks(result.payload):
        for item in _items_of(block):
            for section in item.get("sections") or []:
                if not isinstance(section, Mapping):
                    continue
                for annotation in section.get("annotations") or []:
                    if isinstance(annotation, Mapping):
                        domain = str(annotation.get("domain") or _domain_of(annotation.get("url")))
                        if domain:
                            cited.append((domain, None))

    return _observe(
        query_text=query_text,
        kind=RetrievalKind.LLM_VISIBILITY,
        result=result,
        ranked=cited,
        profile=profile,
        looked_for_visibility=bool(cited),
    )


def parse_keyword_metrics(
    result: RawApiResult, *, query_text: str, profile: Profile
) -> NormalizedObservation:
    volume: int | None = None
    competition: int | None = None

    for record in _result_blocks(result.payload):
        keyword = str(record.get("keyword") or "")
        if keyword and keyword.casefold() != query_text.casefold():
            continue
        raw_volume = record.get("search_volume")
        raw_competition = record.get("competition_index")
        if isinstance(raw_volume, int):
            volume = raw_volume
        if isinstance(raw_competition, int):
            competition = raw_competition
        break

    return _observe(
        query_text=query_text,
        kind=RetrievalKind.KEYWORD_METRICS,
        result=result,
        ranked=[],
        profile=profile,
        search_volume=volume,
        competition_index=competition,
        # This capability sizes a query; it says nothing about who ranks for it.
        looked_for_visibility=False,
    )


_PARSERS = {
    RetrievalKind.ORGANIC_SERP: parse_organic_serp,
    RetrievalKind.AI_OVERVIEW: parse_ai_overview,
    RetrievalKind.LLM_VISIBILITY: parse_llm_visibility,
    RetrievalKind.KEYWORD_METRICS: parse_keyword_metrics,
}

#: Maps an endpoint path back to the capability that produced it. Organic SERP and AI Overview
#: share a path, so the ordering of the tool call is what distinguishes them - resolved by
#: pairing each payload with the tool that was actually invoked.
_ENDPOINT_KIND_HINTS = {
    "/keywords_data/": RetrievalKind.KEYWORD_METRICS,
    "/ai_optimization/": RetrievalKind.LLM_VISIBILITY,
}


def _kind_for(result: RawApiResult, requested: Sequence[RetrievalKind]) -> RetrievalKind:
    """Determine which capability a payload came from."""
    for marker, kind in _ENDPOINT_KIND_HINTS.items():
        if marker in result.endpoint:
            return kind

    # Both SERP capabilities share an endpoint; the request parameters distinguish them.
    for task in result.payload.get("tasks") or []:
        if (
            isinstance(task, Mapping)
            and isinstance(task.get("data"), Mapping)
            and task["data"].get("load_async_ai_overview")
        ):
            return RetrievalKind.AI_OVERVIEW
    return (
        RetrievalKind.ORGANIC_SERP
        if RetrievalKind.ORGANIC_SERP in requested
        else next(iter(requested), RetrievalKind.ORGANIC_SERP)
    )


def normalize_outcome(outcome: RetrievalOutcome, *, profile: Profile) -> NormalizedQuery:
    """Turn one branch's raw payloads into a typed observation set."""
    observations: list[NormalizedObservation] = []

    for result in outcome.results:
        kind = _kind_for(result, outcome.sub_query.retrieval_kinds)
        parser = _PARSERS.get(kind)
        if parser is None:  # pragma: no cover - every kind has a parser
            continue
        try:
            observations.append(parser(result, query_text=outcome.query_text, profile=profile))
        except (TypeError, ValueError, AttributeError):
            # One malformed payload degrades one observation. Everything else this branch
            # retrieved is still usable, and the run continues.
            _logger.exception(
                "normalization.payload_unparseable",
                endpoint=result.endpoint,
                kind=kind.value,
                query=outcome.query_text,
            )

    return NormalizedQuery(query_text=outcome.query_text, observations=tuple(observations))


def build_discovered_query(
    normalized: NormalizedQuery, *, profile: Profile, run_uuid: UUID
) -> DiscoveredQuery:
    """Assemble the persisted entity, scoring it with the domain service.

    The score is computed by :mod:`sightline.domain.services.opportunity_scoring`, never by a
    model: it is a business rule, and a rule that lives in a prompt drifts silently between
    runs.
    """
    status, position = normalized.best_visibility()
    volume = SearchVolume(normalized.search_volume() or 0)
    difficulty = CompetitiveDifficulty(normalized.competition_index() or _UNKNOWN_DIFFICULTY)

    return DiscoveredQuery(
        profile_uuid=profile.uuid,
        run_uuid=run_uuid,
        query_text=normalized.query_text,
        estimated_search_volume=volume,
        competitive_difficulty=difficulty,
        opportunity_score=calculate_opportunity_score(
            search_volume=volume,
            competitive_difficulty=difficulty,
            visibility_status=status,
            visibility_position=position,
        ),
        visibility_status=status,
        visibility_position=position,
        evidence=normalized.to_evidence(),
    )


async def normalization_node(state: PipelineState, *, deps: NodeDependencies) -> PipelineState:
    profile = state["profile"]
    run_uuid = state["run_uuid"]
    outcomes = [outcome for outcome in state.get("retrieval_outcomes", []) if outcome.succeeded]

    async with node_span(
        NodeName.NORMALIZATION,
        metrics=current_metrics(),
        inputs={"successful_branches": len(outcomes)},
    ) as span:
        normalized = [normalize_outcome(outcome, profile=profile) for outcome in outcomes]
        usable = [entry for entry in normalized if not entry.is_empty]
        queries = [
            build_discovered_query(entry, profile=profile, run_uuid=run_uuid) for entry in usable
        ]

        span.set_output(
            normalized_queries=len(usable),
            observations=sum(len(entry.observations) for entry in usable),
            visible=sum(1 for query in queries if query.domain_visible),
        )

    return PipelineState(normalized=usable, queries=queries)

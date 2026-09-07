"""Deterministic DataForSEO response fixtures.

Spec S3.4 permits mocked responses behind a clearly-marked flag. These reproduce the real
API's envelope faithfully - the ``status_code`` / ``tasks[].status_code`` pair, the ``result``
nesting, the ``items`` array with ``rank_absolute`` - so the Normalization agent parses the
same structure it would parse against the live API, and switching ``DATAFORSEO_MODE=live``
exercises code already proven against this shape.

Every payload is **derived deterministically from the request**: the same keyword always
returns the same SERP. That makes runs reproducible and assertions stable - a mock returning
random data would make every downstream test flaky.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, Final

#: A realistic domain pool spanning the two verticals the spec's own examples use (SEO tooling
#: and project-management software), so a reviewer running either example question sees a
#: believable mix of visible and not-visible outcomes rather than a uniform blank.
_DOMAIN_POOL: Final[tuple[str, ...]] = (
    "surferseo.com",
    "clearscope.io",
    "marketmuse.com",
    "frase.io",
    "semrush.com",
    "ahrefs.com",
    "moz.com",
    "backlinko.com",
    "searchenginejournal.com",
    "hubspot.com",
    "asana.com",
    "monday.com",
    "clickup.com",
    "notion.so",
    "trello.com",
    "wrike.com",
    "g2.com",
    "capterra.com",
    "zapier.com",
    "reddit.com",
)

_COMPETITION_BANDS: Final[tuple[tuple[str, int, int], ...]] = (
    ("LOW", 0, 33),
    ("MEDIUM", 34, 66),
    ("HIGH", 67, 100),
)

_MOCK_VERSION: Final = "0.1.20260101"


def _rng_for(*parts: str) -> random.Random:
    """A generator seeded from the request, so identical requests yield identical responses."""
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))  # noqa: S311 - fixture shaping, not security


def _now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S +00:00")


def _envelope(*, path: Sequence[str], data: Mapping[str, Any], result: list[Any]) -> dict[str, Any]:
    """Wrap a result in DataForSEO's two-level envelope, including both status codes."""
    task_id = hashlib.sha256("/".join(path).encode()).hexdigest()[:16]
    return {
        "version": _MOCK_VERSION,
        "status_code": 20000,
        "status_message": "Ok.",
        "time": "0.1073 sec.",
        "cost": 0.002,
        "tasks_count": 1,
        "tasks_error": 0,
        "tasks": [
            {
                "id": f"mock-{task_id}",
                "status_code": 20000,
                "status_message": "Ok.",
                "time": "0.0921 sec.",
                "cost": 0.002,
                "result_count": len(result),
                "path": list(path),
                "data": dict(data),
                "result": result,
            }
        ],
    }


def _ranked_domains(keyword: str, count: int) -> list[str]:
    rng = _rng_for("serp", keyword)
    pool = list(_DOMAIN_POOL)
    rng.shuffle(pool)
    return pool[: min(count, len(pool))]


def organic_serp(
    *, keyword: str, location_code: int, language_code: str, depth: int
) -> dict[str, Any]:
    """A live organic SERP with ranked ``organic`` items."""
    domains = _ranked_domains(keyword, depth)
    rng = _rng_for("serp_meta", keyword)

    items = [
        {
            "type": "organic",
            "rank_group": index,
            "rank_absolute": index,
            "position": "left",
            "domain": domain,
            "title": f"{keyword.title()} - {domain.split('.')[0].title()}",
            "url": f"https://{domain}/{keyword.replace(' ', '-')}",
            "breadcrumb": f"https://{domain} > guides",
            "description": (
                f"An in-depth guide to {keyword}, covering evaluation criteria, pricing "
                f"and how {domain.split('.')[0].title()} compares to alternatives."
            ),
        }
        for index, domain in enumerate(domains, start=1)
    ]

    return _envelope(
        path=["v3", "serp", "google", "organic", "live", "advanced"],
        data={
            "api": "serp",
            "function": "live",
            "se": "google",
            "se_type": "organic",
            "keyword": keyword,
            "location_code": location_code,
            "language_code": language_code,
            "device": "desktop",
            "os": "windows",
        },
        result=[
            {
                "keyword": keyword,
                "type": "organic",
                "se_domain": "google.com",
                "location_code": location_code,
                "language_code": language_code,
                "check_url": f"https://www.google.com/search?q={keyword.replace(' ', '+')}",
                "datetime": _now_iso(),
                "spell": None,
                "item_types": ["organic"],
                "se_results_count": rng.randint(1_200_000, 480_000_000),
                "items_count": len(items),
                "items": items,
            }
        ],
    )


def ai_overview(*, keyword: str, location_code: int, language_code: str) -> dict[str, Any]:
    """A SERP containing an ``ai_overview`` block with cited references."""
    cited = _ranked_domains(keyword, 5)[:4]
    rng = _rng_for("aio", keyword)
    present = rng.random() > 0.15  # a minority of keywords legitimately trigger no overview

    overview_item = {
        "type": "ai_overview",
        "rank_group": 1,
        "rank_absolute": 1,
        "position": "left",
        "asynchronous_ai_overview": False,
        "items": [
            {
                "type": "ai_overview_element",
                "title": None,
                "text": (
                    f"When evaluating {keyword}, the most frequently recommended options "
                    f"are {', '.join(d.split('.')[0].title() for d in cited[:3])}. "
                    "Selection usually turns on integration depth, pricing model and "
                    "reporting depth."
                ),
                "links": [
                    {"type": "link_element", "domain": d, "url": f"https://{d}"} for d in cited
                ],
            }
        ],
        "references": [
            {
                "type": "ai_overview_reference",
                "source": domain.split(".")[0].title(),
                "domain": domain,
                "url": f"https://{domain}/{keyword.replace(' ', '-')}",
                "title": f"{keyword.title()} - {domain.split('.')[0].title()}",
                "text": f"{domain.split('.')[0].title()} is frequently cited for {keyword}.",
            }
            for domain in cited
        ],
    }

    return _envelope(
        path=["v3", "serp", "google", "organic", "live", "advanced"],
        data={
            "api": "serp",
            "function": "live",
            "se": "google",
            "se_type": "organic",
            "keyword": keyword,
            "location_code": location_code,
            "language_code": language_code,
            "load_async_ai_overview": True,
        },
        result=[
            {
                "keyword": keyword,
                "type": "organic",
                "se_domain": "google.com",
                "location_code": location_code,
                "language_code": language_code,
                "check_url": f"https://www.google.com/search?q={keyword.replace(' ', '+')}",
                "datetime": _now_iso(),
                "item_types": ["ai_overview"] if present else ["organic"],
                "items_count": 1 if present else 0,
                "items": [overview_item] if present else [],
            }
        ],
    )


def llm_visibility(*, prompt: str, llm_name: str) -> dict[str, Any]:
    """An LLM answer with URL citations, as the AI Optimization API returns it."""
    cited = _ranked_domains(prompt, 4)[:3]
    rng = _rng_for("llm", prompt, llm_name)

    return _envelope(
        path=["v3", "ai_optimization", llm_name, "llm_responses", "live"],
        data={
            "api": "ai_optimization",
            "function": "llm_responses",
            "llm": llm_name,
            "user_prompt": prompt,
            "web_search": True,
        },
        result=[
            {
                "prompt": prompt,
                "model_name": f"{llm_name}-latest",
                "input_tokens": rng.randint(40, 180),
                "output_tokens": rng.randint(120, 600),
                "money_spent": round(rng.uniform(0.0004, 0.004), 6),
                "items": [
                    {
                        "type": "llm_response",
                        "sections": [
                            {
                                "type": "text",
                                "text": (
                                    f'For "{prompt}", the tools most often recommended are '
                                    f"{', '.join(d.split('.')[0].title() for d in cited)}. "
                                    "Each differs mainly in workflow depth and pricing."
                                ),
                                "annotations": [
                                    {
                                        "type": "url_citation",
                                        "url": f"https://{domain}/",
                                        "title": f"{domain.split('.')[0].title()}",
                                        "domain": domain,
                                    }
                                    for domain in cited
                                ],
                            }
                        ],
                    }
                ],
            }
        ],
    )


def keyword_metrics(
    *, keywords: Sequence[str], location_code: int, language_code: str
) -> dict[str, Any]:
    """Search volume and competition index per keyword, as Google Ads data returns it."""
    result: list[dict[str, Any]] = []

    for keyword in keywords:
        rng = _rng_for("metrics", keyword)
        # Log-uniform volume: real keyword distributions are heavy-tailed, and a uniform draw
        # would make the log-scaled demand component of the opportunity score meaningless.
        volume = int(10 ** rng.uniform(1.7, 4.9))
        index = rng.randint(5, 95)
        band = next(name for name, low, high in _COMPETITION_BANDS if low <= index <= high)
        cpc = round(rng.uniform(0.6, 18.0), 2)

        result.append(
            {
                "keyword": keyword,
                "location_code": location_code,
                "language_code": language_code,
                "search_partners": False,
                "competition": band,
                "competition_index": index,
                "search_volume": volume,
                "low_top_of_page_bid": round(cpc * 0.55, 2),
                "high_top_of_page_bid": round(cpc * 1.8, 2),
                "cpc": cpc,
                "monthly_searches": [
                    {
                        "year": 2026,
                        "month": month,
                        "search_volume": int(volume * rng.uniform(0.72, 1.28)),
                    }
                    for month in range(1, 13)
                ],
            }
        )

    return _envelope(
        path=["v3", "keywords_data", "google_ads", "search_volume", "live"],
        data={
            "api": "keywords_data",
            "function": "search_volume",
            "se": "google_ads",
            "keywords": list(keywords),
            "location_code": location_code,
            "language_code": language_code,
        },
        result=result,
    )

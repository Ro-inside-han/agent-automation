"""Search agent: finds candidate source URLs for an entity.

Uses Tavily when TAVILY_API_KEY is configured. Without a key, the pipeline
still works off `seed_urls` supplied directly in the input CSV (a
pipe-separated `sources` column) -- useful if you already have a SalesQL /
manual research URL list and don't want to pay for a search API.
"""
from __future__ import annotations

import logging
import os

from src.models import EntityInput

logger = logging.getLogger(__name__)

DEFAULT_QUERIES = [
    '"{company}" company profile revenue',
    '"{company}" CSR activities news recent developments',
]


def build_queries(entity: EntityInput) -> list[str]:
    return [
        template.format(company=entity.company, leader=entity.leader_name)
        for template in DEFAULT_QUERIES
    ]


def run_query(query: str, max_results: int = 4, entity_id: str = "") -> list[str]:
    """Runs one Tavily search query and returns candidate URLs. Returns []
    (never raises) if no API key is configured or the request fails, so
    callers can treat "no results" and "search unavailable" the same way."""

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return []

    from tavily import TavilyClient  # imported lazily: optional dependency

    client = TavilyClient(api_key=api_key)
    try:
        result = client.search(query=query, max_results=max_results)
    except Exception as exc:
        logger.warning(
            "Tavily search failed for entity=%s query=%r: %s", entity_id, query, exc
        )
        return []

    return [hit["url"] for hit in result.get("results", []) if hit.get("url")]


def search_entity(entity: EntityInput, max_results_per_query: int = 4) -> list[str]:
    """Returns candidate URLs for an entity. Falls back to seed_urls only
    when no search provider is configured."""

    if not os.getenv("TAVILY_API_KEY"):
        return list(entity.seed_urls)

    urls: list[str] = list(entity.seed_urls)
    seen = set(urls)

    for query in build_queries(entity):
        for url in run_query(query, max_results_per_query, entity.entity_id):
            if url not in seen:
                seen.add(url)
                urls.append(url)

    return urls


def search_for_field(entity: EntityInput, field_name: str, source_hint: str) -> list[str]:
    """Targeted retry search for one specific low-confidence field, e.g.
    biasing toward structured financial-data sites for a 'revenue' field
    instead of the generic company-profile query."""

    query = f'"{entity.company}" {field_name.replace("_", " ")} {source_hint}'.strip()
    return run_query(query, max_results=4, entity_id=entity.entity_id)

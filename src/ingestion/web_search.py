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


def search_entity(entity: EntityInput, max_results_per_query: int = 4) -> list[str]:
    """Returns candidate URLs for an entity. Falls back to seed_urls only
    when no search provider is configured."""

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return list(entity.seed_urls)

    from tavily import TavilyClient  # imported lazily: optional dependency

    client = TavilyClient(api_key=api_key)
    urls: list[str] = list(entity.seed_urls)
    seen = set(urls)

    for query in build_queries(entity):
        try:
            result = client.search(query=query, max_results=max_results_per_query)
        except Exception as exc:
            logger.warning(
                "Tavily search failed for entity=%s query=%r: %s",
                entity.entity_id,
                query,
                exc,
            )
            continue
        for hit in result.get("results", []):
            url = hit.get("url")
            if url and url not in seen:
                seen.add(url)
                urls.append(url)

    return urls

"""Fetches a URL and extracts clean article/body text from it."""
from __future__ import annotations

import trafilatura

from src.ingestion.store import now_iso
from src.models import RawDocument


def fetch_document(entity_id: str, url: str, source_type: str = "web_search") -> RawDocument | None:
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        return None

    text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
    if not text or len(text.strip()) < 100:
        return None

    metadata = trafilatura.extract_metadata(downloaded)
    title = metadata.title if metadata and metadata.title else url

    return RawDocument(
        entity_id=entity_id,
        url=url,
        title=title,
        text=text.strip(),
        fetched_at=now_iso(),
        source_type=source_type,
    )

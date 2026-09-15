"""Orchestrator agent: drives the per-entity ingest -> index -> extract loop.

This is the "agentic" control flow: for each entity, each field, if the
first retrieval + extraction pass comes back low-confidence or null, it
autonomously triggers one extra targeted web search + scrape + re-index +
re-extract (see `_retry_field_with_targeted_search`) before giving up and
flagging the field for human review. The pipeline decides for itself, per
field, whether it has enough evidence -- it isn't a fixed linear script.
"""
from __future__ import annotations

import logging
from pathlib import Path

from src.agents.extraction_agent import extract_field
from src.config import FieldSpec, SchemaConfig
from src.ingestion.scraper import fetch_document
from src.ingestion.store import has_documents, load_documents, save_document
from src.ingestion.web_search import search_entity, search_for_field
from src.models import EntityInput, EntityProfile, FieldExtraction
from src.rag.chunker import chunk_documents
from src.rag.vectorstore import index_chunks, retrieve

logger = logging.getLogger(__name__)


def ingest_entity(entity: EntityInput, raw_dir: Path, force: bool = False) -> int:
    """Fetches and caches raw documents for one entity. Returns doc count."""
    if not force and has_documents(entity.entity_id, raw_dir):
        return len(load_documents(entity.entity_id, raw_dir))

    urls = search_entity(entity)
    if not urls:
        logger.warning(
            "%s: search returned no candidate URLs (no TAVILY_API_KEY / no "
            "search results / no seed_urls)",
            entity.entity_id,
        )
        return 0

    saved = 0
    for url in urls:
        doc = fetch_document(entity.entity_id, url)
        if doc:
            save_document(doc, raw_dir)
            saved += 1
        else:
            logger.warning("%s: failed to fetch/parse %s", entity.entity_id, url)

    if saved == 0:
        logger.warning(
            "%s: found %d candidate URL(s) but none could be scraped",
            entity.entity_id,
            len(urls),
        )
    return saved


def index_entity(entity_id: str, raw_dir: Path, index_dir: Path) -> int:
    docs = load_documents(entity_id, raw_dir)
    chunks = chunk_documents(docs)
    index_chunks(chunks, index_dir)
    return len(chunks)


def _retry_field_with_targeted_search(
    entity: EntityInput,
    field: FieldSpec,
    entity_label: str,
    raw_dir: Path,
    index_dir: Path,
    top_k: int,
) -> FieldExtraction | None:
    """Agentic self-correction: when the first pass comes back low-confidence
    or null, fire one extra targeted search (biased toward this specific
    field, e.g. financial-data sites for "revenue"), scrape whatever's new,
    re-index, and retry the extraction once. Returns None if the retry found
    nothing new, so the caller keeps the original (low-confidence) result."""

    urls = search_for_field(entity, field.name, field.source_hint)
    if not urls:
        return None

    new_docs = 0
    for url in urls:
        doc = fetch_document(entity.entity_id, url, source_type="retry_search")
        if doc:
            save_document(doc, raw_dir)
            new_docs += 1

    if new_docs == 0:
        logger.info(
            "%s: retry search for field=%s found no new scrapeable sources",
            entity.entity_id,
            field.name,
        )
        return None

    index_entity(entity.entity_id, raw_dir, index_dir)
    query = f"{field.name}: {field.source_hint or field.name}"
    chunks = retrieve(entity.entity_id, query, top_k=top_k, index_dir=index_dir)
    logger.info(
        "%s: retried field=%s with %d new source(s)",
        entity.entity_id,
        field.name,
        new_docs,
    )
    return extract_field(field, entity_label, chunks)


def extract_entity_profile(
    entity: EntityInput,
    schema: SchemaConfig,
    raw_dir: Path,
    index_dir: Path,
    top_k: int = 5,
) -> EntityProfile:
    entity_label = f"{entity.leader_name} / {entity.company}".strip(" /")
    profile = EntityProfile(entity_id=entity.entity_id)

    for field in schema.fields:
        query = f"{field.name}: {field.source_hint or field.name}"
        chunks = retrieve(entity.entity_id, query, top_k=top_k, index_dir=index_dir)
        result: FieldExtraction = extract_field(field, entity_label, chunks)

        if result.value is None or result.confidence < schema.confidence_threshold:
            retried = _retry_field_with_targeted_search(
                entity, field, entity_label, raw_dir, index_dir, top_k
            )
            if retried is not None:
                result = retried

        result.needs_review = (
            result.needs_review or result.confidence < schema.confidence_threshold
        )
        profile.fields[field.name] = result

    return profile


def run_pipeline_for_entity(
    entity: EntityInput,
    schema: SchemaConfig,
    raw_dir: Path,
    index_dir: Path,
) -> EntityProfile:
    ingest_entity(entity, raw_dir)
    index_entity(entity.entity_id, raw_dir, index_dir)
    return extract_entity_profile(entity, schema, raw_dir, index_dir)

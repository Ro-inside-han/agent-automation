"""Orchestrator agent: drives the per-entity ingest -> index -> extract loop.

This is the "agentic" control flow: for each entity, each field, if the
first retrieval + extraction pass comes back low-confidence, it triggers one
extra targeted web search + re-ingest + re-extract before giving up and
flagging the field for human review.
"""
from __future__ import annotations

import logging
from pathlib import Path

from src.agents.extraction_agent import extract_field
from src.config import SchemaConfig
from src.ingestion.scraper import fetch_document
from src.ingestion.store import has_documents, load_documents, save_document
from src.ingestion.web_search import search_entity
from src.models import EntityInput, EntityProfile, FieldExtraction
from src.rag.chunker import chunk_documents
from src.rag.vectorstore import index_chunks, retrieve

logger = logging.getLogger(__name__)


def ingest_entity(entity: EntityInput, raw_dir: Path, force: bool = False) -> int:
    """Fetches and caches raw documents for one entity. Returns doc count."""
    if not force and has_documents(entity.entity_id, raw_dir):
        return len(load_documents(entity.entity_id, raw_dir))

    urls = search_entity(entity)
    saved = 0
    for url in urls:
        doc = fetch_document(entity.entity_id, url)
        if doc:
            save_document(doc, raw_dir)
            saved += 1
    return saved


def index_entity(entity_id: str, raw_dir: Path, index_dir: Path) -> int:
    docs = load_documents(entity_id, raw_dir)
    chunks = chunk_documents(docs)
    index_chunks(chunks, index_dir)
    return len(chunks)


def extract_entity_profile(
    entity: EntityInput, schema: SchemaConfig, index_dir: Path, top_k: int = 5
) -> EntityProfile:
    entity_label = f"{entity.leader_name} / {entity.company}".strip(" /")
    profile = EntityProfile(entity_id=entity.entity_id)

    for field in schema.fields:
        query = f"{field.name}: {field.source_hint or field.name}"
        chunks = retrieve(entity.entity_id, query, top_k=top_k, index_dir=index_dir)

        result: FieldExtraction = extract_field(field, entity_label, chunks)
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
    return extract_entity_profile(entity, schema, index_dir)

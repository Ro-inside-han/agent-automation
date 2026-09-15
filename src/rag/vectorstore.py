"""Persistent vector store (Chroma) holding all entities' chunks.

A single collection is used with `entity_id` stored as metadata so retrieval
for one entity is a metadata-filtered similarity search. This keeps the
index reusable: adding a new schema field later just means running new
queries against the same index, not re-embedding.
"""
from __future__ import annotations

from pathlib import Path

from src.rag.chunker import Chunk
from src.rag.embeddings import embed_query, embed_texts

DEFAULT_INDEX_DIR = Path("data/index")
COLLECTION_NAME = "profiling_chunks"


def _get_collection(index_dir: Path = DEFAULT_INDEX_DIR):
    import chromadb

    index_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(index_dir))
    return client.get_or_create_collection(COLLECTION_NAME)


def index_chunks(chunks: list[Chunk], index_dir: Path = DEFAULT_INDEX_DIR) -> None:
    if not chunks:
        return

    collection = _get_collection(index_dir)
    embeddings = embed_texts([c.text for c in chunks])

    collection.upsert(
        ids=[c.chunk_id for c in chunks],
        embeddings=embeddings,
        documents=[c.text for c in chunks],
        metadatas=[
            {
                "entity_id": c.entity_id,
                "source_url": c.source_url,
                "source_title": c.source_title,
            }
            for c in chunks
        ],
    )


def retrieve(
    entity_id: str, query: str, top_k: int = 5, index_dir: Path = DEFAULT_INDEX_DIR
) -> list[dict]:
    collection = _get_collection(index_dir)
    if collection.count() == 0:
        return []

    result = collection.query(
        query_embeddings=[embed_query(query)],
        n_results=top_k,
        where={"entity_id": entity_id},
    )

    hits = []
    docs = result.get("documents") or [[]]
    metas = result.get("metadatas") or [[]]
    if docs and docs[0]:
        for text, meta in zip(docs[0], metas[0]):
            hits.append({"text": text, **meta})
    return hits

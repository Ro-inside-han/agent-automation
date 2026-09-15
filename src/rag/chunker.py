"""Splits raw documents into overlapping chunks for embedding/retrieval."""
from __future__ import annotations

from dataclasses import dataclass

from src.models import RawDocument


@dataclass
class Chunk:
    entity_id: str
    chunk_id: str
    text: str
    source_url: str
    source_title: str


def chunk_document(
    doc: RawDocument, chunk_size: int = 800, overlap: int = 150
) -> list[Chunk]:
    words = doc.text.split()
    if not words:
        return []

    chunks: list[Chunk] = []
    start = 0
    idx = 0
    step = max(chunk_size - overlap, 1)

    while start < len(words):
        piece = " ".join(words[start : start + chunk_size])
        chunk_id = f"{doc.entity_id}__{hash(doc.url) & 0xFFFFFFFF}__{idx}"
        chunks.append(
            Chunk(
                entity_id=doc.entity_id,
                chunk_id=chunk_id,
                text=piece,
                source_url=doc.url,
                source_title=doc.title,
            )
        )
        idx += 1
        start += step

    return chunks


def chunk_documents(docs: list[RawDocument], **kwargs) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in docs:
        chunks.extend(chunk_document(doc, **kwargs))
    return chunks

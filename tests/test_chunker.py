from src.models import RawDocument
from src.rag.chunker import chunk_document


def _doc(word_count: int) -> RawDocument:
    text = " ".join(f"word{i}" for i in range(word_count))
    return RawDocument(
        entity_id="acme",
        url="https://example.com/a",
        title="A",
        text=text,
        fetched_at="2026-01-01T00:00:00Z",
    )


def test_short_document_is_single_chunk():
    chunks = chunk_document(_doc(50), chunk_size=800, overlap=150)
    assert len(chunks) == 1
    assert chunks[0].entity_id == "acme"


def test_long_document_is_split_with_overlap():
    chunks = chunk_document(_doc(2000), chunk_size=800, overlap=150)
    assert len(chunks) > 1
    # consecutive chunks should share some overlapping words
    first_words = set(chunks[0].text.split())
    second_words = set(chunks[1].text.split())
    assert first_words & second_words


def test_empty_document_yields_no_chunks():
    chunks = chunk_document(_doc(0), chunk_size=800, overlap=150)
    assert chunks == []

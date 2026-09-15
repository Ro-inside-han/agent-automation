from src.ingestion.store import load_documents, save_document
from src.models import RawDocument


def test_save_and_load_roundtrips_non_ascii_text(tmp_path):
    doc = RawDocument(
        entity_id="acme",
        url="https://example.com/a",
        title="Acme reports revenue of ₹1,200 crore",
        text="Revenue was ₹1,200 crore this fiscal year.",
        fetched_at="2026-01-01T00:00:00Z",
    )

    save_document(doc, raw_dir=tmp_path)
    loaded = load_documents("acme", raw_dir=tmp_path)

    assert len(loaded) == 1
    assert loaded[0].text == doc.text
    assert "₹" in loaded[0].title

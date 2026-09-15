from src.agents import orchestrator
from src.config import FieldSpec, SchemaConfig
from src.models import EntityInput, FieldExtraction, RawDocument


def _schema(threshold=0.6) -> SchemaConfig:
    return SchemaConfig(
        project_name="Test",
        confidence_threshold=threshold,
        fields=[FieldSpec(name="revenue", type="number", unit="INR crore")],
    )


def _entity() -> EntityInput:
    return EntityInput(entity_id="acme", company="Acme Ltd", leader_name="Jane Doe")


def test_low_confidence_field_triggers_retry_and_uses_better_result(
    monkeypatch, tmp_path
):
    raw_dir = tmp_path / "raw"
    index_dir = tmp_path / "index"

    monkeypatch.setattr(orchestrator, "retrieve", lambda *a, **k: [])

    calls = {"n": 0}

    def fake_extract_field(field, entity_label, chunks):
        calls["n"] += 1
        if calls["n"] == 1:
            return FieldExtraction(value=None, confidence=0.1, citation="", needs_review=True)
        return FieldExtraction(value=500, confidence=0.9, citation="https://good-source", needs_review=False)

    monkeypatch.setattr(orchestrator, "extract_field", fake_extract_field)
    monkeypatch.setattr(
        orchestrator, "search_for_field", lambda entity, name, hint: ["https://good-source"]
    )
    monkeypatch.setattr(
        orchestrator,
        "fetch_document",
        lambda entity_id, url, source_type="web_search": RawDocument(
            entity_id=entity_id,
            url=url,
            title="t",
            text="Revenue was 500 crore.",
            fetched_at="2026-01-01T00:00:00Z",
            source_type=source_type,
        ),
    )
    monkeypatch.setattr(orchestrator, "index_entity", lambda *a, **k: 1)

    profile = orchestrator.extract_entity_profile(_entity(), _schema(), raw_dir, index_dir)

    assert profile.fields["revenue"].value == 500
    assert profile.fields["revenue"].needs_review is False
    assert calls["n"] == 2  # first pass + one retry


def test_retry_finds_nothing_keeps_original_low_confidence_result(monkeypatch, tmp_path):
    raw_dir = tmp_path / "raw"
    index_dir = tmp_path / "index"

    monkeypatch.setattr(orchestrator, "retrieve", lambda *a, **k: [])
    monkeypatch.setattr(
        orchestrator,
        "extract_field",
        lambda field, label, chunks: FieldExtraction(
            value=None, confidence=0.1, citation="", needs_review=True
        ),
    )
    monkeypatch.setattr(orchestrator, "search_for_field", lambda entity, name, hint: [])

    profile = orchestrator.extract_entity_profile(_entity(), _schema(), raw_dir, index_dir)

    assert profile.fields["revenue"].value is None
    assert profile.fields["revenue"].needs_review is True


def test_high_confidence_first_pass_skips_retry(monkeypatch, tmp_path):
    raw_dir = tmp_path / "raw"
    index_dir = tmp_path / "index"

    monkeypatch.setattr(orchestrator, "retrieve", lambda *a, **k: [])
    calls = {"n": 0}

    def fake_extract_field(field, entity_label, chunks):
        calls["n"] += 1
        return FieldExtraction(value=500, confidence=0.95, citation="https://x", needs_review=False)

    monkeypatch.setattr(orchestrator, "extract_field", fake_extract_field)

    def fail_if_called(*a, **k):
        raise AssertionError("search_for_field should not be called when confidence is high")

    monkeypatch.setattr(orchestrator, "search_for_field", fail_if_called)

    profile = orchestrator.extract_entity_profile(_entity(), _schema(), raw_dir, index_dir)

    assert calls["n"] == 1
    assert profile.fields["revenue"].value == 500

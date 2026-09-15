import pickle

from src import cli
from src.models import EntityProfile, FieldExtraction


def test_extract_resumes_and_skips_already_done_entities(tmp_path, monkeypatch):
    cache_path = tmp_path / "profiles.pkl"
    monkeypatch.setattr(cli, "PROFILES_CACHE", cache_path)

    existing = EntityProfile(
        entity_id="acme",
        fields={"revenue": FieldExtraction(value=100, confidence=0.9, citation="x")},
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump([existing], f)

    calls = []

    class FakeEntity:
        def __init__(self, entity_id):
            self.entity_id = entity_id

    monkeypatch.setattr(
        cli, "load_entities", lambda path: [FakeEntity("acme"), FakeEntity("other")]
    )
    monkeypatch.setattr(cli, "load_schema", lambda path: object())

    def fake_extract(entity, schema, raw_dir, index_dir, top_k):
        calls.append(entity.entity_id)
        return EntityProfile(entity_id=entity.entity_id, fields={})

    monkeypatch.setattr(cli, "extract_entity_profile", fake_extract)

    args = type("Args", (), {"input": "x", "schema": "y", "top_k": 5, "force": False})()
    cli.cmd_extract(args)

    # "acme" was already cached -> skipped; only "other" gets extracted.
    assert calls == ["other"]

    with open(cache_path, "rb") as f:
        saved = {p.entity_id for p in pickle.load(f)}
    assert saved == {"acme", "other"}


def test_extract_one_entity_failure_does_not_lose_others(tmp_path, monkeypatch):
    cache_path = tmp_path / "profiles.pkl"
    monkeypatch.setattr(cli, "PROFILES_CACHE", cache_path)

    class FakeEntity:
        def __init__(self, entity_id):
            self.entity_id = entity_id

    monkeypatch.setattr(
        cli,
        "load_entities",
        lambda path: [FakeEntity("good1"), FakeEntity("bad"), FakeEntity("good2")],
    )
    monkeypatch.setattr(cli, "load_schema", lambda path: object())

    def fake_extract(entity, schema, raw_dir, index_dir, top_k):
        if entity.entity_id == "bad":
            raise RuntimeError("simulated crash")
        return EntityProfile(entity_id=entity.entity_id, fields={})

    monkeypatch.setattr(cli, "extract_entity_profile", fake_extract)

    args = type("Args", (), {"input": "x", "schema": "y", "top_k": 5, "force": False})()
    cli.cmd_extract(args)

    with open(cache_path, "rb") as f:
        saved = {p.entity_id for p in pickle.load(f)}
    assert saved == {"good1", "good2"}

from src.agents import extraction_agent
from src.config import FieldSpec


def _field() -> FieldSpec:
    return FieldSpec(name="revenue", type="number", unit="INR crore")


def test_extract_field_uses_configured_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setitem(
        extraction_agent._PROVIDERS,
        "groq",
        lambda prompt: {"value": 500, "confidence": 0.8, "citation": "https://x"},
    )

    result = extraction_agent.extract_field(_field(), "Acme / Acme Ltd", [])

    assert result.value == 500
    assert result.confidence == 0.8
    assert result.needs_review is False


def test_extract_field_flags_null_value_for_review(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setitem(
        extraction_agent._PROVIDERS,
        "groq",
        lambda prompt: {"value": None, "confidence": 0.1, "citation": ""},
    )

    result = extraction_agent.extract_field(_field(), "Acme / Acme Ltd", [])

    assert result.value is None
    assert result.needs_review is True


def test_extract_field_rejects_unknown_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "made-up-provider")

    try:
        extraction_agent.extract_field(_field(), "Acme / Acme Ltd", [])
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "made-up-provider" in str(exc)

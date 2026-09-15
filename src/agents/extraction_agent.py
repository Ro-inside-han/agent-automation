"""Extraction agent: turns retrieved chunks into one structured field value.

Uses Claude tool-calling so the model is forced to return a value +
confidence + citation instead of free text, which is what makes the output
auditable (a jury/client can trace every cell back to a source).
"""
from __future__ import annotations

import os

from src.config import FieldSpec
from src.models import FieldExtraction

_TOOL = {
    "name": "record_field",
    "description": "Record the extracted value for one profiling field.",
    "input_schema": {
        "type": "object",
        "properties": {
            "value": {
                "description": "The extracted value. Use null if it cannot "
                "be determined from the provided context.",
            },
            "confidence": {
                "type": "number",
                "description": "0.0-1.0 confidence that the value is correct "
                "and well-supported by the given context.",
            },
            "citation": {
                "type": "string",
                "description": "Which source URL(s)/titles from the context "
                "support this value. Empty string if value is null.",
            },
        },
        "required": ["value", "confidence", "citation"],
    },
}


def _build_prompt(field: FieldSpec, entity_label: str, context_chunks: list[dict]) -> str:
    if not context_chunks:
        context_block = "(no retrieved context available)"
    else:
        context_block = "\n\n".join(
            f"[Source: {c.get('source_title', c.get('source_url', 'unknown'))} "
            f"| {c.get('source_url', '')}]\n{c['text']}"
            for c in context_chunks
        )

    constraints = []
    if field.type == "number" and field.unit:
        constraints.append(f"Report the value as a plain number in {field.unit}.")
    if field.type == "text_summary" and field.max_words:
        constraints.append(f"Summarize in at most {field.max_words} words.")

    return f"""You are extracting one field for a profile of: {entity_label}

Field to extract: {field.name}
What this field means: {field.source_hint or field.name}
{' '.join(constraints)}

Context (retrieved from cached research documents):
{context_block}

Using only the context above, call record_field with the best supported
value. If the context does not support a confident answer, set value to
null and confidence to a low number rather than guessing."""


def extract_field(
    field: FieldSpec, entity_label: str, context_chunks: list[dict]
) -> FieldExtraction:
    from anthropic import Anthropic

    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5")

    prompt = _build_prompt(field, entity_label, context_chunks)

    response = client.messages.create(
        model=model,
        max_tokens=500,
        tools=[_TOOL],
        tool_choice={"type": "tool", "name": "record_field"},
        messages=[{"role": "user", "content": prompt}],
    )

    tool_use = next(b for b in response.content if b.type == "tool_use")
    result = tool_use.input

    value = result.get("value")
    confidence = float(result.get("confidence", 0.0))
    citation = result.get("citation", "")

    return FieldExtraction(
        value=value,
        confidence=confidence,
        citation=citation,
        needs_review=value is None,
    )

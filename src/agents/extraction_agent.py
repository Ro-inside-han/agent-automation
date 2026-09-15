"""Extraction agent: turns retrieved chunks into one structured field value.

Uses tool/function-calling so the model is forced to return a value +
confidence + citation instead of free text, which is what makes the output
auditable (a jury/client can trace every cell back to a source). Supports
two interchangeable providers, picked via the LLM_PROVIDER env var:

- "anthropic" (default): Claude, paid API credits required.
- "groq": free-tier API (console.groq.com) running open models (e.g.
  Llama 3.3), good for a no-budget pilot before paying for Anthropic.
"""
from __future__ import annotations

import json
import os

from src.config import FieldSpec
from src.models import FieldExtraction

_TOOL_NAME = "record_field"
_TOOL_DESCRIPTION = "Record the extracted value for one profiling field."
_TOOL_PARAMETERS = {
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
}


# Caps how much retrieved text goes into one prompt. PDF-extracted text
# (annual reports, CSR policy PDFs) tokenizes far less efficiently than
# normal prose -- a handful of "800-word" chunks can blow past free-tier
# token-per-minute limits even though the word count looks small.
_MAX_CHARS_PER_CHUNK = 1200


def _build_prompt(field: FieldSpec, entity_label: str, context_chunks: list[dict]) -> str:
    if not context_chunks:
        context_block = "(no retrieved context available)"
    else:
        parts = []
        for c in context_chunks:
            text = c["text"]
            if len(text) > _MAX_CHARS_PER_CHUNK:
                text = text[:_MAX_CHARS_PER_CHUNK] + " ...[truncated]"
            source = c.get("source_title", c.get("source_url", "unknown"))
            parts.append(f"[Source: {source} | {c.get('source_url', '')}]\n{text}")
        context_block = "\n\n".join(parts)

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


def _call_anthropic(prompt: str) -> dict:
    from anthropic import Anthropic

    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5")

    response = client.messages.create(
        model=model,
        max_tokens=500,
        tools=[
            {
                "name": _TOOL_NAME,
                "description": _TOOL_DESCRIPTION,
                "input_schema": _TOOL_PARAMETERS,
            }
        ],
        tool_choice={"type": "tool", "name": _TOOL_NAME},
        messages=[{"role": "user", "content": prompt}],
    )

    tool_use = next(b for b in response.content if b.type == "tool_use")
    return tool_use.input


def _call_groq(prompt: str) -> dict:
    # Groq's model lineup changes over time (models get deprecated/renamed).
    # If GROQ_MODEL 404s, list what your key currently has access to:
    #   curl -H "Authorization: Bearer $GROQ_API_KEY" \
    #     https://api.groq.com/openai/v1/models
    # and pick one with "tools" in supported_features.
    from groq import Groq

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

    kwargs = {}
    if model.startswith("openai/gpt-oss"):
        # These are reasoning models: they spend output tokens on internal
        # chain-of-thought before emitting the tool call. A tight
        # max_completion_tokens can exhaust the budget on reasoning alone,
        # leaving no tool call at all (400 tool_use_failed, empty
        # failed_generation). reasoning_effort="low" curbs that spend, and
        # the higher cap leaves room for reasoning + the actual call.
        # Other Groq models don't accept reasoning_effort, so this is
        # scoped to gpt-oss only.
        kwargs["max_completion_tokens"] = 1500
        kwargs["reasoning_effort"] = "low"
    else:
        kwargs["max_completion_tokens"] = 500

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": _TOOL_NAME,
                    "description": _TOOL_DESCRIPTION,
                    "parameters": _TOOL_PARAMETERS,
                },
            }
        ],
        tool_choice={"type": "function", "function": {"name": _TOOL_NAME}},
        **kwargs,
    )

    tool_call = response.choices[0].message.tool_calls[0]
    return json.loads(tool_call.function.arguments)


_PROVIDERS = {
    "anthropic": _call_anthropic,
    "groq": _call_groq,
}


def extract_field(
    field: FieldSpec, entity_label: str, context_chunks: list[dict]
) -> FieldExtraction:
    provider = os.getenv("LLM_PROVIDER", "anthropic").lower()
    call = _PROVIDERS.get(provider)
    if call is None:
        raise ValueError(
            f"Unknown LLM_PROVIDER={provider!r}; expected one of {sorted(_PROVIDERS)}"
        )

    prompt = _build_prompt(field, entity_label, context_chunks)
    result = call(prompt)

    value = result.get("value")
    confidence = float(result.get("confidence", 0.0))
    citation = result.get("citation", "")

    return FieldExtraction(
        value=value,
        confidence=confidence,
        citation=citation,
        needs_review=value is None,
    )

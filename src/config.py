"""Loads the project schema config that drives what gets extracted."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class FieldSpec:
    name: str
    type: str  # "string" | "number" | "text_summary"
    required: bool = False
    source_hint: str = ""
    unit: str | None = None
    max_words: int | None = None


@dataclass
class SchemaConfig:
    project_name: str
    confidence_threshold: float
    fields: list[FieldSpec] = field(default_factory=list)

    def field_names(self) -> list[str]:
        return [f.name for f in self.fields]


def load_schema(path: str | Path) -> SchemaConfig:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    fields = [
        FieldSpec(
            name=f["name"],
            type=f["type"],
            required=f.get("required", False),
            source_hint=f.get("source_hint", "").strip(),
            unit=f.get("unit"),
            max_words=f.get("max_words"),
        )
        for f in raw.get("fields", [])
    ]

    return SchemaConfig(
        project_name=raw.get("project_name", "Untitled Project"),
        confidence_threshold=float(raw.get("confidence_threshold", 0.6)),
        fields=fields,
    )

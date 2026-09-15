"""Shared data structures passed between pipeline stages."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def make_entity_id(company: str, leader_name: str = "") -> str:
    raw = f"{company}_{leader_name}".lower()
    safe = "".join(c if c.isalnum() else "_" for c in raw)
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe.strip("_") or "entity"


@dataclass
class EntityInput:
    """One row from the input CSV: a company/leader to profile."""

    entity_id: str
    company: str
    leader_name: str = ""
    family_name_hint: str = ""
    seed_urls: list[str] = field(default_factory=list)


@dataclass
class RawDocument:
    """A single fetched/scraped document cached for an entity."""

    entity_id: str
    url: str
    title: str
    text: str
    fetched_at: str
    source_type: str = "web_search"  # web_search | seed_url | manual


@dataclass
class FieldExtraction:
    value: Any
    confidence: float
    citation: str
    needs_review: bool = False


@dataclass
class EntityProfile:
    entity_id: str
    fields: dict[str, FieldExtraction] = field(default_factory=dict)

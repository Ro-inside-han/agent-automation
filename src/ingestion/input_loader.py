"""Loads the entity list (companies/leaders to profile) from a CSV."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.models import EntityInput, make_entity_id


def load_entities(csv_path: str | Path) -> list[EntityInput]:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.fillna("")

    if "company" not in df.columns:
        raise ValueError("Input CSV must have a 'company' column")

    entities = []
    for _, row in df.iterrows():
        company = str(row.get("company", "")).strip()
        leader_name = str(row.get("leader_name", "") or "").strip()
        family_name = str(row.get("family_name", "") or "").strip()
        sources_raw = str(row.get("sources", "") or "").strip()
        seed_urls = [u.strip() for u in sources_raw.split("|") if u.strip()]

        entities.append(
            EntityInput(
                entity_id=make_entity_id(company, leader_name),
                company=company,
                leader_name=leader_name,
                family_name_hint=family_name,
                seed_urls=seed_urls,
            )
        )
    return entities

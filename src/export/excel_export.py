"""Writes the assembled entity profiles to an Excel deliverable."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import SchemaConfig
from src.models import EntityProfile


def profiles_to_dataframe(
    profiles: list[EntityProfile], schema: SchemaConfig
) -> pd.DataFrame:
    rows = []
    for profile in profiles:
        row = {"entity_id": profile.entity_id}
        for field in schema.fields:
            extraction = profile.fields.get(field.name)
            if extraction is None:
                row[field.name] = None
                row[f"{field.name}_confidence"] = None
                row[f"{field.name}_citation"] = None
                row[f"{field.name}_needs_review"] = True
                continue
            row[field.name] = extraction.value
            row[f"{field.name}_confidence"] = extraction.confidence
            row[f"{field.name}_citation"] = extraction.citation
            row[f"{field.name}_needs_review"] = extraction.needs_review
        rows.append(row)

    return pd.DataFrame(rows)


def export_profiles(
    profiles: list[EntityProfile], schema: SchemaConfig, output_path: str | Path
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = profiles_to_dataframe(profiles, schema)

    # Two sheets: a clean view for the client/jury, and a full audit trail.
    clean_columns = ["entity_id"] + schema.field_names()
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df[clean_columns].to_excel(writer, sheet_name="Profiles", index=False)
        df.to_excel(writer, sheet_name="Audit Trail", index=False)

    return output_path

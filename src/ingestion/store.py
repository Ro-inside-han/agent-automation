"""Cache for raw fetched documents, keyed by entity, so ingestion is resumable.

For 1000+ entities re-fetching on every run is slow and wastes API/search
quota. Each document is written as one JSON file under
data/raw/<entity_id>/<hash>.json so `ingest` can be safely re-run and will
skip entities that already have cached documents.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from src.models import RawDocument

DEFAULT_RAW_DIR = Path("data/raw")


def _doc_path(raw_dir: Path, doc: RawDocument) -> Path:
    entity_dir = raw_dir / doc.entity_id
    entity_dir.mkdir(parents=True, exist_ok=True)
    url_hash = hashlib.sha1(doc.url.encode("utf-8")).hexdigest()[:16]
    return entity_dir / f"{url_hash}.json"


def save_document(doc: RawDocument, raw_dir: Path = DEFAULT_RAW_DIR) -> Path:
    path = _doc_path(raw_dir, doc)
    path.write_text(json.dumps(asdict(doc), ensure_ascii=False, indent=2))
    return path


def load_documents(entity_id: str, raw_dir: Path = DEFAULT_RAW_DIR) -> list[RawDocument]:
    entity_dir = raw_dir / entity_id
    if not entity_dir.exists():
        return []
    docs = []
    for path in entity_dir.glob("*.json"):
        data = json.loads(path.read_text())
        docs.append(RawDocument(**data))
    return docs


def has_documents(entity_id: str, raw_dir: Path = DEFAULT_RAW_DIR) -> bool:
    entity_dir = raw_dir / entity_id
    return entity_dir.exists() and any(entity_dir.glob("*.json"))


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

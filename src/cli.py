"""Command-line entry point for the profiling pipeline.

    python -m src.cli run --input data/input/companies.csv --schema config/schema.yaml

Stages can also be run independently (useful for 1000+ row datasets where
you want to ingest overnight, then extract in a separate pass):

    python -m src.cli ingest --input data/input/companies.csv
    python -m src.cli index
    python -m src.cli extract --schema config/schema.yaml
    python -m src.cli export --schema config/schema.yaml --output data/output/profiles.xlsx
"""
from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from src.agents.orchestrator import (
    extract_entity_profile,
    ingest_entity,
    index_entity,
)
from src.config import load_schema
from src.export.excel_export import export_profiles
from src.ingestion.input_loader import load_entities
from src.models import EntityProfile

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
INDEX_DIR = Path("data/index")
PROFILES_CACHE = Path("data/output/profiles.pkl")


def cmd_ingest(args: argparse.Namespace) -> None:
    entities = load_entities(args.input)
    for entity in tqdm(entities, desc="ingest"):
        count = ingest_entity(entity, RAW_DIR, force=args.force)
        logger.info("%s: %d documents cached", entity.entity_id, count)


def cmd_index(args: argparse.Namespace) -> None:
    entities = load_entities(args.input)
    for entity in tqdm(entities, desc="index"):
        n_chunks = index_entity(entity.entity_id, RAW_DIR, INDEX_DIR)
        logger.info("%s: %d chunks indexed", entity.entity_id, n_chunks)


def cmd_extract(args: argparse.Namespace) -> None:
    schema = load_schema(args.schema)
    entities = load_entities(args.input)

    profiles: list[EntityProfile] = []
    for entity in tqdm(entities, desc="extract"):
        profile = extract_entity_profile(entity, schema, INDEX_DIR, top_k=args.top_k)
        profiles.append(profile)

    PROFILES_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(PROFILES_CACHE, "wb") as f:
        pickle.dump(profiles, f)
    logger.info("Extracted %d profiles -> %s", len(profiles), PROFILES_CACHE)


def cmd_export(args: argparse.Namespace) -> None:
    schema = load_schema(args.schema)
    if not PROFILES_CACHE.exists():
        raise SystemExit("No extracted profiles found. Run `extract` first.")
    with open(PROFILES_CACHE, "rb") as f:
        profiles = pickle.load(f)
    path = export_profiles(profiles, schema, args.output)
    logger.info("Exported profiles -> %s", path)


def cmd_run(args: argparse.Namespace) -> None:
    cmd_ingest(args)
    cmd_index(args)
    cmd_extract(args)
    cmd_export(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Agentic AI + RAG profiling pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--input", default="data/input/companies.csv")

    p_ingest = sub.add_parser("ingest", parents=[common])
    p_ingest.add_argument("--force", action="store_true", help="re-fetch even if cached")
    p_ingest.set_defaults(func=cmd_ingest)

    p_index = sub.add_parser("index", parents=[common])
    p_index.set_defaults(func=cmd_index)

    p_extract = sub.add_parser("extract", parents=[common])
    p_extract.add_argument("--schema", default="config/schema.yaml")
    p_extract.add_argument("--top-k", type=int, default=5)
    p_extract.set_defaults(func=cmd_extract)

    p_export = sub.add_parser("export")
    p_export.add_argument("--schema", default="config/schema.yaml")
    p_export.add_argument("--output", default="data/output/profiles.xlsx")
    p_export.set_defaults(func=cmd_export)

    p_run = sub.add_parser("run", parents=[common])
    p_run.add_argument("--schema", default="config/schema.yaml")
    p_run.add_argument("--output", default="data/output/profiles.xlsx")
    p_run.add_argument("--force", action="store_true")
    p_run.add_argument("--top-k", type=int, default=5)
    p_run.set_defaults(func=cmd_run)

    return parser


def main() -> None:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

# Agentic AI + RAG Profiling Pipeline

Reusable pipeline for profiling business leaders / family enterprises (or any
similar entity-profiling task) using a retrieval-augmented, agentic
extraction workflow. Built for the "Family Business Excellence" style project
described in `docs/` but designed to be reused across future client projects
by editing a schema config instead of the code.

## Why this design

Manual profiling (search → read → copy into a spreadsheet) does not scale to
1000+ entities and breaks every time a client changes the required columns.
This pipeline separates concerns so that only the **schema** changes between
projects:

1. **Ingestion** — fetch raw source documents per entity (web search, company
   sites, news, CSR reports, SalesQL/LinkedIn exports) and cache them.
2. **RAG index** — chunk + embed the raw documents into a per-entity vector
   store, so any field (current or added later) can be answered by retrieval
   instead of re-scraping.
3. **Agentic extraction** — for each schema field, retrieve the most relevant
   chunks and ask an LLM (Claude) to extract a structured value with a
   confidence score and source citation. Low-confidence fields trigger an
   extra web-search round before falling back to human review.
4. **Export** — assemble all entities into a single Excel/CSV with per-field
   citations, ready for jury/client review.

## Project layout

```
config/schema.yaml          # <-- the only file you edit for a new project
data/
  input/companies.csv       # entity list: name, company, family_name, ...
  raw/<entity_id>/*.json     # cached raw documents (search results, scrapes)
  index/                     # persistent vector store (Chroma)
  output/profiles.xlsx       # final deliverable
src/
  config.py                 # loads config/schema.yaml
  ingestion/                # web search + scraping + raw doc cache
  rag/                      # chunking, embeddings, vector store
  agents/                   # extraction / verification / orchestrator
  export/                   # Excel/CSV writer
  cli.py                    # entry point
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# fill in TAVILY_API_KEY (optional, see above), and an LLM provider:
#   LLM_PROVIDER=anthropic + ANTHROPIC_API_KEY  (paid, best quality), or
#   LLM_PROVIDER=groq      + GROQ_API_KEY       (free tier, console.groq.com)
```

## Usage

```bash
# 1. put your entity list here (name, company, family_name columns required)
#    data/input/companies.csv

# 2. run the full pipeline
python -m src.cli run --input data/input/companies.csv --schema config/schema.yaml

# or run stages individually (useful for 1000+ rows / resumability):
python -m src.cli ingest --input data/input/companies.csv
python -m src.cli index
python -m src.cli extract --schema config/schema.yaml
python -m src.cli export --output data/output/profiles.xlsx
```

## Adapting to a new client project

Only `config/schema.yaml` needs to change: add/remove/rename fields, adjust
`source_hint` text (used to steer the search agent), and change
`type: text_summary` vs `type: number` vs `type: string`. No code changes are
required to onboard a new set of required columns.

## Notes on scale (1000+ entities)

- Ingestion and extraction are resumable: raw documents are cached in
  `data/raw/<entity_id>/`, and the vector index is persistent, so re-running
  `extract` after adding a field does not re-fetch documents.
- Extraction runs are batched with a concurrency limit (`--concurrency`) to
  respect API rate limits.
- Low-confidence fields (below `confidence_threshold` in the schema) are
  written to the output with a `needs_review` flag rather than silently
  guessed.

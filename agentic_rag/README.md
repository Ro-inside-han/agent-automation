# Agentic RAG MVP

Minimal agentic RAG system for the assignment: retrieves from the subject's
study material and generates a cited, evidence-grounded answer to a
faculty-provided query.

## Workflow

```
Input: faculty query
  -> Agent 1 (query understanding & planning): LLM breaks the query into
     search phrases
  -> Retriever: TF-IDF + cosine similarity over chunked study material
     (study_material/*.txt|.md|.pdf) returns the top-k most relevant chunks
  -> Agent 2 (analysis & synthesis): LLM answers the query using only the
     retrieved chunks, citing [source#chunk] for every claim
  -> Output: final answer + list of sources used
```

Retrieval is lexical (TF-IDF), not an embedding vector DB, by design: it
needs no extra API calls/keys, runs instantly, and is fully sufficient for a
single-subject study-material corpus. Only the two agent steps (planning,
synthesis) call the LLM, via OpenCode Zen's OpenAI-compatible API.

## Setup

```bash
cd agentic_rag
pip install -r requirements.txt
cp .env.example .env
# edit .env: set OPENCODE_API_KEY to your OpenCode Zen key
# (get one at https://opencode.ai/docs/zen/)
```

## Add your real study material

Drop your subject's notes/slides/PDFs into `study_material/` (replace or
keep `sample_notes.txt`). No code changes needed — the pipeline re-indexes
every run.

## Run

```bash
python agentic_rag.py "Your faculty-provided query here"
```

or edit `DEFAULT_QUERY` in `agentic_rag.py` and just run `python agentic_rag.py`.

## Grounding guarantee

The synthesis agent is instructed to use *only* the retrieved chunks and to
say so explicitly if the study material doesn't cover something, rather than
answering from outside knowledge — satisfying the "no unsupported
information" requirement.

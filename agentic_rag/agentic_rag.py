#!/usr/bin/env python3
"""
Agentic RAG MVP.

Workflow (matches the assignment's expected pipeline):
  Input: faculty-provided query
    -> Agent: query understanding & task planning (LLM call)
    -> Retriever: TF-IDF lexical search over chunked study material
    -> Agent: analyse & synthesise retrieved chunks into a final answer (LLM call)
    -> Output: final answer + source citations

Knowledge source: every file under study_material/ (.txt/.md/.pdf).
No information outside those files is used to ground the answer.

Usage:
    python agentic_rag.py "Your query here"
    python agentic_rag.py            # uses DEFAULT_QUERY below
"""

import glob
import os
import re
import sys

from dotenv import load_dotenv
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

STUDY_MATERIAL_DIR = os.path.join(os.path.dirname(__file__), "study_material")
CHUNK_SIZE_WORDS = 180
CHUNK_OVERLAP_WORDS = 40
TOP_K = 5

DEFAULT_QUERY = (
    "Explain what makes a RAG system 'agentic' and describe the typical "
    "architecture of an agentic RAG pipeline."
)

client = OpenAI(
    base_url=os.environ.get("OPENCODE_BASE_URL", "https://opencode.ai/zen/v1"),
    api_key=os.environ.get("OPENCODE_API_KEY", "missing"),
)
MODEL = os.environ.get("OPENCODE_MODEL", "big-pickle")


# ---------------------------------------------------------------------------
# Ingestion + chunking
# ---------------------------------------------------------------------------

def load_documents():
    docs = []
    for path in sorted(glob.glob(os.path.join(STUDY_MATERIAL_DIR, "*"))):
        if not os.path.isfile(path):
            continue
        if path.lower().endswith(".pdf"):
            text = _read_pdf(path)
        else:
            with open(path, encoding="utf-8", errors="ignore") as f:
                text = f.read()
        if text.strip():
            docs.append((os.path.basename(path), text))
    return docs


def _read_pdf(path):
    from pypdf import PdfReader

    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def chunk_text(source_name, text):
    words = re.sub(r"\s+", " ", text).strip().split(" ")
    chunks = []
    step = CHUNK_SIZE_WORDS - CHUNK_OVERLAP_WORDS
    for start in range(0, len(words), step):
        piece = " ".join(words[start : start + CHUNK_SIZE_WORDS])
        if piece.strip():
            chunks.append({"source": source_name, "text": piece})
        if start + CHUNK_SIZE_WORDS >= len(words):
            break
    return chunks


def build_index():
    all_chunks = []
    for source_name, text in load_documents():
        all_chunks.extend(chunk_text(source_name, text))
    if not all_chunks:
        raise SystemExit(
            f"No study material found in {STUDY_MATERIAL_DIR}. "
            "Add .txt/.md/.pdf files there first."
        )
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform([c["text"] for c in all_chunks])
    return all_chunks, vectorizer, matrix


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

def retrieve(query, chunks, vectorizer, matrix, top_k=TOP_K):
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, matrix).flatten()
    ranked = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
    results = []
    for i in ranked[:top_k]:
        if scores[i] <= 0:
            continue
        results.append({**chunks[i], "score": float(scores[i]), "chunk_id": i})
    return results


# ---------------------------------------------------------------------------
# Agent 1: query understanding & task planning
# ---------------------------------------------------------------------------

def plan_query(query):
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are the planning agent of a RAG system. Given a user "
                    "query, output 3-6 short search phrases (one per line, no "
                    "numbering) that would help retrieve the most relevant "
                    "passages from a study-material knowledge base to answer it. "
                    "Do not answer the query itself."
                ),
            },
            {"role": "user", "content": query},
        ],
        temperature=0,
    )
    lines = resp.choices[0].message.content.strip().splitlines()
    search_terms = [l.strip("-* ").strip() for l in lines if l.strip()]
    return search_terms or [query]


# ---------------------------------------------------------------------------
# Agent 2: analysis & synthesis
# ---------------------------------------------------------------------------

def synthesize_answer(query, retrieved_chunks):
    context_blocks = []
    for r in retrieved_chunks:
        context_blocks.append(f"[{r['source']}#chunk{r['chunk_id']}]\n{r['text']}")
    context = "\n\n".join(context_blocks)

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are the synthesis agent of a RAG system. Answer the "
                    "user's query using ONLY the information in the provided "
                    "context chunks. Every claim must be traceable to a chunk. "
                    "After each sentence or bullet that uses a fact, cite its "
                    "source tag in square brackets, e.g. [file.txt#chunk3]. "
                    "If the context does not contain enough information to "
                    "answer fully, say so explicitly instead of guessing. Do "
                    "not use outside knowledge."
                ),
            },
            {
                "role": "user",
                "content": f"CONTEXT CHUNKS:\n\n{context}\n\nQUERY: {query}",
            },
        ],
        temperature=0,
    )
    return resp.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run(query):
    print(f"\n=== INPUT QUERY ===\n{query}\n")

    print("=== AGENT: query understanding & task planning ===")
    search_terms = plan_query(query)
    for t in search_terms:
        print(f"  - {t}")

    print("\n=== RETRIEVER: searching study material ===")
    chunks, vectorizer, matrix = build_index()
    seen_ids = set()
    retrieved = []
    for term in search_terms + [query]:
        for r in retrieve(term, chunks, vectorizer, matrix, top_k=TOP_K):
            if r["chunk_id"] not in seen_ids:
                seen_ids.add(r["chunk_id"])
                retrieved.append(r)
    retrieved = sorted(retrieved, key=lambda r: r["score"], reverse=True)[:TOP_K]

    if not retrieved:
        print("  No relevant chunks found in study material.")
        return

    for r in retrieved:
        preview = r["text"][:90].replace("\n", " ")
        print(f"  [{r['source']}#chunk{r['chunk_id']}] score={r['score']:.3f}  {preview}...")

    print("\n=== AGENT: analyse & synthesise retrieved information ===")
    answer = synthesize_answer(query, retrieved)

    print("\n=== FINAL ANSWER ===\n")
    print(answer)

    print("\n=== SOURCES USED ===")
    for r in retrieved:
        print(f"  [{r['source']}#chunk{r['chunk_id']}]")

    return answer


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]).strip() or DEFAULT_QUERY
    run(q)

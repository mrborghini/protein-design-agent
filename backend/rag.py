"""Lightweight local-paper RAG for the Paper Analyst agent.

Reads PDFs from ``BIOAPP_PAPERS_DIR``, chunks them, embeds chunks via Ollama's
embedding endpoint, and answers ``paper_search`` queries by cosine similarity —
pure stdlib + Ollama, **no vector-DB dependency** (per the project's minimal-deps
policy). The index is built lazily on first query and cached for the process.

POC shortcuts (call these out, don't pretend otherwise):
- In-memory index, rebuilt on restart; no incremental updates while running.
- Naive fixed-size chunking and brute-force cosine over all chunks — fine for a
  handful of papers, not a corpus. A real deployment wants a proper vector store.
- Requires an embedding model pulled in Ollama (default ``nomic-embed-text``).
"""
import asyncio
import json
import math
import os
import urllib.request
from pathlib import Path

from autogen_core.tools import FunctionTool

from backend.agents import OLLAMA_HOST
from backend.bioapps.config import PAPERS_DIR
from backend.research import research_sink

EMBED_MODEL = os.environ.get("BIOAPP_EMBED_MODEL", "nomic-embed-text")
_CHUNK_CHARS = 1200
_CHUNK_OVERLAP = 200

# Cached index: list of {source, text, vector}. None until first build.
_index: list[dict] | None = None
_index_lock = asyncio.Lock()


async def _emit(event: dict) -> None:
    queue = research_sink.get(None)
    if queue is not None:
        await queue.put(event)


def _chunk(text: str) -> list[str]:
    text = " ".join(text.split())
    step = max(1, _CHUNK_CHARS - _CHUNK_OVERLAP)
    return [text[i:i + _CHUNK_CHARS] for i in range(0, len(text), step) if text[i:i + _CHUNK_CHARS].strip()]


def _embed_sync(text: str) -> list[float]:
    body = json.dumps({"model": EMBED_MODEL, "prompt": text}).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_HOST}/api/embeddings", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 - local Ollama
        data = json.loads(resp.read().decode("utf-8"))
    vec = data.get("embedding")
    if not vec:
        raise RuntimeError(f"Ollama returned no embedding (is '{EMBED_MODEL}' pulled?).")
    return vec


def _load_pdf_text(path: Path) -> str:
    from pypdf import PdfReader  # lazy, mirrors backend/main.py
    try:
        reader = PdfReader(str(path))
        return "\n".join((p.extract_text() or "") for p in reader.pages)
    except Exception:  # noqa: BLE001 - skip unreadable PDFs, don't abort the whole index
        return ""


def _build_index_sync() -> list[dict]:
    if not PAPERS_DIR.is_dir():
        return []
    index: list[dict] = []
    for pdf in sorted(PAPERS_DIR.glob("*.pdf")):
        for chunk in _chunk(_load_pdf_text(pdf)):
            index.append({"source": pdf.name, "text": chunk, "vector": _embed_sync(chunk)})
    return index


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


async def _ensure_index() -> list[dict]:
    global _index
    async with _index_lock:
        if _index is None:
            await _emit({"type": "status", "stage": "rag", "text": "Indexing local papers…"})
            _index = await asyncio.to_thread(_build_index_sync)
        return _index


async def paper_search(query: str, k: int = 5) -> str:
    """Search the local research-paper library for passages relevant to `query`.

    Use this to pull specific mutations, melting temperatures (Tm), structural
    domains, or experimental findings from papers placed in the papers folder.
    Returns the top `k` matching passages with their source filenames.
    """
    index = await _ensure_index()
    if not index:
        return (
            f"No papers indexed — add PDFs to '{PAPERS_DIR}' (or set BIOAPP_PAPERS_DIR) and "
            "ensure the embedding model is available."
        )
    await _emit({"type": "status", "stage": "rag", "text": f"Searching {len(index)} passages for: {query}"})
    try:
        q = await asyncio.to_thread(_embed_sync, query)
    except Exception as e:  # noqa: BLE001 - surface to the agent
        return f"Paper search failed (embedding error): {e}"

    ranked = sorted(index, key=lambda c: _cosine(q, c["vector"]), reverse=True)[:max(1, k)]
    await _emit({
        "type": "research",
        "query": f"papers: {query}",
        "sources": [{"title": c["source"], "url": ""} for c in ranked],
        "screenshot_b64": "",
    })
    return "Top local-paper passages:\n\n" + "\n\n".join(
        f"### {c['source']}\n{c['text']}" for c in ranked
    )


paper_search_tool = FunctionTool(
    paper_search,
    description="Search the local research-paper library (PDFs) for passages relevant to a query.",
)

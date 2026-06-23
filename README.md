# Protein Design Agent System

A local multi-agent system utilizing AutoGen and Ollama for protein design analysis, incorporating AlphaFold, CollabFold, and RFdiffusion workflows.

## Architecture

- **Coordinator Agent**: Oversees the design cycle.
- **Qwen Summarizer & Teacher**: Processes original biological papers and extracts critical parameters.
- **DeepSeek Critic**: Compares the summaries directly against the original papers to eliminate hallucinations.
- **Hypothesis Agent**: Proposes testable mutations and experiments.
- **Literature Agent**: Retrieves paper chunks via RAG.

## Prerequisites

- Python 3.10+
- Local [Ollama](https://ollama.com/) installation.
- Pull the required models:
  ```bash
  ollama pull qwen2.5
  ollama pull deepseek-coder
  ```

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/mrborghini/protein-design-agent.git
   cd protein-design-agent
   ```

2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

Ensure your local Ollama server is running. If you are accessing Ollama over Tailscale or another network interface, set:
```bash
export OLLAMA_HOST=0.0.0.0
```

## Running the Agent (CLI)

To start the one-shot agentic loop:
```bash
python agent.py
```

## Web Interface

A local web app lets you upload a PDF and chat with a **roster of agents that
debate until they reach consensus** (round-robin, capped by a max-turns limit).
Agents can browse the web with **headless Playwright**, and the research activity
(sources + screenshots) is streamed back into the chat. Everything stays local:
inference via Ollama, browsing via local Playwright, static files served by the
backend.

Features: Markdown-rendered replies, light/dark mode, a configurable agent roster
(add/remove agents, pick any locally-installed Ollama model per agent — discovered
live via `/api/tags`), and adjustable context window + debate length.

Additional UI features:

- **Per-agent context window slider** (up to **256K** tokens), with friendly `K`
  labels — no raw token typing. A global slider seeds new agents; each agent can
  override it in the roster.
- **Token usage** per agent (tokens *generated* = completion tokens) plus running
  totals, read from each turn's model usage.
- **Reset to defaults** rebuilds the original three-agent pipeline (default prompts)
  on the first three installed models.
- **Save / load configuration**: download the roster + settings as JSON and upload it
  back later.
- **Image upload** (vision): attach images to a question. Vision capability is detected
  per model via Ollama `/api/show`; images are only used by vision-capable agents
  (others ignore them). **Disclaimer: not all models support vision.**
- **Collapsible debate transcript**: each question's agent turns are grouped under a
  collapsible "Debate" dropdown.
- **Critic targeting**: the Critic is a protected (non-removable) default agent with an
  editable prompt; choose which agent(s) it critiques (defaults to the first).
- **Persistent conversation**: the chat is kept across reloads until you press
  **Clear chat**, and can be exported with **Download chat** (Markdown).

### Quick start (one command)

```bash
./start.sh        # builds the frontend, then serves it at http://localhost:8000
```

`start.sh` is idempotent: it creates the venv, installs deps, downloads the
Playwright browser if missing, builds the SPA with Bun, and launches uvicorn.
Honors `OLLAMA_HOST`, `OLLAMA_NUM_CTX`, and `PORT`.

### Manual setup

### Backend (FastAPI)

```bash
source venv/bin/activate
pip install -r backend/requirements.txt
playwright install chromium          # one-time browser download
uvicorn backend.main:app --reload --port 8000
```

Override the Ollama host with `OLLAMA_HOST` (defaults to `http://localhost:11434`).

**Context window (`num_ctx`).** Each agent call sets Ollama's `num_ctx` via the
request `options` object (per https://docs.ollama.com/). This caps the KV-cache
memory and stops long PDF + tool-call histories from being truncated. The default
is `32768`, overridable with `OLLAMA_NUM_CTX`. In the web UI it is set **per agent**
(each agent has its own slider) with a global default seeding new agents; values are
clamped to 512–262144 (256K).

**Web research.** The Literature agent's `web_research` tool uses a hardened
headless browser (realistic UA/headers, `navigator.webdriver` masking,
consent-banner dismissal) so DuckDuckGo works headless, with a Wikipedia fallback.
Result pages are read via a fast stdlib `urllib` path, falling back to the browser
for JS-heavy pages. This is for benign, low-volume reads of public pages — not
mass scraping.

### Frontend (Bun + React + Tailwind)

```bash
cd frontend
bun install
bun run dev                          # dev server on :5173, proxies /api to :8000
```

For a single-process static deployment, build the SPA and let FastAPI serve it:

```bash
cd frontend && bun run build         # outputs frontend/dist
uvicorn backend.main:app --port 8000 # now serves the UI at http://localhost:8000/
```

> **POC note:** web research scrapes DuckDuckGo's HTML endpoint (not a stable
> API), there is no auth/rate-limiting, and a single in-memory session holds the
> uploaded PDF. Intended for localhost use only.

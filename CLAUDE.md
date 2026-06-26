# CLAUDE.md — protein-design-agent

Guidance for working in this repo. This is a **local proof-of-concept**: a multi-agent
protein-design debate (AutoGen + local Ollama models) — plus an orchestrated **Pipeline mode**
that drives structural-biology tools (Boltz-2, RFdiffusion, ProteinMPNN, PyRosetta) — with a
FastAPI backend that also serves a built React/Vite/Tailwind SPA. Browsing uses headless
Playwright. Everything runs locally — no cloud.

## Do

- **Frontend: use `bun`.** Build/type-check with `bun run build` (runs `tsc && vite build`);
  install with `bun install`. Run these from `frontend/`.
- **Backend: use the Python venv.** `source venv/bin/activate`; dependencies live in
  `backend/requirements.txt`. Import-check with `python -c "import backend.main"`.
- **TypeScript** for frontend code; keep dependencies minimal (prefer stdlib / what's already
  here).
- **Treat everything as POC.** Call out shortcuts and missing production concerns explicitly
  (auth, persistence, scaling). Don't claim production-readiness.
- **Verify against live Ollama** when feasible before declaring something works; report failures
  honestly. Note that local thinking models are slow (~minutes/turn) and large models can OOM.
- **Commit only when asked.** Conventional commit messages; work is committed directly to `main`
  per the maintainer's preference.

## Don't

- **Do NOT start or run the dev server.** The user starts it themselves (`./start.sh`, or
  `uvicorn backend.main:app --port 8000`). Never launch it — not in the foreground and not in the
  background.
- **Do NOT use pnpm or npm** for the frontend — bun only (no `|| pnpm …` fallbacks).
- **Do NOT hardcode secrets/credentials**; no PII or proprietary data in prompts.
- **Do NOT commit build artifacts** (`frontend/dist/`) — they're gitignored.

## How it runs (for reference — the user runs this, not Claude)

```bash
./start.sh                 # builds the SPA (bun) + starts uvicorn on :8000
# or, manually:
cd frontend && bun run build
source venv/bin/activate && uvicorn backend.main:app --port 8000
```
Requires a local Ollama (`ollama serve`) with the rostered models pulled.

## Architecture map

- `backend/main.py` — FastAPI app: SSE `/api/chat` (debate) and `/api/pipeline` (pipeline mode),
  model/capability discovery, PDF upload, the **shared conversation** endpoints
  (`GET /api/conversation`, `POST /api/conversation/clear`), the `/api/artifact/{path}` download
  route, and static SPA serving. The conversation is global and shared across all sessions; one
  run at a time (409 while busy). `_sse_response` is the shared SSE responder for both modes.
- `backend/agents.py` — single source of truth for the agent roster + model clients
  (`DEFAULT_AGENTS`, `build_roster`, `build_agent`). The Critic critiques all agents by default
  and has a clarification tool. `build_agent` takes `extra_tools` (used by Pipeline mode).
- `backend/termination.py` — `DebateTermination`: ends on unanimous consensus, max rounds, or a
  stuck loop (`max_rounds=None` ⇒ unlimited). A "turn" = one full round (every agent speaks once).
- `backend/streaming_client.py` — Ollama client subclass that streams answer tokens and a
  separate thinking channel.
- `backend/session.py` — in-memory shared state (PDF context + conversation/busy/status/usage).
  Conversation items now include `gpu`/`bioapp`/`artifact` kinds (Pipeline mode).
- `backend/pipeline.py` — **Pipeline mode**: a `SelectorGroupChat` of Professor (orchestrator,
  no tools) → Paper Analyst (RAG) + Bio-App Operator (the four bio-app tools). Ends on the
  Professor's `DESIGN_COMPLETE` token or a message backstop. `DEFAULT_PIPELINE_AGENTS`, `build_pipeline`.
- `backend/bioapps/` — external structural-biology tools, each in its **own venv** (see
  `SETUP_BIOAPPS.md`). `config.py` (env-driven paths/timeouts), `runner.py` (`run_in_env` —
  subprocess in the tool's venv, streams `bioapp` logs, GPU tools wrapped in `gpu_exclusive`),
  and one module per tool: `boltz`, `rfdiffusion`, `proteinmpnn`, `pyrosetta` (each exposes a
  `FunctionTool` + `emit_artifact`).
- `backend/bioapps/taskity.py` — **taskities**: the Operator can author/run reusable Python scripts
  in venvs (`taskity_create/update/run/view/list/delete`). A script reads params from `sys.argv[1]`
  (JSON) and prints one final JSON line; runs in a chosen bio-app venv (`base_venv`, GPU-aware via
  `run_in_env`) or a dedicated pip-installed venv. Lint-guarded, timeout/output-capped, persisted
  under `<WORKDIR>/taskities`. **Every script is Critic-reviewed and must be APPROVED before it
  runs** (fail-closed gate in `_run`; `set_reviewer` is injected by `pipeline.build_pipeline`;
  approval cached per script sha256). ⚠️ executes arbitrary local Python — local POC only, no sandbox.
- `backend/gpu.py` — `gpu_exclusive()`: serializes GPU use and **evicts Ollama models from VRAM**
  (`/api/ps` + `keep_alive:0`) before a GPU bio-app runs (the 32 GB GPU can't hold the LLM + a
  structural-biology model at once). The LLM reloads lazily on the next turn.
- `backend/rag.py` — lightweight local-PDF RAG (`paper_search` tool): chunk + Ollama embeddings +
  pure-Python cosine, no vector-DB dep. POC: in-memory, naive chunking.
- `frontend/src/App.tsx` — main UI; mode toggle (Debate/Pipeline), polls the shared conversation,
  streams live for the initiator. Pipeline mode has its own role/model selector + max-messages.
- `frontend/src/lib/sse.ts` — SSE client + request/event types (`streamChat`, `streamPipeline`).
- `frontend/src/components/` — `Chat`, `DebatePanel` (mode-aware), `AgentMessage`, `AgentRoster`,
  `BioAppCard` (GPU banner / bio-app job / artifact chip), etc.

## Conventions

- Default Ollama host: `http://localhost:11434` (override via `OLLAMA_HOST`).
- Dark mode is a flat `#333` palette with white/neutral text.
- **Pipeline mode is POC scaffolding**: it shells out to externally-installed bio-app venvs
  (`SETUP_BIOAPPS.md`); nothing runs until those are set up. GPU tools pause the run to swap the
  LLM out of VRAM. PyRosetta needs a (free, academic) license. Real runs take minutes→hours.

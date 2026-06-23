# CLAUDE.md — protein-design-agent

Guidance for working in this repo. This is a **local proof-of-concept**: a multi-agent
protein-design debate (AutoGen + local Ollama models) with a FastAPI backend that also
serves a built React/Vite/Tailwind SPA. Browsing uses headless Playwright. Everything runs
locally — no cloud.

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

- `backend/main.py` — FastAPI app: SSE `/api/chat`, model/capability discovery, PDF upload,
  the **shared conversation** endpoints (`GET /api/conversation`, `POST /api/conversation/clear`),
  and static SPA serving. The conversation is global and shared across all sessions; one debate
  runs at a time (409 while busy).
- `backend/agents.py` — single source of truth for the agent roster + model clients
  (`DEFAULT_AGENTS`, `build_roster`, `build_agent`). The Critic critiques all agents by default
  and has a clarification tool.
- `backend/termination.py` — `DebateTermination`: ends on unanimous consensus, max rounds, or a
  stuck loop (`max_rounds=None` ⇒ unlimited). A "turn" = one full round (every agent speaks once).
- `backend/streaming_client.py` — Ollama client subclass that streams answer tokens and a
  separate thinking channel.
- `backend/session.py` — in-memory shared state (PDF context + conversation/busy/status/usage).
- `frontend/src/App.tsx` — main UI; polls the shared conversation, streams live for the initiator.
- `frontend/src/lib/sse.ts` — SSE client + request/event types.
- `frontend/src/components/` — `Chat`, `DebatePanel`, `AgentMessage`, `AgentRoster`, etc.

## Conventions

- Default Ollama host: `http://localhost:11434` (override via `OLLAMA_HOST`).
- Dark mode is a flat `#333` palette with white/neutral text.

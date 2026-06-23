#!/usr/bin/env bash
#
# One-shot build + run for the Protein Design Agent web app.
# Builds the React frontend and starts the FastAPI server, which serves the
# built SPA statically. Everything stays local (Ollama + Playwright).
#
# Env vars (optional):
#   OLLAMA_HOST     Ollama URL (default http://localhost:11434)
#   OLLAMA_NUM_CTX  Default context window (default 32768)
#   PORT            Server port (default 8000)
#
set -euo pipefail
cd "$(dirname "$0")"

VENV="${VENV:-venv}"
PORT="${PORT:-8000}"

echo "==> Python environment"
if [ ! -d "$VENV" ]; then
  echo "    creating venv ($VENV)"
  python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install -q -r backend/requirements.txt

echo "==> Playwright browser"
# Install Chromium only if it isn't already present (one-time ~100MB download).
if ! python -c "from pathlib import Path; import glob, sys; \
  sys.exit(0 if glob.glob(str(Path.home()/'.cache/ms-playwright/chromium*')) else 1)" 2>/dev/null; then
  echo "    downloading Chromium"
  playwright install chromium
else
  echo "    already installed"
fi

echo "==> Frontend build"
if ! command -v bun >/dev/null 2>&1; then
  echo "    ERROR: bun is not installed (https://bun.sh)" >&2
  exit 1
fi
( cd frontend && bun install && bun run build )

echo "==> Checking Ollama at ${OLLAMA_HOST:-http://localhost:11434}"
curl -fsS "${OLLAMA_HOST:-http://localhost:11434}/api/tags" >/dev/null 2>&1 \
  && echo "    reachable" \
  || echo "    WARNING: Ollama not reachable — start it with 'ollama serve'"

echo "==> Starting server on http://localhost:${PORT}"
exec uvicorn backend.main:app --host 0.0.0.0 --port "${PORT}"

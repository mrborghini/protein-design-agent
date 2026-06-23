"""FastAPI app: PDF upload, SSE consensus-debate chat, model discovery, static SPA.

Run:  uvicorn backend.main:app --reload --port 8000
Everything is local: inference via Ollama, browsing via headless Playwright.
"""
import asyncio
import json
import os
import re
import urllib.request
from io import BytesIO

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from autogen_agentchat.messages import TextMessage, MultiModalMessage
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.base import TaskResult
from autogen_core import CancellationToken, Image as AGImage

from backend.agents import build_roster, DEFAULT_AGENTS, DEFAULT_NUM_CTX, OLLAMA_HOST
from backend.research import research_sink
from backend.session import session

NUM_CTX_MIN = 512
NUM_CTX_MAX = 262144  # 256K — some models (e.g. long-context Qwen/Llama) support this
DEFAULT_MAX_TURNS = 12
MAX_TURNS_MIN = 2
MAX_TURNS_MAX = 40

app = FastAPI(title="Protein Design Agent")

# Dev convenience: the Vite dev server runs on a different port.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_DONE = object()  # sentinel marking the end of an event stream


class AgentConfig(BaseModel):
    name: str
    model: str
    system_message: str = ""
    with_research: bool = False
    num_ctx: int | None = None
    is_critic: bool = False
    critiques: list[str] | None = None


class ChatRequest(BaseModel):
    message: str
    num_ctx: int | None = None  # fallback default for agents without their own num_ctx
    max_turns: int | None = None
    agents: list[AgentConfig] | None = None
    images: list[str] | None = None  # base64 (data-URL or raw) images for vision agents


def clamp_num_ctx(n: int | None) -> int:
    """Bound the requested context window; fall back to the configured default."""
    if not n:
        return DEFAULT_NUM_CTX
    return max(NUM_CTX_MIN, min(NUM_CTX_MAX, n))


def clamp_max_turns(n: int | None) -> int:
    if not n:
        return DEFAULT_MAX_TURNS
    return max(MAX_TURNS_MIN, min(MAX_TURNS_MAX, n))


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "pdf_loaded": bool(session.pdf_text),
        "default_num_ctx": DEFAULT_NUM_CTX,
        "num_ctx_min": NUM_CTX_MIN,
        "num_ctx_max": NUM_CTX_MAX,
        "default_max_turns": DEFAULT_MAX_TURNS,
        "max_turns_min": MAX_TURNS_MIN,
        "max_turns_max": MAX_TURNS_MAX,
        "default_agents": DEFAULT_AGENTS,
    }


def _fetch_models_sync() -> list[str]:
    req = urllib.request.Request(f"{OLLAMA_HOST}/api/tags")
    with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310 - local Ollama
        data = json.loads(resp.read().decode("utf-8"))
    return sorted(m["name"] for m in data.get("models", []))


# Per-model vision capability, cached. None = unknown (older Ollama without the
# `capabilities` field); True/False once /api/show reports it.
_vision_cache: dict[str, bool | None] = {}


def _model_vision_sync(name: str) -> bool | None:
    """Query Ollama /api/show for a model's capabilities; True if it supports vision."""
    if name in _vision_cache:
        return _vision_cache[name]
    vision: bool | None = None
    try:
        body = json.dumps({"model": name}).encode("utf-8")
        req = urllib.request.Request(
            f"{OLLAMA_HOST}/api/show", data=body, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310 - local Ollama
            info = json.loads(resp.read().decode("utf-8"))
        caps = info.get("capabilities")
        if isinstance(caps, list):
            vision = "vision" in caps
    except Exception:  # noqa: BLE001 - capability stays unknown
        vision = None
    _vision_cache[name] = vision
    return vision


def _vision_models_sync(names: list[str]) -> set[str]:
    return {n for n in names if _model_vision_sync(n) is True}


@app.get("/api/models")
async def models():
    """List models available on the local Ollama server (proxies /api/tags)."""
    try:
        names = await asyncio.to_thread(_fetch_models_sync)
        return {"models": names}
    except Exception as e:  # noqa: BLE001 - surface a usable error to the UI
        return {"models": [], "error": str(e)}


@app.get("/api/models/capabilities")
async def model_capabilities():
    """Report per-model vision capability (via Ollama /api/show)."""
    try:
        names = await asyncio.to_thread(_fetch_models_sync)
        caps = await asyncio.to_thread(lambda: {n: {"vision": _model_vision_sync(n)} for n in names})
        return {"capabilities": caps}
    except Exception as e:  # noqa: BLE001
        return {"capabilities": {}, "error": str(e)}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")

    from pypdf import PdfReader

    raw = await file.read()
    try:
        reader = PdfReader(BytesIO(raw))
        pages = [(page.extract_text() or "") for page in reader.pages]
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Could not read PDF: {e}")

    text = "\n\n".join(pages).strip()
    session.set_pdf(file.filename, text, len(reader.pages))
    return {"name": file.filename, "pages": len(reader.pages), "chars": len(text)}


def _strip_data_url(b64: str) -> str:
    """Accept either a raw base64 string or a `data:image/...;base64,XXXX` data URL."""
    if b64.startswith("data:") and "," in b64:
        return b64.split(",", 1)[1]
    return b64


async def _debate(
    message: str,
    agents_cfg: list[dict] | None,
    queue: asyncio.Queue,
    token: CancellationToken,
    num_ctx: int,
    max_turns: int,
    images: list[str] | None = None,
    vision_models: set[str] | None = None,
) -> None:
    """Round-robin consensus debate. Streams each agent turn onto `queue`."""
    try:
        configs = agents_cfg if agents_cfg else DEFAULT_AGENTS
        roster_models = {c["model"] for c in configs}
        actual_vision = (vision_models or set()) & roster_models

        # Decide whether images can be attached. We send them with the (shared)
        # round-robin seed message; non-vision models simply ignore the image bytes
        # (Ollama behaviour). We still require at least one vision agent so the image
        # is actually used, and we flag any agents that will ignore it.
        build_vision = actual_vision
        if images:
            if not actual_vision:
                await queue.put({
                    "type": "error",
                    "text": "No vision-capable agent in the roster. Remove the image or add a vision model.",
                })
                return
            ignored = roster_models - actual_vision
            if ignored:
                await queue.put({
                    "type": "status", "stage": "vision",
                    "text": f"{len(ignored)} agent(s) without vision will ignore the image.",
                })
            # Mark all clients vision-enabled so AutoGen doesn't reject the shared
            # image content; only true vision models will actually use it.
            build_vision = roster_models

        roster = build_roster(agents_cfg, num_ctx=num_ctx, consensus=True, vision_models=build_vision)
        await queue.put(
            {"type": "status", "stage": "debate", "text": f"Debating with {len(roster)} agents (max {max_turns} turns)…"}
        )

        # Local models don't always reproduce the exact uppercase token, so accept
        # common case variants. The hard max_turns cap is the backstop.
        consensus_hit = (
            TextMentionTermination("CONSENSUS")
            | TextMentionTermination("Consensus")
            | TextMentionTermination("consensus")
        )
        termination = consensus_hit | MaxMessageTermination(max_turns)
        team = RoundRobinGroupChat(roster, termination_condition=termination, max_turns=max_turns)

        doc = session.context_block()
        doc_prefix = f"{doc}\n\n" if doc else ""
        # NB: keep the literal consensus token OUT of the task text — it lives in
        # each agent's system prompt. Putting it here would trip TextMentionTermination
        # on the seed message and end the debate before anyone speaks.
        task_text = (
            f"{doc_prefix}User question: {message}\n\n"
            "Collaborate to produce a single agreed answer. Use web_research when external "
            "facts would help, and signal agreement exactly as your instructions describe."
        )

        task: str | MultiModalMessage = task_text
        if images:
            content: list = [task_text]
            content.extend(AGImage.from_base64(_strip_data_url(b)) for b in images)
            task = MultiModalMessage(content=content, source="user")

        async for msg in team.run_stream(task=task, cancellation_token=token):
            if isinstance(msg, TaskResult):
                break
            if isinstance(msg, TextMessage) and msg.source != "user":
                content_text = msg.content or ""
                if "consensus" in content_text.lower():
                    # Strip the agreement token (any case / markdown emphasis) for display.
                    stripped = re.sub(r"\**\bconsensus\b\**\.?", "", content_text, flags=re.IGNORECASE).strip()
                    if stripped:
                        await queue.put({"type": "message", "agent": msg.source, "content": stripped})
                    await queue.put({"type": "status", "stage": "consensus", "text": "Consensus reached ✓"})
                else:
                    await queue.put({"type": "message", "agent": msg.source, "content": content_text})
                # Emit token usage for this turn when the client reports it.
                usage = getattr(msg, "models_usage", None)
                if usage is not None:
                    await queue.put({
                        "type": "usage",
                        "agent": msg.source,
                        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
                    })

    except asyncio.CancelledError:
        raise
    except Exception as e:  # noqa: BLE001
        await queue.put({"type": "error", "text": str(e)})
    finally:
        await queue.put(_DONE)


@app.post("/api/chat")
async def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Empty message.")

    queue: asyncio.Queue = asyncio.Queue()
    token = CancellationToken()
    num_ctx = clamp_num_ctx(req.num_ctx)
    max_turns = clamp_max_turns(req.max_turns)
    agents_cfg = [a.model_dump() for a in req.agents] if req.agents else None
    # Clamp each agent's own context window (falling back to the request default).
    for a in agents_cfg or []:
        a["num_ctx"] = clamp_num_ctx(a.get("num_ctx") or num_ctx)

    # Only resolve vision capability when an image is actually attached.
    vision_models: set[str] = set()
    if req.images:
        roster_models = [c["model"] for c in (agents_cfg or DEFAULT_AGENTS)]
        vision_models = await asyncio.to_thread(_vision_models_sync, roster_models)

    # Set the sink BEFORE creating the task so the task (and the research tool
    # running inside it) inherits this client's queue via contextvars.
    research_sink.set(queue)
    task = asyncio.create_task(
        _debate(req.message, agents_cfg, queue, token, num_ctx, max_turns, req.images, vision_models)
    )

    async def event_stream():
        try:
            while True:
                event = await queue.get()
                if event is _DONE:
                    yield "data: {\"type\": \"done\"}\n\n"
                    break
                yield f"data: {json.dumps(event)}\n\n"
        except asyncio.CancelledError:
            # Client disconnected mid-stream — cancel the in-flight debate.
            token.cancel()
            task.cancel()
            raise

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# Serve the built SPA if it exists (production / static-serve mode).
_DIST = os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")
if os.path.isdir(_DIST):
    app.mount("/", StaticFiles(directory=_DIST, html=True), name="spa")

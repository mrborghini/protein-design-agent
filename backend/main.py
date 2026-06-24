"""FastAPI app: PDF upload, SSE consensus-debate chat, model discovery, static SPA.

Run:  uvicorn backend.main:app --reload --port 8000
Everything is local: inference via Ollama, browsing via headless Playwright.
"""
import asyncio
import json
import os
import urllib.request
from datetime import datetime, timezone
from io import BytesIO

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage, MultiModalMessage, ModelClientStreamingChunkEvent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.base import TaskResult
from autogen_core import CancellationToken, Image as AGImage
from autogen_core.tools import FunctionTool

from backend.agents import build_roster, safe_name, DEFAULT_AGENTS, DEFAULT_NUM_CTX, OLLAMA_HOST
from backend.research import research_sink
from backend.session import session
from backend.termination import (
    DebateTermination,
    signals_consensus,
    strip_consensus,
    REASON_CONSENSUS,
    REASON_STUCK_LOOP,
)

NUM_CTX_MIN = 512
NUM_CTX_MAX = 262144  # 256K — some models (e.g. long-context Qwen/Llama) support this
# "Turns" are ROUNDS: one turn = every agent speaks once, in order. The bounded cap
# is MAX_TURNS_MAX; the UI's "No limit" checkbox sends `unlimited` for unbounded rounds.
DEFAULT_MAX_TURNS = 20
MAX_TURNS_MIN = 1
MAX_TURNS_MAX = 100

app = FastAPI(title="Multi-Agent Debate")

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
    # Manual vision override: None ⇒ auto-detect via /api/show; True/False ⇒ force.
    vision: bool | None = None
    # Optional per-agent Ollama sampling knobs (None ⇒ use Ollama's own default).
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    repeat_penalty: float | None = None
    num_predict: int | None = None


class ChatRequest(BaseModel):
    message: str
    num_ctx: int | None = None  # fallback default for agents without their own num_ctx
    max_turns: int | None = None  # = max ROUNDS (one round = every agent speaks once)
    unlimited: bool = False  # run until consensus/deadlock/Stop (no round cap)
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


# Bounds for the per-agent sampling sliders. Defensive clamping only — values come
# from the client. Keys mirror SAMPLING_KEYS in backend/agents.py (num_ctx handled
# separately). num_predict's -1 means "unlimited" (generate until the context fills);
# its upper bound is the agent's context window (see clamp_sampling), so the value here
# is only a fallback used if num_ctx is somehow absent.
SAMPLING_BOUNDS = {
    "temperature": (0.0, 2.0),
    "top_p": (0.0, 1.0),
    "top_k": (0, 100),
    "min_p": (0.0, 1.0),
    "repeat_penalty": (0.5, 2.0),
    "num_predict": (-1, NUM_CTX_MAX),
}


def clamp_sampling(cfg: dict) -> None:
    """Clamp present sampling knobs in-place to their slider bounds. num_predict is
    capped to the agent's resolved context window (num_ctx must already be on cfg)."""
    for key, (lo, hi) in SAMPLING_BOUNDS.items():
        v = cfg.get(key)
        if v is None:
            continue
        if key == "num_predict":
            hi = cfg.get("num_ctx", hi)  # -1 stays unlimited; else can't exceed the context window
        cfg[key] = max(lo, min(hi, v))


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


# Per-model capability list, cached. None = unknown (older Ollama without the
# `capabilities` field); a list once /api/show reports it.
_caps_cache: dict[str, list[str] | None] = {}


def _model_caps_sync(name: str) -> list[str] | None:
    """Query Ollama /api/show for a model's capabilities list (e.g. vision, thinking)."""
    if name in _caps_cache:
        return _caps_cache[name]
    caps: list[str] | None = None
    try:
        body = json.dumps({"model": name}).encode("utf-8")
        req = urllib.request.Request(
            f"{OLLAMA_HOST}/api/show", data=body, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310 - local Ollama
            info = json.loads(resp.read().decode("utf-8"))
        value = info.get("capabilities")
        if isinstance(value, list):
            caps = value
    except Exception:  # noqa: BLE001 - capability stays unknown
        caps = None
    _caps_cache[name] = caps
    return caps


def _has_cap(name: str, cap: str) -> bool | None:
    caps = _model_caps_sync(name)
    return None if caps is None else (cap in caps)


def _vision_models_sync(names: list[str]) -> set[str]:
    return {n for n in names if _has_cap(n, "vision") is True}


def _thinking_models_sync(names: list[str]) -> set[str]:
    return {n for n in names if _has_cap(n, "thinking") is True}


def _tools_models_sync(names: list[str]) -> set[str]:
    # Unknown capability (None, older Ollama without the field) ⇒ treated as
    # unsupported, matching the vision/thinking convention above.
    return {n for n in names if _has_cap(n, "tools") is True}


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
    """Report per-model vision + thinking + tools capability (via Ollama /api/show)."""
    def collect() -> dict:
        out = {}
        for n in _fetch_models_sync():
            out[n] = {
                "vision": _has_cap(n, "vision"),
                "thinking": _has_cap(n, "thinking"),
                "tools": _has_cap(n, "tools"),
            }
        return out

    try:
        caps = await asyncio.to_thread(collect)
        return {"capabilities": caps}
    except Exception as e:  # noqa: BLE001
        return {"capabilities": {}, "error": str(e)}


@app.post("/api/models/capabilities/refresh")
async def refresh_capabilities():
    """Clear the per-model capability cache so the next read re-queries /api/show.

    The cache (`_caps_cache`) otherwise lives for the process lifetime; this lets the
    UI pick up a model that was pulled/updated after the server started.
    """
    _caps_cache.clear()
    return {"ok": True}


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


async def _forward(msg, queue: asyncio.Queue, round_no: int | None = None) -> None:
    """Emit one agent event onto the SSE queue (shared by the debate + closing turn).

    Streaming chunks become `delta`s; complete agent messages become `message`s
    (with the consensus token stripped for display) plus a `usage` event. When
    `round_no` is given (the main debate loop), it's attached so the UI can show a
    round badge; the closing/clarification messages pass None (no round).
    """
    if isinstance(msg, ModelClientStreamingChunkEvent) and msg.source != "user":
        if msg.content:
            ev = {"type": "delta", "agent": msg.source, "content": msg.content}
            if round_no is not None:
                ev["round"] = round_no
            await queue.put(ev)
        return
    if isinstance(msg, TextMessage) and msg.source != "user":
        text = msg.content or ""
        if signals_consensus(text):
            # Pure-agreement turn: show the substantive remainder, or a short marker
            # so the streamed bubble reconciles to something meaningful (not "CONSENSUS").
            text = strip_consensus(text) or "✓ Agrees — consensus"
        elif not text.strip():
            # Empty turn (e.g. a thinking model that spent its budget on reasoning).
            # Don't emit an empty bubble; there were no deltas to reconcile.
            return
        ev = {"type": "message", "agent": msg.source, "content": text}
        if round_no is not None:
            ev["round"] = round_no
        await queue.put(ev)
        usage = getattr(msg, "models_usage", None)
        if usage is not None:
            await queue.put({
                "type": "usage",
                "agent": msg.source,
                "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
            })


async def _debate(
    message: str,
    agents_cfg: list[dict] | None,
    queue: asyncio.Queue,
    token: CancellationToken,
    num_ctx: int,
    max_rounds: int | None,  # max ROUNDS (one round = every agent speaks once); None = unlimited
    images: list[str] | None = None,
    vision_models: set[str] | None = None,
    thinking_models: set[str] | None = None,
    tools_models: set[str] | None = None,
) -> None:
    """Round-robin consensus debate. Streams each agent turn onto `queue`."""
    try:
        configs = agents_cfg if agents_cfg else DEFAULT_AGENTS
        roster_models = {c["model"] for c in configs}
        auto_vision = (vision_models or set()) & roster_models

        # Resolve which models carry vision. A per-agent `vision` override wins over
        # auto-detection (None/absent ⇒ auto). Resolved per *model*: if two agents share
        # one model with conflicting overrides, the model is treated as vision if ANY
        # agent forces it on. With these flags correct, AutoGen's _get_compatible_context
        # strips images for non-vision agents on its own, so we no longer force every
        # client to vision=True (which used to push image bytes into models that can't
        # handle them).
        def _agent_vision(c: dict) -> bool:
            v = c.get("vision")
            return (c["model"] in auto_vision) if v is None else bool(v)

        effective_vision = {c["model"] for c in configs if _agent_vision(c)}
        build_vision = effective_vision
        thinking_models = thinking_models or set()

        if images:
            if not effective_vision:
                await queue.put({
                    "type": "error",
                    "text": "No vision-capable agent in the roster. Remove the image, pick a "
                            "vision model, or toggle an agent's vision override on.",
                })
                return
            ignored = roster_models - effective_vision
            if ignored:
                await queue.put({
                    "type": "status", "stage": "vision",
                    "text": f"{len(ignored)} agent(s) without vision will ignore the image.",
                })
            # Sending think=True alongside an image suppresses image grounding on many
            # models, so disable thinking for the vision (image-carrying) models on this
            # debate. Normal (no-image) debates keep their thinking channel.
            disabled = thinking_models & effective_vision
            if disabled:
                thinking_models = thinking_models - effective_vision
                await queue.put({
                    "type": "status", "stage": "vision",
                    "text": f"Thinking disabled for {len(disabled)} vision agent(s) so the image is read correctly.",
                })

        # The Critic gets a clarification tool that can ask one specific agent a
        # direct question. The registry is filled after the roster is built (the
        # tool only runs later, mid-debate), so a late-bound dict is fine.
        registry: dict[str, AssistantAgent] = {}
        calls_left = {"n": 2}  # hard cap — "only if absolutely necessary"
        critic_cfg = next((c for c in configs if c.get("is_critic")), None)
        critic_name = safe_name(critic_cfg["name"]) if critic_cfg else ""

        async def ask_clarification(target_agent: str, question: str) -> str:
            """Ask ONE specific agent a direct clarifying question; return its answer."""
            if calls_left["n"] <= 0:
                return "Clarification limit reached — proceed with the information available."
            target = registry.get(target_agent) or registry.get(safe_name(target_agent))
            if target is None:
                return f"No agent named '{target_agent}'. Available: {', '.join(registry)}."
            if critic_name and target.name == critic_name:
                return "You cannot ask yourself; reason from the discussion instead."
            calls_left["n"] -= 1
            await queue.put({"type": "message", "agent": "Critic", "content": f"❓ **(to {target.name})** {question}"})
            result = await target.run(task=question, cancellation_token=token)
            answer = ""
            for m in result.messages:
                if isinstance(m, TextMessage) and m.source == target.name:
                    answer = m.content or ""
            answer = answer or "(no answer)"
            await queue.put({"type": "message", "agent": target.name, "content": answer})
            return answer

        clarification_tool = FunctionTool(
            ask_clarification,
            name="ask_clarification",
            description=(
                "Ask ONE specific agent a direct clarifying question. Args: target_agent "
                "(the agent's name) and question. Use only when something is genuinely unclear "
                "and blocks your judgment."
            ),
        )

        roster = build_roster(
            agents_cfg, num_ctx=num_ctx, consensus=True,
            vision_models=build_vision, thinking_models=thinking_models or set(),
            tools_models=tools_models or set(),
            clarification_tool=clarification_tool,
        )
        registry.update({a.name: a for a in roster})
        critic = registry.get(critic_name) or roster[-1]

        n = len(roster)
        # 1 round = n agent messages. DebateTermination decides the outcome; the
        # MaxMessageTermination (counting tool/summary messages too) is a hard backstop.
        # `unlimited` (max_rounds is None) disables the round cap — consensus, the
        # stuck-loop detector, the Stop button, or the 2000-message backstop end it.
        backstop = 2000 if max_rounds is None else max_rounds * n * 3 + 5
        termination = DebateTermination(n, max_rounds) | MaxMessageTermination(backstop)
        team = RoundRobinGroupChat(roster, termination_condition=termination, max_turns=backstop)
        limit_text = "no round limit" if max_rounds is None else f"max {max_rounds} rounds"
        await queue.put(
            {"type": "status", "stage": "debate",
             "text": f"Debating with {n} agents ({limit_text})…"}
        )

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        doc = session.context_block()
        doc_prefix = f"{doc}\n\n" if doc else ""
        # NB: keep the literal consensus token OUT of the task text — it lives in
        # each agent's system prompt. Putting it here could trip the consensus check
        # on the seed message and end the debate before anyone speaks.
        task_text = (
            f"Current date (UTC): {today}.\n\n"
            f"{doc_prefix}User question: {message}\n\n"
            "Collaborate to produce a single agreed answer. If the answer could depend on "
            "information that changes over time, do not rely on training knowledge that may "
            f"be stale — use web_research with the current date ({today}) or recency terms in "
            "the query to confirm the current facts. Signal agreement exactly as your "
            "instructions describe."
        )

        task: str | MultiModalMessage = task_text
        if images:
            content: list = [task_text]
            content.extend(AGImage.from_base64(_strip_data_url(b)) for b in images)
            task = MultiModalMessage(content=content, source="user")

        result: TaskResult | None = None
        completed_turns = 0  # one round = n agent turns (round-robin cadence)
        async for msg in team.run_stream(task=task, cancellation_token=token):
            if isinstance(msg, TaskResult):
                result = msg
                break
            # Real-time answer tokens + completed messages. (Thinking deltas are
            # pushed straight onto the queue by the streaming client.) The current
            # turn's round is completed_turns // n + 1.
            await _forward(msg, queue, round_no=completed_turns // n + 1)
            if isinstance(msg, TextMessage) and msg.source != "user":
                completed_turns += 1

        reason = (result.stop_reason if result else "") or ""

        # Either way, the Critic gets the last word: a detailed closing statement
        # streamed via a fresh run (the Critic still holds the full debate in its
        # model context from the round-robin). Shared local helper so both the
        # consensus and no-consensus branches reuse the same streaming loop.
        async def _run_closing(closing_task: str) -> None:
            async for cmsg in critic.run_stream(task=closing_task, cancellation_token=token):
                if isinstance(cmsg, TaskResult):
                    break
                await _forward(cmsg, queue)

        if REASON_CONSENSUS in reason:
            # Consensus: the Critic states the agreed answer, then elaborates.
            await queue.put({"type": "status", "stage": "debate",
                             "text": "Consensus reached — the Critic is writing the conclusion…"})
            consensus_task = (
                f"Current date (UTC): {today}. The group has reached unanimous consensus. As the "
                "Critic, give the FINAL conclusion in markdown: FIRST state the agreed answer to "
                "the user's question clearly and directly, THEN elaborate in detail — the key "
                "reasoning behind it, the supporting evidence and sources cited during the "
                "discussion, important caveats and assumptions, and any remaining limitations. "
                "Do not emit the consensus token."
            )
            await _run_closing(consensus_task)
            await queue.put({"type": "status", "stage": "consensus", "text": "Consensus reached ✓"})
        else:
            # No agreement: the Critic delivers a detailed closing statement.
            why = "the discussion kept repeating itself (deadlock)" if REASON_STUCK_LOOP in reason \
                else "the round limit was reached"
            await queue.put({"type": "status", "stage": "debate",
                             "text": "No agreement — the Critic is summarising why…"})
            closing_task = (
                f"Current date (UTC): {today}. The discussion ended without the group reaching "
                f"agreement ({why}). As the Critic, give the FINAL closing statement as markdown "
                "bullet points: one bullet per specific point of disagreement, each elaborated in "
                "detail — what each side held, why it blocked consensus, and (where possible) what "
                "evidence or decision would resolve it. Do not emit the consensus token."
            )
            await _run_closing(closing_task)
            await queue.put({"type": "status", "stage": "closed",
                             "text": "Debate closed by the Critic — no consensus."})

    except asyncio.CancelledError:
        raise
    except Exception as e:  # noqa: BLE001
        await queue.put({"type": "error", "text": str(e)})
    finally:
        await queue.put(_DONE)


def _record_event(event: dict, thinking: dict[str, str]) -> None:
    """Mirror a streamed event into the shared (server-side) conversation."""
    t = event.get("type")
    if t == "thinking_delta":
        thinking[event["agent"]] = thinking.get(event["agent"], "") + (event.get("content") or "")
    elif t == "message":
        item = {"kind": "agent", "agent": event["agent"], "content": event.get("content", "")}
        if thinking.get(event["agent"]):
            item["thinking"] = thinking.pop(event["agent"])
        if event.get("round") is not None:
            item["round"] = event["round"]
        session.append_item(item)
    elif t == "research":
        session.append_item({
            "kind": "research", "query": event.get("query", ""),
            "sources": event.get("sources", []), "screenshot": event.get("screenshot_b64", ""),
        })
    elif t == "status":
        session.set_status(event.get("text", ""))
        if event.get("stage") == "consensus":
            session.append_item({"kind": "consensus"})
        elif event.get("stage") == "closed":
            session.append_item({"kind": "closed"})
    elif t == "usage":
        session.add_usage(event["agent"], event.get("prompt_tokens", 0), event.get("completion_tokens", 0))
    elif t == "usage_thinking":
        session.add_thinking_usage(event["agent"], event.get("thinking_tokens", 0))
    elif t == "error":
        session.append_item({"kind": "error", "text": event.get("text", "")})


@app.get("/api/conversation")
async def get_conversation():
    """The shared conversation + live status (polled by every session)."""
    return session.snapshot()


@app.post("/api/conversation/clear")
async def clear_conversation():
    """Clear the shared conversation for everyone."""
    if session.busy:
        raise HTTPException(status_code=409, detail="A debate is running; stop it first.")
    session.clear_conversation()
    return {"status": "ok"}


@app.post("/api/chat")
async def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Empty message.")
    # One debate at a time — the conversation is shared across all sessions.
    if session.busy:
        raise HTTPException(status_code=409, detail="A debate is already running.")
    session.busy = True

    try:
        queue: asyncio.Queue = asyncio.Queue()
        token = CancellationToken()
        num_ctx = clamp_num_ctx(req.num_ctx)
        max_rounds = None if req.unlimited else clamp_max_turns(req.max_turns)
        agents_cfg = [a.model_dump() for a in req.agents] if req.agents else None
        # Clamp each agent's own context window (falling back to the request default)
        # and its sampling knobs.
        for a in agents_cfg or []:
            a["num_ctx"] = clamp_num_ctx(a.get("num_ctx") or num_ctx)
            clamp_sampling(a)

        # Record the user's message into the shared conversation immediately.
        session.append_item({
            "kind": "user", "text": req.message,
            **({"images": req.images} if req.images else {}),
        })
        session.set_status("Starting…")

        roster_models = [c["model"] for c in (agents_cfg or DEFAULT_AGENTS)]
        # Thinking capability is resolved every request (cheap + cached) so reasoning
        # models stream a separate thinking channel; vision only when an image is attached.
        thinking_models = await asyncio.to_thread(_thinking_models_sync, roster_models)
        vision_models: set[str] = set()
        if req.images:
            vision_models = await asyncio.to_thread(_vision_models_sync, roster_models)

        # Tool calling: an agent is handed a tool iff it does research, or it's the
        # Critic (which always receives the ask_clarification tool in _debate). Hard-fail
        # up front if such an agent's model doesn't advertise the Ollama `tools`
        # capability — otherwise it would crash mid-debate or hallucinate tool calls.
        tools_models = await asyncio.to_thread(_tools_models_sync, roster_models)
        offenders = [
            (c["name"], c["model"])
            for c in (agents_cfg or DEFAULT_AGENTS)
            if (c.get("with_research") or c.get("is_critic")) and c["model"] not in tools_models
        ]
        if offenders:
            raise HTTPException(
                status_code=400,
                detail=(
                    "These agents need tool calling but their model lacks the Ollama "
                    "`tools` capability — pick a tools-capable model: "
                    + ", ".join(f"{n} ({m})" for n, m in offenders)
                ),
            )

        # Set the sink BEFORE creating the task so the task (and the research tool +
        # streaming client running inside it) inherit this client's queue via contextvars.
        research_sink.set(queue)
        task = asyncio.create_task(
            _debate(
                req.message, agents_cfg, queue, token, num_ctx, max_rounds,
                req.images, vision_models, thinking_models, tools_models,
            )
        )
    except Exception:
        session.busy = False
        session.set_status("")
        raise

    async def event_stream():
        thinking: dict[str, str] = {}
        try:
            while True:
                event = await queue.get()
                if event is _DONE:
                    yield "data: {\"type\": \"done\"}\n\n"
                    break
                _record_event(event, thinking)  # mirror into the shared conversation
                yield f"data: {json.dumps(event)}\n\n"
        except asyncio.CancelledError:
            # Client disconnected mid-stream — cancel the in-flight debate.
            token.cancel()
            task.cancel()
            raise
        finally:
            session.busy = False
            session.set_status("")

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# Serve the built SPA if it exists (production / static-serve mode).
_DIST = os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")
if os.path.isdir(_DIST):
    app.mount("/", StaticFiles(directory=_DIST, html=True), name="spa")

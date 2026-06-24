"""Shared Ollama model clients and AutoGen agent factories.

This module is the single source of truth for the model configuration so that
both the CLI (`agent.py`) and the web app (`backend/main.py`) build agents the
same way. Agents are config-driven so the web UI can assemble an arbitrary
roster (different models, extra verifiers) for the consensus debate.
"""
import os
import re
from urllib.parse import urlparse

from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import ModelInfo
from autogen_core.tools import FunctionTool

from backend.research import web_research_tool
from backend.streaming_client import StreamingOllamaChatCompletionClient


def _normalize_ollama_host(raw: str | None) -> str:
    """Coerce an OLLAMA_HOST value into a valid client base URL.

    The same env var is often set to a *bind* address for the Ollama server
    (e.g. `0.0.0.0`, no scheme/port). Used verbatim as a connect URL that breaks
    raw urllib calls (`unknown url type`). Normalize: add scheme/port, and map the
    unroutable bind-any address to loopback.
    """
    raw = (raw or "").strip().rstrip("/")
    if not raw:
        return "http://localhost:11434"
    if "://" not in raw:
        raw = "http://" + raw
    parsed = urlparse(raw)
    host = parsed.hostname or "localhost"
    if host == "0.0.0.0":
        host = "127.0.0.1"
    port = parsed.port or 11434
    return f"{parsed.scheme}://{host}:{port}"


# Ollama host can be overridden via env var (e.g. when behind Tailscale). Normalized
# so a bare bind address like `0.0.0.0` still yields a usable client URL.
OLLAMA_HOST = _normalize_ollama_host(os.environ.get("OLLAMA_HOST"))

# Default context-window size (num_ctx). Bounding this caps the Ollama KV-cache
# memory and keeps long PDF + tool-call histories from being silently truncated.
# Overridable via env; the web UI can also set it per request.
DEFAULT_NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", "32768"))

# Appended to each agent's system prompt during a consensus debate. The debate
# runs in turn-based rounds (each agent speaks once per round, in order) and ends
# only when EVERY agent agrees in the same round. Agents emit this exact token,
# on its own line, when they personally agree — see backend/termination.py.
CONSENSUS_RULE = (
    "You are one of several expert agents collaborating in a turn-by-turn round-table "
    "discussion. Each round, every agent speaks once, in order. On your turn, read all "
    "prior messages, then build on or critique the current proposal and add your expertise. "
    "The discussion continues round after round and ends only when EVERY agent agrees. When, "
    "and only when, you personally judge that the group has reached a correct, well-supported "
    "conclusion and you have nothing further to add, reply with exactly the single token "
    "CONSENSUS (uppercase, on its own line) and nothing else. If you still have any reservation, "
    "do NOT emit the token — briefly explain your remaining concern instead."
)

# Appended (after the critique directive) only to the protected Critic. The Critic
# may ask one targeted question via the ask_clarification tool when genuinely stuck.
CRITIC_CLARIFY_RULE = (
    "\n\nIf something is genuinely unclear and blocks your judgment, you may call the "
    "ask_clarification tool to ask ONE specific agent a direct question — but only if "
    "absolutely necessary. Prefer reasoning from the discussion over asking."
)

# These models are function-calling + json capable; family "unknown" keeps the
# native Ollama client from making provider-specific assumptions. We keep a
# vision and a non-vision variant: the `vision` flag tells AutoGen whether image
# content may be sent to this client (only set it for models that support it).
def _model_info(vision: bool = False) -> ModelInfo:
    return ModelInfo(
        vision=vision,
        function_calling=True,
        json_output=True,
        family="unknown",
    )


# Seed roster mirroring the original three-role pipeline. The literature agent
# gets the web_research (headless Playwright) tool. The Critic critiques ALL other
# agents by default (empty/unset `critiques` ⇒ everyone); it is the protected critic.
DEFAULT_AGENTS: list[dict] = [
    {
        "name": "LiteratureAgent",
        "model": "qwen3.5:latest",
        "with_research": True,
        "system_message": (
            "You gather and synthesize key facts and evidence relevant to the user's "
            "question, from the provided document and from the literature. When recent or "
            "external information would help, call the web_research tool with a focused query, "
            "then give a concise, factual summary grounded in what you found. Always cite "
            "source titles/URLs you used."
        ),
    },
    {
        "name": "HypothesisAgent",
        "model": "gemma4:latest",
        "with_research": False,
        "system_message": "You generate actionable, testable ideas and proposals that address the user's question.",
    },
    {
        "name": "Critic",
        "model": "gpt-oss:latest",
        "with_research": False,
        "is_critic": True,
        "critiques": [],  # empty ⇒ critique ALL other agents (default)
        "system_message": "You critique the proposals based strictly on the established facts.",
    },
]


def critique_directive(targets: list[str] | None) -> str:
    """Directive appended to a critic's prompt so it focuses on specific agents."""
    names = [t for t in (targets or []) if t]
    if not names:
        return ""
    joined = ", ".join(names)
    return (
        f"\n\nSpecifically critique the contributions of: {joined}. Focus your critique "
        "on their reasoning, claims, and any unsupported assumptions."
    )


def _client(
    model: str,
    num_ctx: int,
    vision: bool = False,
    agent_name: str = "",
    enable_thinking: bool = False,
) -> StreamingOllamaChatCompletionClient:
    # num_ctx is passed inside `options`, matching the Ollama /api/chat schema
    # (https://docs.ollama.com/). Constructing a client is cheap (no connection),
    # so we build per request to honour the caller's num_ctx. The streaming client
    # surfaces token deltas and (for thinking models) a separate reasoning channel.
    return StreamingOllamaChatCompletionClient(
        model=model,
        host=OLLAMA_HOST,
        model_info=_model_info(vision),
        options={"num_ctx": num_ctx},
        agent_name=agent_name,
        enable_thinking=enable_thinking,
    )


def safe_name(name: str) -> str:
    """AutoGen agent names must be valid identifiers; sanitize UI-supplied names."""
    cleaned = re.sub(r"\W+", "_", name).strip("_")
    if not cleaned or not cleaned[0].isalpha():
        cleaned = "Agent_" + cleaned
    return cleaned


def build_agent(
    name: str,
    model: str,
    system_message: str,
    num_ctx: int = DEFAULT_NUM_CTX,
    with_research: bool = False,
    consensus: bool = False,
    vision: bool = False,
    enable_thinking: bool = False,
    critiques: list[str] | None = None,
    is_critic: bool = False,
    clarification_tool=None,
) -> AssistantAgent:
    """Generic agent factory used by both the CLI and the consensus debate."""
    sys = system_message + critique_directive(critiques)
    if is_critic and clarification_tool is not None:
        sys += CRITIC_CLARIFY_RULE
    sys += f"\n\n{CONSENSUS_RULE}" if consensus else ""
    safe = safe_name(name)

    tools = []
    if with_research:
        tools.append(web_research_tool)
    if is_critic and clarification_tool is not None:
        tools.append(clarification_tool)

    return AssistantAgent(
        name=safe,
        model_client=_client(model, num_ctx, vision=vision, agent_name=safe, enable_thinking=enable_thinking),
        tools=tools,
        reflect_on_tool_use=bool(tools),  # reflect when any tool (research / clarify) is present
        model_client_stream=True,  # surface token-by-token deltas to the UI
        system_message=sys,
    )


def build_roster(
    agents: list[dict] | None = None,
    num_ctx: int = DEFAULT_NUM_CTX,
    consensus: bool = True,
    vision_models: set[str] | None = None,
    thinking_models: set[str] | None = None,
    clarification_tool=None,
) -> list[AssistantAgent]:
    """Build the list of agents for the debate from config (or the defaults).

    `num_ctx` is the fallback context window; each config may override it with its
    own `num_ctx`. `vision_models` is the set of model names known to support image
    input. `thinking_models` is the set known to support reasoning — those agents get
    `think=True` and stream a separate thinking channel.
    """
    configs = agents if agents else DEFAULT_AGENTS
    vision_models = vision_models or set()
    thinking_models = thinking_models or set()
    all_names = [safe_name(c["name"]) for c in configs]

    def _critic_targets(c: dict) -> list[str] | None:
        """A critic with no explicit targets critiques ALL other agents (default)."""
        if not c.get("is_critic"):
            return None
        chosen = [safe_name(t) for t in (c.get("critiques") or []) if t]
        if chosen:
            return chosen
        self_name = safe_name(c["name"])
        return [n for n in all_names if n != self_name]

    return [
        build_agent(
            name=c["name"],
            model=c["model"],
            system_message=c.get("system_message", ""),
            num_ctx=int(c.get("num_ctx") or num_ctx),
            with_research=bool(c.get("with_research", False)),
            consensus=consensus,
            vision=c["model"] in vision_models,
            enable_thinking=c["model"] in thinking_models,
            critiques=_critic_targets(c),
            is_critic=bool(c.get("is_critic")),
            clarification_tool=clarification_tool,
        )
        for c in configs
    ]


# --- Thin wrappers for the CLI (agent.py), preserving its one-shot behaviour --- #
def build_literature_agent(
    tools: list[FunctionTool] | None = None, num_ctx: int = DEFAULT_NUM_CTX
) -> AssistantAgent:
    c = DEFAULT_AGENTS[0]
    return build_agent(c["name"], c["model"], c["system_message"], num_ctx, with_research=True)


def build_hypothesis_agent(num_ctx: int = DEFAULT_NUM_CTX) -> AssistantAgent:
    c = DEFAULT_AGENTS[1]
    return build_agent(c["name"], c["model"], c["system_message"], num_ctx)


def build_critic_agent(num_ctx: int = DEFAULT_NUM_CTX) -> AssistantAgent:
    c = DEFAULT_AGENTS[2]
    return build_agent(c["name"], c["model"], c["system_message"], num_ctx)

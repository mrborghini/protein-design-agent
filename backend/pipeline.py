"""Pipeline mode: a Professor orchestrating a Paper Analyst + a Bio-App Operator.

A different topology from the consensus debate (backend/main.py's RoundRobinGroupChat):
here a ``SelectorGroupChat`` lets an LLM selector (the Professor's model) pick who
acts next, so the Professor drives and delegates concrete sub-tasks:

  - **Professor** (orchestrator): plans, interprets results, decides the next step,
    and writes the final answer. Holds no tools — it reasons and directs.
  - **Paper Analyst**: local-PDF RAG (``paper_search``) for mutations, Tm, domains.
  - **Bio-App Operator**: runs the four structural-biology tools (Boltz-2,
    RFdiffusion, ProteinMPNN, PyRosetta) — the only agent that touches the GPU.
  - **Critic**: NOT a chat turn-taker — built here and wired (via
    ``taskity.set_reviewer``) as a fail-closed gate that must APPROVE any
    Operator-written taskity script before it is allowed to execute.

Termination: the Professor emits ``DESIGN_COMPLETE`` when the goal is met or further
progress is blocked; a message backstop is the hard cap.

POC: model tool-calling is imperfect on small local models, so the selector is
prompted to keep the Professor in the loop and route narrowly. GPU tools pause the
session to evict the LLM from VRAM (see backend/gpu.py).
"""
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.teams import SelectorGroupChat

from backend.agents import DEFAULT_NUM_CTX, _client, build_agent
from backend.bioapps.boltz import boltz_mutagenesis_tool, boltz_predict_tool
from backend.bioapps.proteinmpnn import proteinmpnn_design_tool
from backend.bioapps.pyrosetta import pyrosetta_score_tool
from backend.bioapps.rfdiffusion import rfdiffusion_generate_tool
from backend.bioapps.taskity import TASKITY_TOOLS, set_reviewer
from backend.rag import paper_search_tool
from backend.research import research_sink

# The Professor emits this on its own when the design goal is met or blocked.
DONE_TOKEN = "DESIGN_COMPLETE"

OPERATOR_TOOLS = [
    boltz_predict_tool, boltz_mutagenesis_tool, rfdiffusion_generate_tool,
    proteinmpnn_design_tool, pyrosetta_score_tool,
    # Plus the taskity tools: the Operator can author/run its own Python scripts in venvs.
    *TASKITY_TOOLS,
]
ANALYST_TOOLS = [paper_search_tool]

_GPU_NOTE = (
    "Note: structural-biology tools run on the GPU, which forces the language models out of VRAM "
    "first, so each tool call pauses the session and then the models reload — expect delays; this "
    "is expected, not a failure."
)

# Default three-role roster. Models mirror DEFAULT_AGENTS' :latest tags; the UI/config
# can override. `role` keys the system prompt + tool set in build_pipeline.
DEFAULT_PIPELINE_AGENTS: list[dict] = [
    {
        "name": "Professor",
        "model": "gpt-oss:latest",
        "role": "professor",
        "temperature": 0.4,
        "system_message": (
            "You are the Professor — a structural-biology principal investigator orchestrating a "
            "protein-design session studying neurodegenerative protein misfolding. You do NOT run "
            "tools yourself. Break the user's goal into steps, delegate: ask the Paper Analyst for "
            "literature/experimental facts (mutations, melting temperatures, domains) and instruct "
            "the Bio-App Operator to run specific tools (Boltz-2 to model/mutate a structure, "
            "RFdiffusion to design a binder backbone against an exposed hydrophobic patch, "
            "ProteinMPNN to assign sequences, PyRosetta to score ΔG). Interpret every result "
            "critically, never fabricate numbers, and state assumptions and limitations. When the "
            "goal is achieved or genuinely blocked, write a clear final summary and then output "
            f"{DONE_TOKEN} on its own line. {_GPU_NOTE}"
        ),
    },
    {
        "name": "PaperAnalyst",
        "model": "gemma4:latest",
        "role": "analyst",
        "temperature": 0.3,
        "system_message": (
            "You are the Paper Analyst. When the Professor needs facts grounded in the local "
            "literature, call the paper_search tool and report exactly what the passages say — "
            "specific mutations, melting temperatures (Tm), structural domains, experimental "
            "conditions — always citing the source filename. Do not invent findings; if the "
            "papers don't cover it, say so."
        ),
    },
    {
        "name": "BioAppOperator",
        "model": "qwen3.5:latest",
        "role": "operator",
        "temperature": 0.2,
        "system_message": (
            "You are the Bio-App Operator. You execute the structural-biology tools the Professor "
            "requests, with exactly the parameters given, and report back the concrete outputs: "
            "file paths, mean pLDDT, ΔG/REU scores, and counts. Run one tool at a time and wait "
            "for its result. Never fabricate results or paths — only report what a tool actually "
            f"returned; if a tool errors, report the error verbatim. {_GPU_NOTE}\n\n"
            "For the common operations, prefer the typed tools (boltz_predict, boltz_mutagenesis, "
            "rfdiffusion_generate, proteinmpnn_design, pyrosetta_score). For anything they don't "
            "cover, you can WRITE AND RUN YOUR OWN PYTHON via taskities (taskity_create then "
            "taskity_run; also taskity_update/view/list/delete). A taskity is a reusable script: it "
            "must read params with json.loads(sys.argv[1]) and print exactly one final JSON line "
            "with print(json.dumps(...)); write large output files into the directory in the "
            "BIOAPP_OUT environment variable. To use a heavy library, set base_venv to a bio-app "
            "env: 'pyrosetta' (import pyrosetta), 'boltz', 'rfdiffusion', or 'proteinmpnn' (import "
            "torch); otherwise leave base_venv empty and list any pip packages you need. Write real, "
            "working code — no placeholders, TODOs, mock/sample data, or hardcoded fake values "
            "(such scripts are rejected). Reuse a taskity across params instead of recreating it.\n\n"
            "IMPORTANT: every taskity script is reviewed by the Critic and must be APPROVED before it "
            "will run. If taskity_run comes back blocked/not-approved, read the Critic's feedback, fix "
            "the script with taskity_update, and run again."
        ),
    },
    {
        "name": "Critic",
        "model": "gpt-oss:latest",
        "role": "critic",
        "temperature": 0.2,
        "system_message": (
            "You are the Critic — a meticulous senior code reviewer and the safety gate for code "
            "execution. The Bio-App Operator writes Python scripts (taskities); none may run until "
            "you approve them. Judge each script for: correctness (does it actually do what's "
            "described, with no stubs/placeholders/mock or hardcoded fake data?), safety (no "
            "destructive, dangerous, or clearly out-of-scope operations), and contract adherence "
            "(reads json.loads(sys.argv[1]); prints exactly one final JSON line). Be strict but "
            "fair: approve genuinely correct, safe scripts; reject anything dubious with specific, "
            "actionable reasons."
        ),
    },
]

_SELECTOR_PROMPT = (
    "You are coordinating a protein-design session. Roles:\n{roles}\n\n"
    "Conversation so far:\n{history}\n\n"
    "From {participants}, select the next single participant to speak. Guidance: the Professor "
    "drives — planning, interpreting results, and deciding the next step. Route to PaperAnalyst "
    "only when literature/experimental facts are needed, and to BioAppOperator only when a "
    "structural-biology tool must actually be run; then return to the Professor. Reply with only "
    "the participant's name."
)


_REVIEW_PROMPT = (
    "Review this Python script the Bio-App Operator wants to run. Purpose: {description}\n\n"
    "```python\n{script}\n```\n\n"
    "Decide whether it is correct, safe, and contract-compliant to EXECUTE on this machine. "
    "Reply with `APPROVE` or `REJECT` as the very first word on the first line, then give your "
    "reasons. If anything is dubious, REJECT with specific, actionable fixes."
)


def _parse_verdict(text: str) -> bool:
    """True only if the Critic's reply clearly starts with APPROVE (fail-closed otherwise)."""
    for line in text.splitlines():
        s = line.strip().lstrip("#*->` ").upper()
        if s:
            return s.startswith("APPROVE") and not s.startswith("APPROVE NOT")
    return False


def _role_cfg(configs: list[dict], role: str) -> dict | None:
    return next((c for c in configs if c.get("role") == role), None)


# Default Critic config, injected when a caller's roster omits the critic role so the
# review gate is always present (defense in depth — the gate can't be disabled by the client).
_DEFAULT_CRITIC_CFG = next(c for c in DEFAULT_PIPELINE_AGENTS if c.get("role") == "critic")


def _make_reviewer(critic: AssistantAgent):
    """Build the async reviewer the taskity gate calls: run the Critic, parse APPROVE/REJECT.

    Streams a `Critic` message for the review start and the verdict so the user sees the
    gate working. Fail-closed: any error ⇒ not approved.
    """
    async def _emit(text: str) -> None:
        queue = research_sink.get(None)
        if queue is not None:
            await queue.put({"type": "message", "agent": "Critic", "content": text})

    async def reviewer(name: str, description: str, script: str) -> tuple[bool, str]:
        await _emit(f"🔍 Reviewing taskity **{name}** before it may run…")
        try:
            result = await critic.run(task=_REVIEW_PROMPT.format(description=description, script=script))
        except Exception as e:  # noqa: BLE001 - fail closed
            await _emit(f"⚠️ Review failed, blocking execution: {e}")
            return False, f"review error: {e}"
        text = ""
        for m in result.messages:
            if isinstance(m, TextMessage) and m.source == critic.name:
                text = m.content or ""
        approved = _parse_verdict(text)
        await _emit(f"{'✅ APPROVED' if approved else '❌ REJECTED'} taskity **{name}** — {text}")
        return approved, text

    return reviewer


def build_pipeline(
    agents_cfg: list[dict] | None = None,
    num_ctx: int = DEFAULT_NUM_CTX,
    thinking_models: set[str] | None = None,
    tools_models: set[str] | None = None,
    max_messages: int = 60,
) -> tuple[SelectorGroupChat, dict[str, AssistantAgent]]:
    """Build the Professor/Analyst/Operator team as a SelectorGroupChat.

    `tools_models` is the set of roster models advertising the Ollama `tools`
    capability; the Analyst/Operator must be in it (the caller hard-fails otherwise,
    mirroring backend/main.py's gate). The Professor needs no tools.
    """
    configs = list(agents_cfg if agents_cfg else DEFAULT_PIPELINE_AGENTS)
    thinking_models = thinking_models or set()
    tools_models = tools_models or set()

    # Always have a Critic so the review gate exists even if the client omitted it.
    if not any(c.get("role") == "critic" for c in configs):
        configs = configs + [_DEFAULT_CRITIC_CFG]

    role_tools = {"analyst": ANALYST_TOOLS, "operator": OPERATOR_TOOLS, "professor": [], "critic": []}

    def _build(c: dict) -> AssistantAgent:
        return build_agent(
            name=c["name"],
            model=c["model"],
            system_message=c.get("system_message", ""),
            num_ctx=int(c.get("num_ctx") or num_ctx),
            consensus=False,  # pipeline is orchestrated, not a consensus debate
            enable_thinking=c["model"] in thinking_models,
            tools_capable=c["model"] in tools_models,
            extra_tools=role_tools.get(c.get("role", ""), []),
            sampling={"temperature": c.get("temperature")},
        )

    # The Critic is a gate, not a chat turn-taker: built but kept OUT of the team.
    team_agents: dict[str, AssistantAgent] = {}
    critic: AssistantAgent | None = None
    for c in configs:
        agent = _build(c)
        if c.get("role") == "critic":
            critic = agent
        else:
            team_agents[agent.name] = agent

    # Install the code-review gate the taskity runner enforces before executing scripts.
    set_reviewer(_make_reviewer(critic) if critic is not None else None)

    # The selector runs on the Professor's model (the orchestrator's judgment).
    prof = _role_cfg(configs, "professor") or configs[0]
    selector_client = _client(prof["model"], num_ctx, agent_name="selector")

    termination = TextMentionTermination(DONE_TOKEN) | MaxMessageTermination(max_messages)
    team = SelectorGroupChat(
        list(team_agents.values()),
        model_client=selector_client,
        termination_condition=termination,
        selector_prompt=_SELECTOR_PROMPT,
        allow_repeated_speaker=True,  # let the Operator chain tool calls before handing back
    )
    return team, team_agents

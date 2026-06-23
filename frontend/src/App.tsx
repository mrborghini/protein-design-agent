import { useEffect, useRef, useState } from "react";
import Chat, { ChatItem } from "./components/Chat";
import PdfUpload from "./components/PdfUpload";
import ContextControl from "./components/ContextControl";
import AgentRoster, { ModelCaps } from "./components/AgentRoster";
import { streamChat, AgentConfig } from "./lib/sse";
import { useDarkMode } from "./lib/theme";
import { downloadJSON, downloadText, formatTokens } from "./lib/download";

const FALLBACK = { numCtx: 32768, ctxMin: 512, ctxMax: 262144, maxTurns: 12, turnsMin: 2, turnsMax: 40 };
const ROSTER_KEY = "pda-roster";
const TURNS_KEY = "pda-maxturns";
const CTX_KEY = "pda-numctx";
const CONV_KEY = "pda-conversation";

type Usage = Record<string, { prompt: number; completion: number }>;

/** Fill in any missing AgentConfig fields with sensible defaults. */
function normalizeAgent(a: Partial<AgentConfig>, defaultNumCtx: number): AgentConfig {
  return {
    name: a.name ?? "Agent",
    model: a.model ?? "",
    system_message: a.system_message ?? "",
    with_research: !!a.with_research,
    num_ctx: a.num_ctx ?? defaultNumCtx,
    is_critic: !!a.is_critic,
    critiques: a.critiques ?? undefined,
  };
}

function conversationToMarkdown(items: ChatItem[]): string {
  const lines: string[] = ["# Protein Design Agent — conversation\n"];
  for (const it of items) {
    if (it.kind === "user") lines.push(`## You\n\n${it.text}\n`);
    else if (it.kind === "agent") lines.push(`### ${it.agent}\n\n${it.content}\n`);
    else if (it.kind === "research")
      lines.push(`> 🔎 Research: ${it.query}\n>\n` + it.sources.map((s) => `> - [${s.title || s.url}](${s.url})`).join("\n") + "\n");
    else if (it.kind === "consensus") lines.push(`_Consensus reached ✓_\n`);
    else if (it.kind === "error") lines.push(`> ⚠️ ${it.text}\n`);
  }
  return lines.join("\n");
}

export default function App() {
  const [items, setItems] = useState<ChatItem[]>(() => {
    try {
      const saved = localStorage.getItem(CONV_KEY);
      return saved ? (JSON.parse(saved) as ChatItem[]) : [];
    } catch {
      return [];
    }
  });
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState("");
  const [pdfName, setPdfName] = useState("");
  const [dark, toggleDark] = useDarkMode();
  const [usage, setUsage] = useState<Usage>({});

  const [numCtx, setNumCtx] = useState(() => Number(localStorage.getItem(CTX_KEY)) || FALLBACK.numCtx);
  const [ctxBounds, setCtxBounds] = useState({ min: FALLBACK.ctxMin, max: FALLBACK.ctxMax });
  const [maxTurns, setMaxTurns] = useState(() => Number(localStorage.getItem(TURNS_KEY)) || FALLBACK.maxTurns);
  const [turnsBounds, setTurnsBounds] = useState({ min: FALLBACK.turnsMin, max: FALLBACK.turnsMax });

  const [models, setModels] = useState<string[]>([]);
  const [caps, setCaps] = useState<ModelCaps>({});
  const [defaults, setDefaults] = useState<AgentConfig[]>([]);
  const [roster, setRoster] = useState<AgentConfig[]>(() => {
    try {
      const saved = localStorage.getItem(ROSTER_KEY);
      return saved ? (JSON.parse(saved) as AgentConfig[]) : [];
    } catch {
      return [];
    }
  });
  const [showRoster, setShowRoster] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  // Pull config defaults + available models from the backend.
  useEffect(() => {
    fetch("/api/health")
      .then((r) => r.json())
      .then((h) => {
        if (h.num_ctx_min && h.num_ctx_max) setCtxBounds({ min: h.num_ctx_min, max: h.num_ctx_max });
        if (h.max_turns_min && h.max_turns_max) setTurnsBounds({ min: h.max_turns_min, max: h.max_turns_max });
        const seedCtx = Number(localStorage.getItem(CTX_KEY)) || h.default_num_ctx || FALLBACK.numCtx;
        const defs: AgentConfig[] = (h.default_agents ?? []).map((a: Partial<AgentConfig>) =>
          normalizeAgent(a, seedCtx)
        );
        setDefaults(defs);
        // Seed the roster from defaults only if the user has none saved.
        setRoster((cur) => (cur.length ? cur : defs));
      })
      .catch(() => {});
    fetch("/api/models")
      .then((r) => r.json())
      .then((d) => setModels(d.models ?? []))
      .catch(() => {});
    fetch("/api/models/capabilities")
      .then((r) => r.json())
      .then((d) => setCaps(d.capabilities ?? {}))
      .catch(() => {});
  }, []);

  useEffect(() => {
    if (roster.length) localStorage.setItem(ROSTER_KEY, JSON.stringify(roster));
  }, [roster]);
  useEffect(() => {
    localStorage.setItem(TURNS_KEY, String(maxTurns));
  }, [maxTurns]);
  useEffect(() => {
    localStorage.setItem(CTX_KEY, String(numCtx));
  }, [numCtx]);
  useEffect(() => {
    localStorage.setItem(CONV_KEY, JSON.stringify(items));
  }, [items]);

  function push(item: ChatItem) {
    setItems((prev) => [...prev, item]);
  }

  function addUsage(agent: string, prompt: number, completion: number) {
    setUsage((prev) => {
      const cur = prev[agent] ?? { prompt: 0, completion: 0 };
      return { ...prev, [agent]: { prompt: cur.prompt + prompt, completion: cur.completion + completion } };
    });
  }

  // Reset the roster to the three default agents on the first 3 installed models.
  function resetRoster() {
    if (!defaults.length) return;
    const next = defaults.map((d, i) => ({ ...d, model: models[i] ?? models[0] ?? d.model, num_ctx: numCtx }));
    setRoster(next);
  }

  function exportConfig() {
    downloadJSON("protein-agent-config.json", { version: 1, defaultNumCtx: numCtx, maxTurns, roster });
  }

  function importConfig(raw: string): string | null {
    try {
      const obj = JSON.parse(raw);
      const list = Array.isArray(obj) ? obj : obj.roster;
      if (!Array.isArray(list) || !list.every((a) => a && a.name && a.model)) {
        return "Invalid config: expected a roster of agents with name and model.";
      }
      const ctx = Number(obj.defaultNumCtx) || numCtx;
      setRoster(list.map((a: Partial<AgentConfig>) => normalizeAgent(a, ctx)));
      if (obj.defaultNumCtx) setNumCtx(Math.max(ctxBounds.min, Math.min(ctxBounds.max, ctx)));
      if (obj.maxTurns) setMaxTurns(Math.max(turnsBounds.min, Math.min(turnsBounds.max, Number(obj.maxTurns))));
      return null;
    } catch {
      return "Could not parse JSON.";
    }
  }

  function clearConversation() {
    setItems([]);
    setUsage({});
    localStorage.removeItem(CONV_KEY);
  }

  async function handleSend(text: string, images: string[]) {
    push({ kind: "user", text, images: images.length ? images : undefined });
    setBusy(true);
    setStatus("Starting…");
    const controller = new AbortController();
    abortRef.current = controller;

    try {
      await streamChat(
        text,
        { numCtx, maxTurns, agents: roster, images },
        (e) => {
          if (e.type === "status") {
            setStatus(e.text);
            if (e.stage === "consensus") push({ kind: "consensus" });
          } else if (e.type === "research")
            push({ kind: "research", query: e.query, sources: e.sources, screenshot: e.screenshot_b64 });
          else if (e.type === "message") push({ kind: "agent", agent: e.agent, content: e.content });
          else if (e.type === "usage") addUsage(e.agent, e.prompt_tokens, e.completion_tokens);
          else if (e.type === "error") push({ kind: "error", text: e.text });
        },
        controller.signal
      );
    } catch (err) {
      if (!(err instanceof DOMException && err.name === "AbortError")) {
        push({ kind: "error", text: err instanceof Error ? err.message : "Request failed" });
      }
    } finally {
      setBusy(false);
      setStatus("");
      abortRef.current = null;
    }
  }

  const totalGenerated = Object.values(usage).reduce((s, u) => s + u.completion, 0);
  const totalAll = Object.values(usage).reduce((s, u) => s + u.prompt + u.completion, 0);

  return (
    <div className="flex h-full text-slate-900 dark:text-slate-100">
      <aside className="flex w-72 flex-col gap-4 overflow-y-auto border-r border-slate-200 bg-white px-4 py-6 dark:border-slate-700 dark:bg-slate-800">
        <div className="flex items-start justify-between">
          <div>
            <h1 className="text-lg font-semibold text-slate-800 dark:text-slate-100">Protein Design Agent</h1>
            <p className="mt-1 text-xs text-slate-400 dark:text-slate-500">Local · Ollama · Playwright</p>
          </div>
          <button
            onClick={toggleDark}
            title="Toggle dark mode"
            className="rounded-lg border border-slate-200 px-2 py-1 text-sm dark:border-slate-600"
          >
            {dark ? "☀️" : "🌙"}
          </button>
        </div>

        <PdfUpload onLoaded={(info) => setPdfName(info.name)} />

        <ContextControl
          value={numCtx}
          min={ctxBounds.min}
          max={ctxBounds.max}
          disabled={busy}
          onChange={setNumCtx}
        />

        <div>
          <label className="text-xs font-medium text-slate-500 dark:text-slate-400">Max debate turns</label>
          <input
            type="number"
            value={maxTurns}
            min={turnsBounds.min}
            max={turnsBounds.max}
            disabled={busy}
            onChange={(e) => setMaxTurns(Number(e.target.value))}
            onBlur={(e) =>
              setMaxTurns(Math.max(turnsBounds.min, Math.min(turnsBounds.max, Number(e.target.value) || turnsBounds.min)))
            }
            className="mt-1 w-full rounded-lg border border-slate-300 px-2 py-1.5 text-sm outline-none focus:border-slate-500 disabled:opacity-50 dark:border-slate-600 dark:bg-slate-900 dark:text-slate-100"
          />
        </div>

        <button
          onClick={() => setShowRoster(true)}
          className="rounded-lg border border-slate-300 px-3 py-2 text-sm text-slate-700 hover:border-slate-400 dark:border-slate-600 dark:text-slate-200"
        >
          Configure agents ({roster.length})
        </button>

        <div className="flex gap-2">
          <button
            onClick={clearConversation}
            disabled={items.length === 0}
            className="flex-1 rounded-lg border border-slate-300 px-2 py-1.5 text-xs text-slate-600 hover:border-slate-400 disabled:opacity-40 dark:border-slate-600 dark:text-slate-300"
          >
            Clear chat
          </button>
          <button
            onClick={() => downloadText("conversation.md", conversationToMarkdown(items), "text/markdown")}
            disabled={items.length === 0}
            className="flex-1 rounded-lg border border-slate-300 px-2 py-1.5 text-xs text-slate-600 hover:border-slate-400 disabled:opacity-40 dark:border-slate-600 dark:text-slate-300"
          >
            Download chat
          </button>
        </div>

        {/* Token usage */}
        {Object.keys(usage).length > 0 && (
          <div className="text-xs">
            <p className="font-medium text-slate-500 dark:text-slate-400">Tokens generated</p>
            <div className="mt-1 space-y-0.5 text-slate-500 dark:text-slate-400">
              {Object.entries(usage).map(([agent, u]) => (
                <p key={agent} className="flex justify-between">
                  <span className="truncate">{agent}</span>
                  <span className="font-medium text-slate-700 dark:text-slate-200">{formatTokens(u.completion)}</span>
                </p>
              ))}
              <p className="flex justify-between border-t border-slate-200 pt-0.5 dark:border-slate-700">
                <span>Total generated</span>
                <span className="font-semibold text-slate-800 dark:text-slate-100">{formatTokens(totalGenerated)}</span>
              </p>
              <p className="flex justify-between text-slate-400 dark:text-slate-500">
                <span>Total (incl. prompt)</span>
                <span>{formatTokens(totalAll)}</span>
              </p>
            </div>
          </div>
        )}

        <div className="mt-auto space-y-1 text-xs text-slate-400 dark:text-slate-500">
          <p className="font-medium text-slate-500 dark:text-slate-400">Roster</p>
          {roster.map((a, i) => (
            <p key={i}>
              {i + 1} · {a.name} <span className="text-slate-300 dark:text-slate-600">— {a.model}</span>
              {a.with_research ? " 🔎" : ""}
              {caps[a.model]?.vision ? " 👁" : ""}
            </p>
          ))}
          {pdfName && <p className="pt-2 text-emerald-600 dark:text-emerald-400">Grounded on {pdfName}</p>}
        </div>
      </aside>

      <main className="flex-1 bg-slate-50 dark:bg-slate-900">
        <Chat items={items} busy={busy} status={status} onSend={handleSend} />
      </main>

      {showRoster && (
        <AgentRoster
          agents={roster}
          models={models}
          caps={caps}
          defaultNumCtx={numCtx}
          ctxMin={ctxBounds.min}
          ctxMax={ctxBounds.max}
          onChange={setRoster}
          onClose={() => setShowRoster(false)}
          onReset={resetRoster}
          onExport={exportConfig}
          onImport={importConfig}
        />
      )}
    </div>
  );
}

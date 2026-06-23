import { useEffect, useRef, useState } from "react";
import Chat, { ChatItem } from "./components/Chat";
import PdfUpload from "./components/PdfUpload";
import ContextControl from "./components/ContextControl";
import AgentRoster, { ModelCaps } from "./components/AgentRoster";
import { streamChat, AgentConfig } from "./lib/sse";
import { useDarkMode } from "./lib/theme";
import { downloadJSON, downloadText, formatTokens } from "./lib/download";

const FALLBACK = { numCtx: 32768, ctxMin: 512, ctxMax: 262144, maxTurns: 20, turnsMin: 1, turnsMax: 100 };
const ROSTER_KEY = "pda-roster";
const TURNS_KEY = "pda-maxturns";
const CTX_KEY = "pda-numctx";
const NOLIMIT_KEY = "pda-nolimit";
const POLL_MS = 2000;

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
    else if (it.kind === "agent") {
      let s = `### ${it.agent}\n`;
      if (it.thinking && it.thinking.trim())
        s += `\n<details><summary>💭 Thinking</summary>\n\n${it.thinking}\n\n</details>\n`;
      lines.push(`${s}\n${it.content}\n`);
    }
    else if (it.kind === "research")
      lines.push(`> 🔎 Research: ${it.query}\n>\n` + it.sources.map((s) => `> - [${s.title || s.url}](${s.url})`).join("\n") + "\n");
    else if (it.kind === "consensus") lines.push(`_Consensus reached ✓_\n`);
    else if (it.kind === "closed") lines.push(`_Debate closed by the Critic — no consensus_\n`);
    else if (it.kind === "error") lines.push(`> ⚠️ ${it.text}\n`);
  }
  return lines.join("\n");
}

export default function App() {
  // The conversation lives on the backend (shared across all sessions); we never
  // persist it locally — it is fetched/polled from /api/conversation.
  const [items, setItems] = useState<ChatItem[]>([]);
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState("");
  const [pdfName, setPdfName] = useState("");
  const [dark, toggleDark] = useDarkMode();
  const [usage, setUsage] = useState<Usage>({});

  const [numCtx, setNumCtx] = useState(() => Number(localStorage.getItem(CTX_KEY)) || FALLBACK.numCtx);
  const [ctxBounds, setCtxBounds] = useState({ min: FALLBACK.ctxMin, max: FALLBACK.ctxMax });
  const [maxTurns, setMaxTurns] = useState(() => Number(localStorage.getItem(TURNS_KEY)) || FALLBACK.maxTurns);
  const [noLimit, setNoLimit] = useState(() => localStorage.getItem(NOLIMIT_KEY) === "1");
  const [turnsBounds, setTurnsBounds] = useState({ min: FALLBACK.turnsMin, max: FALLBACK.turnsMax });
  // True only while THIS client is actively streaming a debate it started; used to
  // avoid clobbering the live token stream with the (finalized-only) poll snapshot.
  const localStreamingRef = useRef(false);
  const [streaming, setStreaming] = useState(false); // same flag, for rendering (Stop button)

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
  // Tracks the agent item currently being streamed into (deltas/thinking).
  const streamRef = useRef<{ agent: string; id: number } | null>(null);
  const idRef = useRef(0);

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
    localStorage.setItem(NOLIMIT_KEY, noLimit ? "1" : "0");
  }, [noLimit]);

  // Poll the shared conversation so every session sees the one chat. Skip applying
  // the snapshot while THIS client is streaming its own debate (it renders live).
  useEffect(() => {
    let alive = true;
    async function poll() {
      if (localStreamingRef.current) return;
      try {
        const r = await fetch("/api/conversation");
        const snap = await r.json();
        if (!alive || localStreamingRef.current) return;
        setItems((snap.items ?? []) as ChatItem[]);
        setBusy(!!snap.busy);
        setStatus(snap.status ?? "");
        setUsage(
          Object.fromEntries(
            Object.entries(snap.usage ?? {}).map(([k, v]: [string, any]) => [
              k,
              { prompt: v.prompt ?? 0, completion: v.completion ?? 0 },
            ])
          )
        );
      } catch {
        /* ignore transient poll errors */
      }
    }
    poll();
    const id = setInterval(poll, POLL_MS);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, []);

  function push(item: ChatItem) {
    setItems((prev) => [...prev, item]);
  }

  // Find/create the agent item being streamed; subsequent deltas append to it.
  function ensureStreamItem(agent: string): number {
    if (streamRef.current && streamRef.current.agent === agent) return streamRef.current.id;
    const id = ++idRef.current;
    streamRef.current = { agent, id };
    setItems((prev) => [...prev, { kind: "agent", agent, content: "", thinking: "", id }]);
    return id;
  }
  function appendStream(id: number, field: "content" | "thinking", delta: string) {
    setItems((prev) =>
      prev.map((it) => (it.kind === "agent" && it.id === id ? { ...it, [field]: (it[field] ?? "") + delta } : it))
    );
  }

  function addUsage(agent: string, prompt: number, completion: number) {
    setUsage((prev) => {
      const cur = prev[agent] ?? { prompt: 0, completion: 0 };
      return { ...prev, [agent]: { prompt: cur.prompt + prompt, completion: cur.completion + completion } };
    });
  }

  // Reset ALL settings (roster + context + turns + no-limit) to defaults, after a
  // confirmation warning. Persisted overrides are removed. Does not touch the chat.
  function resetSettings() {
    if (!window.confirm("Reset all settings to defaults? This can't be undone.")) return;
    if (defaults.length) {
      const next = defaults.map((d, i) => ({ ...d, model: models[i] ?? models[0] ?? d.model, num_ctx: FALLBACK.numCtx }));
      setRoster(next);
    }
    setNumCtx(FALLBACK.numCtx);
    setMaxTurns(FALLBACK.maxTurns);
    setNoLimit(false);
    [ROSTER_KEY, CTX_KEY, TURNS_KEY, NOLIMIT_KEY].forEach((k) => localStorage.removeItem(k));
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

  async function clearConversation() {
    try {
      await fetch("/api/conversation/clear", { method: "POST" });
    } catch {
      /* ignore */
    }
    setItems([]);
    setUsage({});
  }

  function stopDebate() {
    abortRef.current?.abort();
  }

  async function handleSend(text: string, images: string[]) {
    // This client owns the live stream; suppress the poll snapshot meanwhile.
    localStreamingRef.current = true;
    setStreaming(true);
    push({ kind: "user", text, images: images.length ? images : undefined });
    setBusy(true);
    setStatus("Starting…");
    streamRef.current = null;
    const controller = new AbortController();
    abortRef.current = controller;

    try {
      await streamChat(
        text,
        { numCtx, maxTurns, unlimited: noLimit, agents: roster, images },
        (e) => {
          if (e.type === "status") {
            setStatus(e.text);
            if (e.stage === "consensus") push({ kind: "consensus" });
            else if (e.stage === "closed") push({ kind: "closed" });
          } else if (e.type === "research")
            push({ kind: "research", query: e.query, sources: e.sources, screenshot: e.screenshot_b64 });
          else if (e.type === "delta") appendStream(ensureStreamItem(e.agent), "content", e.content);
          else if (e.type === "thinking_delta") appendStream(ensureStreamItem(e.agent), "thinking", e.content);
          else if (e.type === "message") {
            // Finalize the streamed item with the authoritative (consensus-stripped) text.
            const cur = streamRef.current;
            if (cur && cur.agent === e.agent) {
              const id = cur.id;
              setItems((prev) =>
                prev.map((it) => (it.kind === "agent" && it.id === id ? { ...it, content: e.content } : it))
              );
              streamRef.current = null;
            } else {
              push({ kind: "agent", agent: e.agent, content: e.content });
            }
          } else if (e.type === "usage") addUsage(e.agent, e.prompt_tokens, e.completion_tokens);
          else if (e.type === "error") push({ kind: "error", text: e.text });
        },
        controller.signal
      );
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") {
        // Stopped by the user — backend cancels the debate.
      } else {
        const msg = err instanceof Error ? err.message : "Request failed";
        push({ kind: "error", text: /409/.test(msg) ? "A debate is already running." : msg });
      }
    } finally {
      setBusy(false);
      setStatus("");
      abortRef.current = null;
      localStreamingRef.current = false;
      setStreaming(false);
    }
  }

  const totalGenerated = Object.values(usage).reduce((s, u) => s + u.completion, 0);
  const totalAll = Object.values(usage).reduce((s, u) => s + u.prompt + u.completion, 0);

  return (
    <div className="flex h-full text-slate-900 dark:text-white">
      <aside className="flex w-72 flex-col gap-4 overflow-y-auto border-r border-slate-200 bg-white px-4 py-6 dark:border-[#4a4a4a] dark:bg-[#3c3c3c]">
        <div className="flex items-start justify-between">
          <div>
            <h1 className="text-lg font-semibold text-slate-800 dark:text-white">Protein Design Agent</h1>
            <p className="mt-1 text-xs text-slate-400 dark:text-[#9a9a9a]">Local · Ollama · Playwright</p>
          </div>
          <button
            onClick={toggleDark}
            title="Toggle dark mode"
            className="rounded-lg border border-slate-200 px-2 py-1 text-sm dark:border-[#4a4a4a]"
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
          <label className="text-xs font-medium text-slate-500 dark:text-[#b5b5b5]">Max debate turns</label>
          <input
            type="number"
            value={maxTurns}
            min={turnsBounds.min}
            max={turnsBounds.max}
            disabled={busy || noLimit}
            onChange={(e) => setMaxTurns(Number(e.target.value))}
            onBlur={(e) =>
              setMaxTurns(Math.max(turnsBounds.min, Math.min(turnsBounds.max, Number(e.target.value) || turnsBounds.min)))
            }
            className="mt-1 w-full rounded-lg border border-slate-300 px-2 py-1.5 text-sm outline-none focus:border-slate-500 disabled:opacity-50 dark:border-[#4a4a4a] dark:bg-[#3c3c3c] dark:text-white"
          />
          <label className="mt-1.5 flex items-center gap-2 text-[11px] text-slate-500 dark:text-[#b5b5b5]">
            <input type="checkbox" checked={noLimit} disabled={busy} onChange={(e) => setNoLimit(e.target.checked)} />
            No limit (run until consensus or deadlock)
          </label>
          <p className="mt-1 text-[11px] text-slate-400 dark:text-[#9a9a9a]">
            1 turn = every agent speaks once. If they never agree, the Critic closes with why.
          </p>
        </div>

        <button
          onClick={() => setShowRoster(true)}
          className="rounded-lg border border-slate-300 px-3 py-2 text-sm text-slate-700 hover:border-slate-400 dark:border-[#4a4a4a] dark:text-[#ededed]"
        >
          Configure agents ({roster.length})
        </button>

        <div className="flex gap-2">
          <button
            onClick={clearConversation}
            disabled={items.length === 0}
            className="flex-1 rounded-lg border border-slate-300 px-2 py-1.5 text-xs text-slate-600 hover:border-slate-400 disabled:opacity-40 dark:border-[#4a4a4a] dark:text-[#dcdcdc]"
          >
            Clear chat
          </button>
          <button
            onClick={() => downloadText("conversation.md", conversationToMarkdown(items), "text/markdown")}
            disabled={items.length === 0}
            className="flex-1 rounded-lg border border-slate-300 px-2 py-1.5 text-xs text-slate-600 hover:border-slate-400 disabled:opacity-40 dark:border-[#4a4a4a] dark:text-[#dcdcdc]"
          >
            Download chat
          </button>
        </div>

        {/* Token usage */}
        {Object.keys(usage).length > 0 && (
          <div className="text-xs">
            <p className="font-medium text-slate-500 dark:text-[#b5b5b5]">Tokens generated</p>
            <div className="mt-1 space-y-0.5 text-slate-500 dark:text-[#b5b5b5]">
              {Object.entries(usage).map(([agent, u]) => (
                <p key={agent} className="flex justify-between">
                  <span className="truncate">{agent}</span>
                  <span className="font-medium text-slate-700 dark:text-[#ededed]">{formatTokens(u.completion)}</span>
                </p>
              ))}
              <p className="flex justify-between border-t border-slate-200 pt-0.5 dark:border-[#4a4a4a]">
                <span>Total generated</span>
                <span className="font-semibold text-slate-800 dark:text-white">{formatTokens(totalGenerated)}</span>
              </p>
              <p className="flex justify-between text-slate-400 dark:text-[#9a9a9a]">
                <span>Total (incl. prompt)</span>
                <span>{formatTokens(totalAll)}</span>
              </p>
            </div>
          </div>
        )}

        <div className="mt-auto space-y-1 text-xs text-slate-400 dark:text-[#9a9a9a]">
          <p className="font-medium text-slate-500 dark:text-[#b5b5b5]">Roster</p>
          {roster.map((a, i) => (
            <p key={i}>
              {i + 1} · {a.name} <span className="text-slate-300 dark:text-[#8a8a8a]">— {a.model}</span>
              {a.with_research ? " 🔎" : ""}
              {caps[a.model]?.vision ? " 👁" : ""}
            </p>
          ))}
          {pdfName && <p className="pt-2 text-emerald-600 dark:text-emerald-400">Grounded on {pdfName}</p>}
        </div>
      </aside>

      <main className="flex-1 bg-slate-50 dark:bg-[#333]">
        <Chat
          items={items}
          busy={busy}
          status={status}
          streaming={streaming}
          onSend={handleSend}
          onStop={stopDebate}
        />
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
          onReset={resetSettings}
          onExport={exportConfig}
          onImport={importConfig}
        />
      )}
    </div>
  );
}

import { useRef, useState } from "react";
import { AgentConfig } from "../lib/sse";
import { formatCtx } from "../lib/download";

const FIELD =
  "w-full rounded-lg border border-slate-300 px-2 py-1.5 text-sm outline-none focus:border-slate-500 dark:border-slate-600 dark:bg-slate-900 dark:text-slate-100";

export type ModelCaps = Record<string, { vision: boolean | null }>;

function VisionBadge({ vision }: { vision: boolean | null | undefined }) {
  if (vision === true)
    return <span className="rounded bg-emerald-100 px-1.5 py-0.5 text-[10px] font-medium text-emerald-700 dark:bg-emerald-900/40 dark:text-emerald-300">👁 vision</span>;
  if (vision === false)
    return <span className="rounded bg-slate-100 px-1.5 py-0.5 text-[10px] font-medium text-slate-500 dark:bg-slate-700 dark:text-slate-400">no vision</span>;
  return <span className="rounded bg-amber-100 px-1.5 py-0.5 text-[10px] font-medium text-amber-700 dark:bg-amber-900/40 dark:text-amber-300" title="Capability unknown — verified on send">vision?</span>;
}

export default function AgentRoster({
  agents,
  models,
  caps,
  defaultNumCtx,
  ctxMin,
  ctxMax,
  onChange,
  onClose,
  onReset,
  onExport,
  onImport,
}: {
  agents: AgentConfig[];
  models: string[];
  caps: ModelCaps;
  defaultNumCtx: number;
  ctxMin: number;
  ctxMax: number;
  onChange: (agents: AgentConfig[]) => void;
  onClose: () => void;
  onReset: () => void;
  onExport: () => void;
  onImport: (raw: string) => string | null; // returns an error message, or null on success
}) {
  const [importError, setImportError] = useState("");
  const fileRef = useRef<HTMLInputElement>(null);

  function update(i: number, patch: Partial<AgentConfig>) {
    onChange(agents.map((a, idx) => (idx === i ? { ...a, ...patch } : a)));
  }
  function remove(i: number) {
    onChange(agents.filter((_, idx) => idx !== i));
  }
  function add() {
    onChange([
      ...agents,
      {
        name: `Verifier${agents.length + 1}`,
        model: models[0] ?? "",
        system_message: "You independently verify the proposal for correctness and flag any flaws.",
        with_research: false,
        num_ctx: defaultNumCtx,
      },
    ]);
  }
  function toggleCritique(i: number, target: string, on: boolean) {
    const cur = agents[i].critiques ?? [];
    const next = on ? [...new Set([...cur, target])] : cur.filter((n) => n !== target);
    update(i, { critiques: next });
  }
  async function onFile(file: File) {
    setImportError("");
    try {
      const err = onImport(await file.text());
      if (err) setImportError(err);
    } catch {
      setImportError("Could not read file.");
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4" onClick={onClose}>
      <div
        className="max-h-[85vh] w-full max-w-2xl overflow-y-auto rounded-2xl bg-white p-5 shadow-xl dark:bg-slate-800"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="mb-4 flex items-center justify-between">
          <div>
            <h2 className="text-lg font-semibold text-slate-800 dark:text-slate-100">Configure agents</h2>
            <p className="text-xs text-slate-400 dark:text-slate-500">
              Agents debate in order until they reach consensus. Add verifiers on different models.
            </p>
          </div>
          <button
            onClick={onClose}
            className="rounded-lg bg-slate-800 px-3 py-1.5 text-sm text-white dark:bg-sky-700"
          >
            Done
          </button>
        </div>

        {/* Config + reset toolbar */}
        <div className="mb-4 flex flex-wrap items-center gap-2">
          <button
            onClick={onReset}
            className="rounded-lg border border-slate-300 px-3 py-1.5 text-xs text-slate-700 hover:border-slate-400 dark:border-slate-600 dark:text-slate-200"
          >
            ↺ Reset to defaults
          </button>
          <button
            onClick={onExport}
            className="rounded-lg border border-slate-300 px-3 py-1.5 text-xs text-slate-700 hover:border-slate-400 dark:border-slate-600 dark:text-slate-200"
          >
            ⭳ Download config
          </button>
          <button
            onClick={() => fileRef.current?.click()}
            className="rounded-lg border border-slate-300 px-3 py-1.5 text-xs text-slate-700 hover:border-slate-400 dark:border-slate-600 dark:text-slate-200"
          >
            ⭱ Upload config
          </button>
          <input
            ref={fileRef}
            type="file"
            accept="application/json"
            className="hidden"
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) void onFile(f);
              e.target.value = "";
            }}
          />
          {importError && <span className="text-xs text-red-600 dark:text-red-400">{importError}</span>}
        </div>

        <div className="space-y-3">
          {agents.map((a, i) => (
            <div key={i} className="rounded-xl border border-slate-200 p-3 dark:border-slate-700">
              <div className="flex gap-2">
                <div className="flex-1">
                  <input
                    className={FIELD}
                    value={a.name}
                    placeholder="Agent name"
                    onChange={(e) => update(i, { name: e.target.value })}
                  />
                </div>
                <div className="flex-1">
                  <select className={FIELD} value={a.model} onChange={(e) => update(i, { model: e.target.value })}>
                    {!models.includes(a.model) && a.model && <option value={a.model}>{a.model} (missing)</option>}
                    {models.map((m) => (
                      <option key={m} value={m}>
                        {m}
                      </option>
                    ))}
                  </select>
                </div>
              </div>

              <div className="mt-1.5 flex items-center gap-2 text-[11px] text-slate-500 dark:text-slate-400">
                <VisionBadge vision={caps[a.model]?.vision} />
                {a.is_critic && (
                  <span className="rounded bg-amber-100 px-1.5 py-0.5 text-[10px] font-medium text-amber-700 dark:bg-amber-900/40 dark:text-amber-300">
                    critic
                  </span>
                )}
              </div>

              <textarea
                className={`${FIELD} mt-2`}
                rows={2}
                value={a.system_message}
                placeholder="Role / system prompt"
                onChange={(e) => update(i, { system_message: e.target.value })}
              />

              {/* Per-agent context window slider */}
              <div className="mt-2">
                <div className="flex items-center justify-between text-[11px] text-slate-500 dark:text-slate-400">
                  <span>Context window</span>
                  <span className="font-semibold text-slate-700 dark:text-slate-200">
                    {formatCtx(a.num_ctx ?? defaultNumCtx)}
                  </span>
                </div>
                <input
                  type="range"
                  min={ctxMin}
                  max={ctxMax}
                  step={512}
                  value={a.num_ctx ?? defaultNumCtx}
                  onChange={(e) => update(i, { num_ctx: Number(e.target.value) })}
                  className="mt-1 w-full accent-sky-600"
                />
              </div>

              {/* Critic targeting */}
              {a.is_critic && (
                <div className="mt-2 rounded-lg bg-amber-50 p-2 dark:bg-amber-900/20">
                  <p className="text-[11px] font-medium text-amber-800 dark:text-amber-300">Critiques which agents?</p>
                  <div className="mt-1 flex flex-wrap gap-x-3 gap-y-1">
                    {agents
                      .filter((_, idx) => idx !== i)
                      .map((other) => (
                        <label
                          key={other.name}
                          className="flex items-center gap-1.5 text-[11px] text-slate-600 dark:text-slate-300"
                        >
                          <input
                            type="checkbox"
                            checked={(a.critiques ?? []).includes(other.name)}
                            onChange={(e) => toggleCritique(i, other.name, e.target.checked)}
                          />
                          {other.name}
                        </label>
                      ))}
                  </div>
                </div>
              )}

              <div className="mt-2 flex items-center justify-between">
                <label className="flex items-center gap-2 text-xs text-slate-600 dark:text-slate-300">
                  <input
                    type="checkbox"
                    checked={a.with_research}
                    onChange={(e) => update(i, { with_research: e.target.checked })}
                  />
                  Can browse the web (Playwright research tool)
                </label>
                <button
                  onClick={() => remove(i)}
                  disabled={agents.length <= 1 || a.is_critic}
                  title={a.is_critic ? "The Critic is a protected default agent" : undefined}
                  className="text-xs text-red-600 hover:underline disabled:opacity-40 dark:text-red-400"
                >
                  Remove
                </button>
              </div>
            </div>
          ))}
        </div>

        <button
          onClick={add}
          disabled={models.length === 0}
          className="mt-4 w-full rounded-lg border border-dashed border-slate-300 py-2 text-sm text-slate-600 hover:border-slate-400 disabled:opacity-40 dark:border-slate-600 dark:text-slate-300"
        >
          + Add agent
        </button>
      </div>
    </div>
  );
}

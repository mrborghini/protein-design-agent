// Renderers for Pipeline-mode items: GPU VRAM swaps, bio-app job lifecycle, and
// produced artifacts (downloadable PDB/FASTA/score files with their metrics).

export function GpuBanner({ text }: { text: string }) {
  return (
    <div className="rounded-2xl border border-amber-200 bg-amber-50/70 px-4 py-2 text-xs text-amber-800 dark:border-amber-900 dark:bg-amber-950/30 dark:text-amber-300">
      🎛️ {text}
    </div>
  );
}

const STAGE_STYLE: Record<string, string> = {
  start: "border-sky-200 bg-sky-50/70 text-sky-800 dark:border-sky-900 dark:bg-sky-950/30 dark:text-sky-300",
  done: "border-emerald-200 bg-emerald-50/70 text-emerald-800 dark:border-emerald-900 dark:bg-emerald-950/30 dark:text-emerald-300",
  error: "border-red-200 bg-red-50 text-red-700 dark:border-red-900 dark:bg-red-950/40 dark:text-red-300",
};
const STAGE_ICON: Record<string, string> = { start: "⚙️", done: "✅", error: "⚠️" };

export function BioAppCard({ tool, stage, label, text }: { tool: string; stage: string; label?: string; text: string }) {
  const style = STAGE_STYLE[stage] ?? STAGE_STYLE.start;
  return (
    <div className={`rounded-2xl border px-4 py-2 text-sm ${style}`}>
      <span className="font-medium">
        {STAGE_ICON[stage] ?? "🧪"} {label || tool}
      </span>
      {text && <span className="ml-2 opacity-80">{text}</span>}
    </div>
  );
}

/** Numeric metric fields worth surfacing on an artifact chip (others are hidden). */
const METRIC_KEYS = ["plddt", "mean_plddt", "delta_plddt", "total_score", "interface_dG", "num_backbones", "num_sequences"];

export function ArtifactChip({ item }: { item: Record<string, unknown> }) {
  const path = String(item.path ?? "");
  const atype = String(item.atype ?? "artifact");
  const tool = String(item.tool ?? "");
  const fileName = path.split("/").pop() || path;
  const metrics = METRIC_KEYS.filter((k) => item[k] !== undefined && item[k] !== null).map(
    (k) => `${k}: ${String(item[k])}`
  );
  return (
    <div className="rounded-2xl border border-slate-200 bg-white/60 px-4 py-2 text-sm dark:border-[#4a4a4a] dark:bg-[#3c3c3c]">
      <div className="flex items-center justify-between gap-2">
        <span className="text-slate-700 dark:text-[#ededed]">
          📦 <span className="font-medium">{atype}</span>
          <span className="ml-1 text-xs text-slate-400 dark:text-[#b0b0b0]">· {tool}</span>
        </span>
        <a
          href={`/api/artifact/${encodeURI(path)}`}
          target="_blank"
          rel="noreferrer"
          className="shrink-0 text-xs text-sky-600 underline underline-offset-2 hover:text-sky-800 dark:text-sky-400"
          title={path}
        >
          {fileName} ↓
        </a>
      </div>
      {metrics.length > 0 && (
        <p className="mt-1 text-xs text-slate-500 dark:text-[#c0c0c0]">{metrics.join("  ·  ")}</p>
      )}
    </div>
  );
}

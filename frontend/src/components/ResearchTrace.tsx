import { useState } from "react";

type Source = { title: string; url: string };

export default function ResearchTrace({
  query,
  sources,
  screenshot,
}: {
  query: string;
  sources: Source[];
  screenshot: string;
}) {
  const [open, setOpen] = useState(false);

  return (
    <div className="rounded-2xl border border-emerald-200 bg-emerald-50/60 px-4 py-3 dark:border-emerald-900 dark:bg-emerald-950/30">
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center justify-between text-left"
      >
        <span className="text-sm font-medium text-emerald-800 dark:text-emerald-300">
          🔎 Web research · {sources.length} source{sources.length === 1 ? "" : "s"}
        </span>
        <span className="text-xs text-emerald-600 dark:text-emerald-400">{open ? "hide" : "show"}</span>
      </button>
      <p className="mt-1 text-xs text-emerald-700 dark:text-emerald-400">“{query}”</p>

      {open && (
        <div className="mt-3 space-y-3">
          <ul className="space-y-1">
            {sources.map((s, i) => (
              <li key={i} className="text-sm">
                <a
                  href={s.url}
                  target="_blank"
                  rel="noreferrer"
                  className="text-emerald-700 underline underline-offset-2 hover:text-emerald-900 dark:text-emerald-400 dark:hover:text-emerald-300"
                >
                  {s.title || s.url}
                </a>
              </li>
            ))}
          </ul>
          {screenshot && (
            <img
              src={`data:image/png;base64,${screenshot}`}
              alt="Headless browser screenshot"
              className="w-full rounded-lg border border-emerald-200 dark:border-emerald-900"
            />
          )}
        </div>
      )}
    </div>
  );
}

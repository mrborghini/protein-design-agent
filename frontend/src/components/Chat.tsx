import { useEffect, useRef, useState } from "react";
import DebatePanel from "./DebatePanel";
import { fileToDataURL } from "../lib/download";

export type ChatItem =
  | { kind: "user"; text: string; images?: string[] }
  | { kind: "agent"; agent: string; content: string; thinking?: string; id?: number; round?: number }
  | { kind: "research"; query: string; sources: { title: string; url: string }[]; screenshot: string }
  | { kind: "consensus" }
  | { kind: "closed" }
  // Pipeline mode items:
  | { kind: "gpu"; stage: string; text: string }
  | { kind: "bioapp"; tool: string; stage: string; label?: string; text: string }
  | { kind: "artifact"; tool: string; atype?: string; path: string; [k: string]: unknown }
  | { kind: "error"; text: string };

export type Mode = "debate" | "pipeline";

type Group = {
  user: { text: string; images?: string[] } | null;
  items: ChatItem[];
  consensus: boolean;
  closed: boolean;
};

/** Split the flat conversation into per-debate groups (a user message starts one). */
function groupItems(items: ChatItem[]): Group[] {
  const groups: Group[] = [];
  let cur: Group | null = null;
  for (const item of items) {
    if (item.kind === "user") {
      cur = { user: { text: item.text, images: item.images }, items: [], consensus: false, closed: false };
      groups.push(cur);
    } else {
      if (!cur) {
        cur = { user: null, items: [], consensus: false, closed: false };
        groups.push(cur);
      }
      if (item.kind === "consensus") cur.consensus = true;
      else if (item.kind === "closed") cur.closed = true;
      else cur.items.push(item);
    }
  }
  return groups;
}

export default function Chat({
  items,
  busy,
  status,
  streaming = false,
  mode = "debate",
  onSend,
  onStop,
}: {
  items: ChatItem[];
  busy: boolean;
  status: string;
  streaming?: boolean;
  mode?: Mode;
  onSend: (text: string, images: string[]) => void;
  onStop?: () => void;
}) {
  const [draft, setDraft] = useState("");
  const [images, setImages] = useState<string[]>([]);
  const scrollRef = useRef<HTMLDivElement>(null);
  // Stick to the bottom only while the user is already there; scrolling up pauses
  // auto-scroll, and scrolling back to the bottom resumes it.
  const stickRef = useRef(true);
  // Set just before we auto-scroll, so the resulting scroll event isn't mistaken
  // for the user scrolling (which would otherwise keep toggling stick off/on).
  const programmaticRef = useRef(false);

  function onScroll() {
    if (programmaticRef.current) {
      programmaticRef.current = false;
      return;
    }
    const el = scrollRef.current;
    if (!el) return;
    stickRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
  }

  useEffect(() => {
    const el = scrollRef.current;
    if (el && stickRef.current) {
      programmaticRef.current = true;
      el.scrollTop = el.scrollHeight; // instant — avoids fighting rapid stream updates
    }
  }, [items, status]);

  function submit() {
    const text = draft.trim();
    if ((!text && images.length === 0) || busy) return;
    onSend(text, images);
    setDraft("");
    setImages([]);
  }

  async function addImages(files: FileList | null) {
    if (!files) return;
    const urls = await Promise.all(Array.from(files).map((f) => fileToDataURL(f)));
    setImages((prev) => [...prev, ...urls]);
  }

  const groups = groupItems(items);

  return (
    <div className="flex h-full flex-col">
      <div ref={scrollRef} onScroll={onScroll} className="flex-1 space-y-3 overflow-y-auto px-6 py-6">
        {items.length === 0 && (
          <div className="mt-20 text-center text-sm text-slate-400 dark:text-[#c8c8c8]">
            {mode === "pipeline" ? (
              <>
                Describe a protein-design goal. The Professor orchestrates a Paper Analyst and a
                Bio-App Operator (Boltz-2, RFdiffusion, ProteinMPNN, PyRosetta) to work toward it.
              </>
            ) : (
              <>
                Ask the agents anything. They'll discuss it (browsing the web if needed)
                and debate until they reach a consensus.
              </>
            )}
          </div>
        )}

        {groups.map((g, gi) => (
          <div key={gi} className="space-y-3">
            {g.user && (
              <div className="flex justify-end">
                <div className="max-w-[80%] space-y-2">
                  {g.user.images && g.user.images.length > 0 && (
                    <div className="flex flex-wrap justify-end gap-2">
                      {g.user.images.map((src, ii) => (
                        <img key={ii} src={src} className="h-20 w-20 rounded-lg object-cover" alt="attachment" />
                      ))}
                    </div>
                  )}
                  {g.user.text && (
                    <div className="rounded-2xl bg-slate-800 px-4 py-2.5 text-sm text-white shadow-sm dark:bg-[#4a4a4a]">
                      {g.user.text}
                    </div>
                  )}
                </div>
              </div>
            )}
            {g.items.length > 0 && (
              <DebatePanel
                items={g.items}
                consensus={g.consensus}
                closed={g.closed}
                mode={mode}
                defaultOpen={gi === groups.length - 1}
              />
            )}
          </div>
        ))}

        {busy && (
          <div className="flex items-center gap-2 px-1 text-sm text-slate-500 dark:text-[#d0d0d0]">
            <span className="h-2 w-2 animate-pulse rounded-full bg-slate-400" />
            {status || "Working…"}
          </div>
        )}
      </div>

      <div className="border-t border-slate-200 bg-white px-6 py-4 dark:border-[#4a4a4a] dark:bg-[#3c3c3c]">
        {images.length > 0 && (
          <div className="mb-2">
            <div className="flex flex-wrap gap-2">
              {images.map((src, i) => (
                <div key={i} className="relative">
                  <img src={src} className="h-16 w-16 rounded-lg object-cover" alt="attachment" />
                  <button
                    onClick={() => setImages((prev) => prev.filter((_, idx) => idx !== i))}
                    className="absolute -right-1.5 -top-1.5 flex h-5 w-5 items-center justify-center rounded-full bg-slate-800 text-xs text-white"
                    title="Remove"
                  >
                    ×
                  </button>
                </div>
              ))}
            </div>
            <p className="mt-1 text-[11px] text-slate-400 dark:text-[#c8c8c8]">
              Images are only sent to vision-capable agents — not all models support vision.
            </p>
          </div>
        )}
        <div className="flex items-end gap-2">
          <label
            className="cursor-pointer rounded-xl border border-slate-300 px-3 py-2 text-sm text-slate-600 hover:border-slate-400 dark:border-[#4a4a4a] dark:text-[#dcdcdc]"
            title="Attach images"
          >
            🖼️
            <input
              type="file"
              accept="image/*"
              multiple
              className="hidden"
              onChange={(e) => {
                void addImages(e.target.files);
                e.target.value = "";
              }}
            />
          </label>
          <textarea
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                submit();
              }
            }}
            rows={1}
            placeholder={mode === "pipeline" ? "Describe a protein-design goal…" : "Ask the agents anything…"}
            className="max-h-40 flex-1 resize-none rounded-xl border border-slate-300 px-3 py-2 text-sm outline-none focus:border-slate-500 dark:border-[#4a4a4a] dark:bg-[#3c3c3c] dark:text-white dark:placeholder:text-slate-500"
          />
          {streaming ? (
            <button
              onClick={onStop}
              className="rounded-xl bg-red-600 px-4 py-2 text-sm font-medium text-white transition hover:bg-red-500"
              title="Stop the debate"
            >
              Stop
            </button>
          ) : (
            <button
              onClick={submit}
              disabled={busy || (!draft.trim() && images.length === 0)}
              title={busy ? "A debate is already running" : undefined}
              className="rounded-xl bg-slate-800 px-4 py-2 text-sm font-medium text-white transition hover:bg-slate-700 disabled:opacity-40 dark:bg-sky-700 dark:hover:bg-sky-600"
            >
              Send
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

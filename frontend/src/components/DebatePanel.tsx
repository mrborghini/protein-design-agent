import type { ChatItem } from "./Chat";
import AgentMessage from "./AgentMessage";
import ResearchTrace from "./ResearchTrace";

/**
 * A collapsible transcript for one debate (everything the agents produced in
 * response to a single user message). Expanded by default; click to collapse.
 */
export default function DebatePanel({
  items,
  consensus,
  closed = false,
  defaultOpen = true,
}: {
  items: ChatItem[];
  consensus: boolean;
  closed?: boolean;
  defaultOpen?: boolean;
}) {
  const replies = items.filter((it) => it.kind === "agent").length;

  return (
    <details open={defaultOpen} className="rounded-2xl border border-slate-200 bg-white/40 dark:border-[#4a4a4a] dark:bg-[#3c3c3c]">
      <summary className="flex cursor-pointer select-none items-center justify-between rounded-2xl px-4 py-2 text-xs font-medium text-slate-500 hover:bg-slate-100 dark:text-[#b5b5b5] dark:hover:bg-[#454545]">
        <span>
          Debate · {replies} {replies === 1 ? "reply" : "replies"}
        </span>
        {consensus && <span className="text-emerald-600 dark:text-emerald-400">consensus ✓</span>}
        {!consensus && closed && (
          <span className="text-amber-600 dark:text-amber-400">closed without consensus</span>
        )}
      </summary>
      <div className="space-y-3 px-3 pb-3">
        {items.map((item, i) => {
          if (item.kind === "agent")
            return <AgentMessage key={i} agent={item.agent} content={item.content} thinking={item.thinking} />;
          if (item.kind === "research")
            return <ResearchTrace key={i} query={item.query} sources={item.sources} screenshot={item.screenshot} />;
          if (item.kind === "error")
            return (
              <div
                key={i}
                className="rounded-2xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700 dark:border-red-900 dark:bg-red-950/40 dark:text-red-300"
              >
                {item.text}
              </div>
            );
          return null;
        })}
      </div>
    </details>
  );
}

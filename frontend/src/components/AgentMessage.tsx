import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

const AGENT_STYLES: Record<string, { badge: string; ring: string }> = {
  LiteratureAgent: { badge: "bg-sky-100 text-sky-700 dark:bg-sky-900/50 dark:text-sky-300", ring: "border-sky-200 dark:border-sky-900" },
  HypothesisAgent: { badge: "bg-violet-100 text-violet-700 dark:bg-violet-900/50 dark:text-violet-300", ring: "border-violet-200 dark:border-violet-900" },
  Critic: { badge: "bg-amber-100 text-amber-700 dark:bg-amber-900/50 dark:text-amber-300", ring: "border-amber-200 dark:border-amber-900" },
};

const FALLBACK = {
  badge: "bg-slate-100 text-slate-700 dark:bg-[#454545] dark:text-[#ededed]",
  ring: "border-slate-200 dark:border-[#4a4a4a]",
};

export default function AgentMessage({
  agent,
  content,
  thinking,
}: {
  agent: string;
  content: string;
  thinking?: string;
}) {
  const style = AGENT_STYLES[agent] ?? FALLBACK;
  const hasThinking = !!thinking && thinking.trim().length > 0;
  // While thinking is streaming and no answer has arrived yet, default the panel open.
  const thinkingOpen = hasThinking && content.trim().length === 0;

  return (
    <div className={`rounded-2xl border ${style.ring} bg-white px-4 py-3 shadow-sm dark:bg-[#3c3c3c]`}>
      <span className={`inline-block rounded-full px-2.5 py-0.5 text-xs font-medium ${style.badge}`}>
        {agent}
      </span>

      {hasThinking && (
        <details open={thinkingOpen} className="mt-2 rounded-lg bg-slate-50 dark:bg-[#2b2b2b]">
          <summary className="cursor-pointer select-none px-3 py-1.5 text-xs font-medium text-slate-500 dark:text-[#b5b5b5]">
            💭 Thinking
          </summary>
          <div className="whitespace-pre-wrap px-3 pb-2 text-xs leading-relaxed text-slate-500 dark:text-[#b5b5b5]">
            {thinking}
          </div>
        </details>
      )}

      {content.trim().length > 0 && (
        <div className="md mt-2 text-slate-800 dark:text-white">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
        </div>
      )}
    </div>
  );
}

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

const AGENT_STYLES: Record<string, { badge: string; ring: string }> = {
  LiteratureAgent: { badge: "bg-sky-100 text-sky-700 dark:bg-sky-900/50 dark:text-sky-300", ring: "border-sky-200 dark:border-sky-900" },
  HypothesisAgent: { badge: "bg-violet-100 text-violet-700 dark:bg-violet-900/50 dark:text-violet-300", ring: "border-violet-200 dark:border-violet-900" },
  Critic: { badge: "bg-amber-100 text-amber-700 dark:bg-amber-900/50 dark:text-amber-300", ring: "border-amber-200 dark:border-amber-900" },
};

const FALLBACK = {
  badge: "bg-slate-100 text-slate-700 dark:bg-slate-700 dark:text-slate-200",
  ring: "border-slate-200 dark:border-slate-700",
};

export default function AgentMessage({ agent, content }: { agent: string; content: string }) {
  const style = AGENT_STYLES[agent] ?? FALLBACK;

  return (
    <div className={`rounded-2xl border ${style.ring} bg-white px-4 py-3 shadow-sm dark:bg-slate-800`}>
      <span className={`inline-block rounded-full px-2.5 py-0.5 text-xs font-medium ${style.badge}`}>
        {agent}
      </span>
      <div className="md mt-2 text-slate-800 dark:text-slate-100">
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
      </div>
    </div>
  );
}

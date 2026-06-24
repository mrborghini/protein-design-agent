import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";

// Shared markdown renderer for agent output. remark-math parses `$...$` (inline)
// and `$$...$$` (display) LaTeX; rehype-katex typesets it. The `.md` class (see
// index.css) styles the rendered HTML. Used for both the answer body and the
// thinking channel so they stay in sync.
export default function Markdown({
  children,
  className = "md",
}: {
  children: string;
  className?: string;
}) {
  return (
    <div className={className}>
      <ReactMarkdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeKatex]}>
        {children}
      </ReactMarkdown>
    </div>
  );
}

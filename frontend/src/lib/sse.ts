// Minimal fetch-based SSE reader for POST /api/chat.
// EventSource only supports GET, so we parse the text/event-stream ourselves.

export type StreamEvent =
  | { type: "status"; stage: string; text: string }
  | { type: "research"; query: string; sources: { title: string; url: string }[]; screenshot_b64: string }
  | { type: "message"; agent: string; content: string; round?: number }
  | { type: "delta"; agent: string; content: string; round?: number }
  | { type: "thinking_delta"; agent: string; content: string }
  | { type: "usage"; agent: string; prompt_tokens: number; completion_tokens: number; thinking_tokens: number }
  // Pipeline mode events:
  | { type: "gpu"; stage: string; text: string }
  | { type: "bioapp"; stage: string; tool: string; label?: string; text: string }
  | { type: "artifact"; tool: string; kind: string; path: string; [k: string]: unknown }
  | { type: "error"; text: string }
  | { type: "done" };

export type AgentConfig = {
  /** Stable client-only id for React keys + reordering. Stripped before send. */
  id?: string;
  name: string;
  model: string;
  system_message: string;
  with_research: boolean;
  /** Per-agent context window in tokens. Falls back to the global default if unset. */
  num_ctx?: number;
  /** Per-agent Ollama sampling knobs. Unset ⇒ Ollama's own default is used. */
  temperature?: number;
  top_p?: number;
  top_k?: number;
  min_p?: number;
  repeat_penalty?: number;
  num_predict?: number;
  /** Marks the protected default Critic (non-removable). */
  is_critic?: boolean;
  /** Names of agents this critic should focus its critique on. */
  critiques?: string[];
  /** Manual vision override: undefined = auto-detect, true/false = force on/off. */
  vision?: boolean | null;
};

export type ChatConfig = {
  numCtx: number;
  maxTurns: number;
  /** When true, run with no round cap (until consensus / deadlock / Stop). */
  unlimited?: boolean;
  agents: AgentConfig[];
  /** Base64 (data-URL) images to send to vision-capable agents. */
  images?: string[];
};

/** One role in Pipeline mode. `role` (professor/analyst/operator) keys the backend tool set. */
export type PipelineAgent = {
  id?: string;
  name: string;
  model: string;
  role: string;
  system_message?: string;
  temperature?: number;
  num_ctx?: number;
};

export type PipelineConfig = {
  numCtx: number;
  maxMessages: number;
  agents: PipelineAgent[];
};

/** Shared text/event-stream parser for a POST that returns SSE. */
async function readSSE(res: Response, onEvent: (e: StreamEvent) => void): Promise<void> {
  if (!res.ok || !res.body) throw new Error(`Request failed: ${res.status}`);
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const frames = buffer.split("\n\n");
    buffer = frames.pop() ?? "";
    for (const frame of frames) {
      const dataLine = frame.split("\n").find((l) => l.startsWith("data:"));
      if (!dataLine) continue;
      try {
        onEvent(JSON.parse(dataLine.slice(5).trim()) as StreamEvent);
      } catch {
        // ignore malformed frame
      }
    }
  }
}

export async function streamPipeline(
  message: string,
  config: PipelineConfig,
  onEvent: (e: StreamEvent) => void,
  signal: AbortSignal
): Promise<void> {
  const res = await fetch("/api/pipeline", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      message,
      num_ctx: config.numCtx,
      max_messages: config.maxMessages,
      // Drop the client-only `id` before sending.
      agents: config.agents.map(({ id: _id, ...a }) => a),
    }),
    signal,
  });
  await readSSE(res, onEvent);
}

export async function streamChat(
  message: string,
  config: ChatConfig,
  onEvent: (e: StreamEvent) => void,
  signal: AbortSignal
): Promise<void> {
  const res = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      message,
      num_ctx: config.numCtx,
      max_turns: config.maxTurns,
      unlimited: !!config.unlimited,
      // Drop the client-only `id` (used for React keys/reordering) before sending.
      agents: config.agents.map(({ id: _id, ...a }) => a),
      images: config.images && config.images.length ? config.images : undefined,
    }),
    signal,
  });
  await readSSE(res, onEvent);
}

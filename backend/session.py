"""Minimal in-memory session store (POC: single global session, no persistence).

Holds the uploaded PDF context AND the shared conversation. The conversation is
global on purpose: every browser session sees the same chat, and clearing it on
the backend clears it for everyone. POC only — in-memory, unauthenticated, lost
on restart.
"""


class Session:
    def __init__(self) -> None:
        self.pdf_text: str = ""
        self.pdf_name: str = ""
        self.pdf_pages: int = 0

        # Shared conversation state (same item shapes the frontend renders).
        self.conversation: list[dict] = []
        self.busy: bool = False  # a debate is currently running
        self.status: str = ""  # latest status text, shown to observers
        self.usage: dict[str, dict] = {}  # per-agent {prompt, completion, thinking} token totals

    def set_pdf(self, name: str, text: str, pages: int) -> None:
        self.pdf_name = name
        self.pdf_text = text
        self.pdf_pages = pages

    def context_block(self, limit: int = 12000) -> str:
        """Return the uploaded document as grounding context, truncated for the prompt."""
        if not self.pdf_text:
            return ""
        text = self.pdf_text[:limit]
        return f"--- Uploaded document: {self.pdf_name} ({self.pdf_pages} pages) ---\n{text}\n--- end document ---"

    # --- shared conversation helpers --- #
    def append_item(self, item: dict) -> None:
        self.conversation.append(item)

    def set_status(self, text: str) -> None:
        self.status = text

    def _usage_entry(self, agent: str) -> dict:
        # `completion` is Ollama's eval_count (thinking INCLUDED); `thinking` is the estimated
        # share of it, so answer-only = completion - thinking. See streaming_client.create_stream.
        return self.usage.get(agent, {"prompt": 0, "completion": 0, "thinking": 0})

    def add_usage(self, agent: str, prompt: int, completion: int) -> None:
        cur = self._usage_entry(agent)
        self.usage[agent] = {
            "prompt": cur["prompt"] + prompt,
            "completion": cur["completion"] + completion,
            "thinking": cur.get("thinking", 0),
        }

    def add_thinking_usage(self, agent: str, thinking: int) -> None:
        cur = self._usage_entry(agent)
        self.usage[agent] = {
            "prompt": cur["prompt"],
            "completion": cur["completion"],
            "thinking": cur.get("thinking", 0) + thinking,
        }

    def clear_conversation(self) -> None:
        self.conversation = []
        self.usage = {}
        self.status = ""

    def snapshot(self) -> dict:
        return {
            "items": self.conversation,
            "busy": self.busy,
            "status": self.status,
            "usage": self.usage,
        }


# Single shared session for this POC.
session = Session()

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
        self.usage: dict[str, dict] = {}  # per-agent {prompt, completion} token totals

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

    def add_usage(self, agent: str, prompt: int, completion: int) -> None:
        cur = self.usage.get(agent, {"prompt": 0, "completion": 0})
        self.usage[agent] = {
            "prompt": cur["prompt"] + prompt,
            "completion": cur["completion"] + completion,
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

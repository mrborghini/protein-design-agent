"""Minimal in-memory session store (POC: single global session, no persistence)."""


class Session:
    def __init__(self) -> None:
        self.pdf_text: str = ""
        self.pdf_name: str = ""
        self.pdf_pages: int = 0

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


# Single shared session for this POC.
session = Session()

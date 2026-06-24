"""Headless Playwright web-research tool exposed to the LiteratureAgent.

The tool returns a plain-text summary (what the LLM consumes) and, as a side
effect, pushes a structured "research" event (sources + a base64 screenshot)
onto the per-request event queue so the web UI can show what the headless
browser actually did.

Search strategy: try DuckDuckGo's HTML endpoint first (open web), and fall back
to Wikipedia search when DuckDuckGo blocks automated access. The browser is
hardened against bot-detection (realistic UA/headers, navigator.webdriver
masking, consent-banner dismissal) so DuckDuckGo is usable headless.

Reading result pages uses a direct-HTTP fast path (stdlib urllib — ~8x faster
than a browser for static markup and no browser bot signature), falling back to
the stealth browser only for JS-heavy or blocked pages.

POC note: these are benign, low-volume reads of public pages (scientific
literature), not mass scraping or attacking protected targets — keep it that
way. For robust fully-local search, run a self-hosted SearXNG instance.
"""
import asyncio
import base64
import contextvars
import urllib.request
from html.parser import HTMLParser
from urllib.parse import parse_qs, quote, unquote, urlparse

from autogen_core.tools import FunctionTool

# Set by the request handler so the tool can stream events back to that client.
research_sink: contextvars.ContextVar[asyncio.Queue | None] = contextvars.ContextVar(
    "research_sink", default=None
)

DDG_HTML = "https://html.duckduckgo.com/html/"
WIKI_SEARCH = "https://en.wikipedia.org/w/index.php"
PAGE_TEXT_LIMIT = 2000  # chars of body text kept per visited page
MIN_DIRECT_TEXT = 200  # if direct HTTP yields less, treat page as JS-heavy and use the browser

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
ACCEPT_LANGUAGE = "en-US,en;q=0.9"

# Bot-detection hardening (see web_scraping_and_automation_report.pdf §8).
_LAUNCH_ARGS = [
    "--disable-blink-features=AutomationControlled",
    "--no-sandbox",
    "--disable-setuid-sandbox",
    "--disable-dev-shm-usage",
    "--disable-gpu",
    "--window-size=1920,1080",
]
_EXTRA_HEADERS = {
    "Accept-Language": ACCEPT_LANGUAGE,
    "sec-ch-ua": '"Chromium";v="124", "Not(A:Brand";v="24"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Linux"',
}
_STEALTH_INIT = """
Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
window.chrome = { runtime: {} };
Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5] });
Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
"""
_CONSENT_SELECTORS = [
    "#onetrust-accept-btn-handler",
    "#CybotCookiebotDialogBodyLevelButtonLevelOptinAllowAll",
    "button:has-text('Accept all')",
    "button:has-text('Accept')",
    "button:has-text('I agree')",
    "button:has-text('Alles accepteren')",
    "button:has-text('Accepteren')",
]


# --------------------------------------------------------------------------- #
# HTML -> text (stdlib only, no BeautifulSoup per dependency policy)
# --------------------------------------------------------------------------- #
class _TextExtractor(HTMLParser):
    _SKIP = {"script", "style", "noscript", "head", "template", "svg"}

    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self._skip_depth += 1

    def handle_endtag(self, tag):
        if tag in self._SKIP and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data):
        if self._skip_depth == 0:
            t = data.strip()
            if t:
                self.parts.append(t)


def _html_to_text(html: str) -> str:
    parser = _TextExtractor()
    try:
        parser.feed(html)
    except Exception:  # noqa: BLE001 - tolerate malformed markup
        pass
    return " ".join(parser.parts)


def _http_get_sync(url: str, timeout: int = 12) -> str:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept-Language": ACCEPT_LANGUAGE,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - http(s) only below
        charset = resp.headers.get_content_charset() or "utf-8"
        return resp.read().decode(charset, errors="replace")


def _decode_ddg_href(href: str) -> str:
    """DuckDuckGo HTML links are redirects like //duckduckgo.com/l/?uddg=<url>."""
    if not href:
        return ""
    if href.startswith("//"):
        href = "https:" + href
    parsed = urlparse(href)
    qs = parse_qs(parsed.query)
    if "uddg" in qs:
        return unquote(qs["uddg"][0])
    return href


async def _emit(event: dict) -> None:
    queue = research_sink.get()
    if queue is not None:
        await queue.put(event)


async def _dismiss_consent(page) -> None:
    for sel in _CONSENT_SELECTORS:
        try:
            loc = page.locator(sel).first
            if await loc.is_visible(timeout=800):
                await loc.click()
                return
        except Exception:  # noqa: BLE001 - best-effort
            pass


async def _settle(page) -> None:
    """Scroll to trigger lazy-loaded content, then return to top (PDF §9)."""
    try:
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight/2);")
        await page.wait_for_timeout(500)
        await page.evaluate("window.scrollTo(0, 0);")
    except Exception:  # noqa: BLE001
        pass


# --------------------------------------------------------------------------- #
# Search backends
# --------------------------------------------------------------------------- #
async def _search_duckduckgo(page, query: str, max_results: int) -> list[dict]:
    try:
        await page.goto(DDG_HTML + "?q=" + quote(query), wait_until="domcontentloaded", timeout=15000)
        await _dismiss_consent(page)
        anchors = await page.query_selector_all("a.result__a")
        out: list[dict] = []
        for a in anchors[:max_results]:
            title = (await a.inner_text()).strip()
            url = _decode_ddg_href(await a.get_attribute("href") or "")
            if url:
                out.append({"title": title, "url": url})
        return out
    except Exception:  # noqa: BLE001 - any failure means fall back
        return []


async def _search_wikipedia(page, query: str, max_results: int) -> list[dict]:
    await page.goto(
        f"{WIKI_SEARCH}?search={quote(query)}&ns0=1", wait_until="domcontentloaded", timeout=15000
    )
    # An exact title match redirects straight to the article.
    if "/wiki/" in page.url and "search=" not in page.url:
        return [{"title": await page.title(), "url": page.url}]

    anchors = await page.query_selector_all(".mw-search-result-heading a")
    out: list[dict] = []
    seen: set[str] = set()
    for a in anchors:
        href = await a.get_attribute("href") or ""
        title = (await a.inner_text()).strip()
        if href.startswith("/wiki/") and title and href not in seen:
            seen.add(href)
            out.append({"title": title, "url": "https://en.wikipedia.org" + href})
        if len(out) >= max_results:
            break
    return out


async def _fetch_page_text(context, url: str) -> str:
    """Read a result page: fast direct-HTTP path first, stealth browser fallback."""
    try:
        html = await asyncio.to_thread(_http_get_sync, url)
        text = " ".join(_html_to_text(html).split())[:PAGE_TEXT_LIMIT]
        if len(text) >= MIN_DIRECT_TEXT:
            return text
    except Exception:  # noqa: BLE001 - blocked / non-static; fall back to browser
        pass
    try:
        sub = await context.new_page()
        await sub.goto(url, wait_until="domcontentloaded", timeout=15000)
        body = await sub.inner_text("body")
        await sub.close()
        return " ".join(body.split())[:PAGE_TEXT_LIMIT]
    except Exception as e:  # noqa: BLE001 - best-effort per source
        return f"(could not read page: {e})"


# --------------------------------------------------------------------------- #
# Public tool
# --------------------------------------------------------------------------- #
async def web_research(query: str, max_results: int = 3) -> str:
    """Search the web for `query`, read the top results, and return a text summary.

    Use this to gather recent or external information relevant to the user's question.
    """
    # Imported lazily so the module imports even before `playwright install`.
    from playwright.async_api import async_playwright

    await _emit({"type": "status", "stage": "research", "text": f"Searching the web for: {query}"})

    sources: list[dict] = []
    screenshot_b64 = ""
    notes: list[str] = []

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True, args=_LAUNCH_ARGS)
            context = await browser.new_context(
                user_agent=USER_AGENT,
                locale="en-US",
                viewport={"width": 1920, "height": 1080},
                extra_http_headers=_EXTRA_HEADERS,
            )
            await context.add_init_script(_STEALTH_INIT)
            page = await context.new_page()

            engine = "DuckDuckGo"
            results = await _search_duckduckgo(page, query, max_results)
            if not results:
                engine = "Wikipedia"
                await _emit(
                    {"type": "status", "stage": "research", "text": "DuckDuckGo unavailable — searching Wikipedia…"}
                )
                results = await _search_wikipedia(page, query, max_results)

            # Screenshot the results page the user would have seen.
            await _settle(page)
            try:
                shot = await page.screenshot(full_page=False)
                screenshot_b64 = base64.b64encode(shot).decode("ascii")
            except Exception:  # noqa: BLE001
                screenshot_b64 = ""

            for r in results:
                snippet = await _fetch_page_text(context, r["url"])
                r["snippet"] = snippet
                sources.append(r)
                notes.append(f"### {r['title']}\n{r['url']}\n{snippet}\n")

            await browser.close()
    except Exception as e:  # noqa: BLE001 - surface failure to the agent + UI
        await _emit({"type": "status", "stage": "research", "text": f"Research failed: {e}"})
        return f"Web research failed for query '{query}': {e}"

    await _emit(
        {
            "type": "research",
            "query": f"{query}  ·  via {engine}",
            "sources": [{"title": s["title"], "url": s["url"]} for s in sources],
            "screenshot_b64": screenshot_b64,
        }
    )

    if not notes:
        return f"No usable web results found for '{query}'."
    return f"Web research results for '{query}' (via {engine}):\n\n" + "\n".join(notes)


# AutoGen tool the LiteratureAgent can call.
web_research_tool = FunctionTool(
    web_research,
    description="Search the web and read top results for a focused query.",
)

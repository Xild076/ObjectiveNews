import os
import sys
import re
from urllib.parse import urlparse
from pathlib import Path
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

APP_URLS_ENV = os.getenv("APP_URLS", "").strip()
APP_URL = os.getenv("APP_URL", "").strip()
SLEEP_PAGE_TEXT = os.getenv("SLEEP_PAGE_TEXT", "app is asleep")
MINIMUM_CONTENT_LENGTH = int(os.getenv("MINIMUM_CONTENT_LENGTH", "1000"))
KEEP_OPEN_MS = int(os.getenv("KEEP_OPEN_MS", "8000"))

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/127.0.0.0 Safari/537.36"
)

def parse_urls():
    if APP_URLS_ENV:
        parts = re.split(r"[\s,]+", APP_URLS_ENV)
        urls = [u.strip() for u in parts if u.strip()]
        return urls
    if APP_URL:
        return [APP_URL]
    return ["https://thinking-gemma.streamlit.app/", "https://xild-stockpred.streamlit.app/", "https://alitheiaai.streamlit.app/"]

def slug_for(url: str, idx: int) -> str:
    parsed = urlparse(url)
    base = f"{parsed.netloc}{parsed.path}".strip("/").lower()
    slug = re.sub(r"[^a-z0-9]+", "-", base).strip("-")
    if not slug:
        slug = f"app-{idx}"
    return slug

def log(msg: str):
    print(msg, flush=True)
    with open("debug.log", "a", encoding="utf-8") as f:
        f.write(msg + "\n")

def save_artifacts(page, base: str, always: bool = False):
    try:
        html = page.content()
        Path(f"response_{base}.html").write_text(html, encoding="utf-8")
    except Exception as e:
        log(f"WARN: Failed to save response_{base}.html: {e}")
    if always:
        try:
            page.screenshot(path=f"screenshot_{base}.png", full_page=False)
        except Exception as e:
            log(f"WARN: Failed to save screenshot_{base}.png: {e}")

def visit(page, url: str, idx: int) -> int:
    slug = slug_for(url, idx)
    ws_urls = []
    page.context.on("websocket", lambda ws: (ws_urls.append(ws.url), log(f"[{slug}] WebSocket opened: {ws.url}")))
    try:
        log(f"[{slug}] Navigating: {url}")
        page.goto(url, wait_until="domcontentloaded", timeout=60_000)
        selectors = ['[data-testid="stAppViewContainer"]', 'section.main', '#root']
        found = False
        for sel in selectors:
            try:
                page.wait_for_selector(sel, timeout=8_000)
                log(f"[{slug}] Detected Streamlit container: {sel}")
                found = True
                break
            except PWTimeout:
                continue
        if not found:
            log(f"[{slug}] WARN: Streamlit container not confirmed; continuing.")
        keep_open_ms = max(KEEP_OPEN_MS, 5000)
        page.wait_for_timeout(keep_open_ms)
        save_artifacts(page, slug, always=False)
        try:
            html = Path(f"response_{slug}.html").read_text(encoding="utf-8")
        except Exception:
            html = page.content()
        content_length = len(html.encode("utf-8", errors="ignore"))
        log(f"[{slug}] Content length: {content_length} bytes")
        if content_length < MINIMUM_CONTENT_LENGTH:
            log(f"[{slug}] CRITICAL: Content below minimum ({MINIMUM_CONTENT_LENGTH}).")
            save_artifacts(page, slug, always=True)
            return 47
        lower_html = html.lower()
        patterns = [
            r"checking your browser",
            r"just a moment",
            r"cf-browser-verification",
            r"<meta[^>]*http-equiv=['\"]refresh['\"]",
            r"window\.location",
        ]
        if SLEEP_PAGE_TEXT.lower() in lower_html:
            log(f"[{slug}] CRITICAL: Sleep message detected.")
            save_artifacts(page, slug, always=True)
            return 47
        for pat in patterns:
            if re.search(pat, lower_html):
                log(f"[{slug}] CRITICAL: Challenge/redirect pattern detected: {pat}")
                save_artifacts(page, slug, always=True)
                return 47
        if ws_urls:
            log(f"[{slug}] ✅ Session established (WebSockets observed: {len(ws_urls)}).")
        else:
            log(f"[{slug}] ℹ️ No WebSocket observed; session may still be active.")
        log(f"[{slug}] ✅ Completed keep-alive.")
        return 0
    except PWTimeout:
        save_artifacts(page, slug, always=True)
        log(f"[{slug}] CRITICAL: Navigation timeout.")
        return 47
    except Exception as e:
        save_artifacts(page, slug, always=True)
        log(f"[{slug}] CRITICAL: Unexpected error: {e}")
        return 47
    finally:
        try:
            page.context.off("websocket", None)
        except Exception:
            pass

def main() -> int:
    urls = parse_urls()
    log(f"Targets: {', '.join(urls)}")
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
                "--disable-background-timer-throttling",
                "--disable-backgrounding-occluded-windows",
                "--disable-renderer-backgrounding",
            ],
        )
        context = browser.new_context(
            user_agent=UA,
            ignore_https_errors=True,
            viewport={"width": 1200, "height": 800},
            extra_http_headers={
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": "https://google.com",
            },
        )
        def block_heavy(route):
            if route.request.resource_type in ("image", "media", "font"):
                return route.abort()
            return route.continue_()
        context.route("**/*", block_heavy)
        page = context.new_page()
        codes = []
        for i, u in enumerate(urls, start=1):
            codes.append(visit(page, u, i))
        try:
            context.close()
        except Exception:
            pass
        try:
            browser.close()
        except Exception:
            pass
    return 0 if all(c == 0 for c in codes) else 47

if __name__ == "__main__":
    sys.exit(main())

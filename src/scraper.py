import random
import time
import threading
from datetime import datetime, timedelta
from urllib.parse import urlparse, parse_qs, quote_plus
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from duckduckgo_search import DDGS
import trafilatura
from newspaper import Article, Config
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.request
from dateutil.parser import parse
from utility import get_stopwords, DLog, get_keywords, normalize_url, cache_data_decorator, IS_STREAMLIT
from typing import Any, Dict
import validators

PROXIES = []
logger = DLog(name="FETCH_ARTICLE")

class RateLimiter:
    lock = threading.Lock()
    last_time = 0
    # Be gentler under Streamlit's constrained runtime
    min_delay = 0.35 if IS_STREAMLIT else 1.0
    @staticmethod
    def wait():
        with RateLimiter.lock:
            elapsed = time.monotonic() - RateLimiter.last_time
            delay = RateLimiter.min_delay * random.uniform(0.8,1.2)
            if elapsed < delay:
                time.sleep(delay - elapsed)
            RateLimiter.last_time = time.monotonic()

logger.info("Setting up HTTP session with retries...")
session = requests.Session()
retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[500,502,503,504])
adapter = HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=10)
session.mount("http://", adapter)
session.mount("https://", adapter)

# Configure newspaper to respect a strict request timeout
_NP_CFG = Config()
_NP_CFG.request_timeout = 8
_NP_CFG.memoize_articles = False

def _get_headers():
    uas = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Mozilla/5.0 (X11; Linux x86_64)",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
        "Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X)",
        "Mozilla/5.0 (iPad; CPU OS 14_0 like Mac OS X)"
    ]
    return [{"User-Agent": ua} for ua in uas]

def _ddg_html_links(query: str, amount: int):
    """Fallback: scrape DuckDuckGo HTML results to avoid library date parsing bugs."""
    import re
    import urllib.parse
    url = "https://duckduckgo.com/html/?q=" + urllib.parse.quote_plus(query)
    hdr = random.choice(_get_headers())
    try:
        r = session.get(url, headers=hdr, timeout=8)
        r.raise_for_status()
        candidates = re.findall(r'href="(http[^"]+)"', r.text)
        cleaned = []
        for h in candidates:
            # DuckDuckGo wraps external links as "/l/?kh=1&uddg=<url>"
            if "/l/?" in h and "uddg=" in h:
                parsed = urllib.parse.urlparse(h)
                qs = urllib.parse.parse_qs(parsed.query)
                real = qs.get("uddg", [None])[0]
                if real and real.startswith("http"):
                    h = urllib.parse.unquote(real)
            if h.startswith("http") and "duckduckgo.com" not in h and h not in cleaned:
                cleaned.append(h)
            if len(cleaned) >= amount:
                break
        return cleaned
    except Exception:
        return []


def _ddg_news_links(keywords, amount=10):
    """Fetch news-ish links via DuckDuckGo.

    Order of attempts:
    1) ddgs.news (fast, structured)
    2) ddgs.text (fallback if news breaks)
    3) HTML scrape (last resort)
    """
    query = " ".join(keywords)
    links: list[str] = []

    # Attempt 1: structured news API
    try:
        with DDGS() as ddgs:
            for res in ddgs.news(keywords=query, max_results=amount * 4):
                href = res.get("url") or res.get("href")
                if href and href.startswith("http") and href not in links:
                    links.append(href)
                if len(links) >= amount * 2:
                    break
    except Exception as e:
        logger.warning(f"DuckDuckGo news failed: {e}")

    # Attempt 2: text search fallback
    if not links:
        logger.info("Falling back to DuckDuckGo text search")
        try:
            with DDGS() as ddgs:
                for res in ddgs.text(keywords=query, max_results=amount * 6):
                    href = res.get("href") or res.get("url")
                    if href and href.startswith("http") and href not in links:
                        links.append(href)
                    if len(links) >= amount * 2:
                        break
        except Exception as e:
            logger.warning(f"DuckDuckGo text search failed: {e}")

    # Attempt 3: HTML scrape fallback
    if not links:
        logger.info("Falling back to DuckDuckGo HTML scrape")
        links = _ddg_html_links(query, amount * 2)

    return links[: max(amount * 2, amount)]


def retrieve_links(keywords, amount=10):
    logger.info(f"Retrieving links for keywords: {keywords}")
    links = _ddg_news_links(keywords, amount)
    if not links:
        logger.warning("No links from DuckDuckGo; returning empty list")
    return links[:amount]

def retrieve_diverse_links(keywords, amount=10):
    logger.info("Retrieving diverse links for keywords: " + ", ".join(keywords))
    raw = _ddg_news_links(keywords, amount * 6)
    if not raw:
        return []
    seen = set()
    domain_map = {}
    for u in raw:
        d = urlparse(u).netloc.replace("www.", "")
        domain_map.setdefault(d, []).append(u)
    selection = []
    domains = list(domain_map.keys())
    random.shuffle(domains)
    for d in domains:
        if len(selection) >= amount:
            break
        for u in domain_map[d]:
            if u not in seen:
                seen.add(u)
                selection.append(u)
                break
    if len(selection) < amount:
        for u in raw:
            if len(selection) >= amount:
                break
            if u not in seen:
                seen.add(u)
                selection.append(u)
    return selection

def _format_date(dt):
    logger.info(f"_Formatting date: {dt}")
    if not dt:
        return ""
    if isinstance(dt, datetime):
        return dt.strftime("%Y-%m-%d")
    if isinstance(dt, timedelta):
        # Some sources return relative times as timedeltas; anchor to now minus delta
        return (datetime.now() - dt).strftime("%Y-%m-%d")
    try:
        return parse(dt).strftime("%Y-%m-%d")
    except:
        return str(dt)

@cache_data_decorator
def extract_article_details(url):
    logger.info(f"Extracting article details from URL: {url[:50]}")
    if not url.startswith("http"):
        return None

    # Strict per-article time budget (seconds)
    BUDGET = 8.0
    start = time.monotonic()

    def remaining_budget():
        return max(0.1, BUDGET - (time.monotonic() - start))

    RateLimiter.wait()
    proxy = random.choice(PROXIES) if PROXIES else None
    downloaded = None

    # Primary attempt: single HTTP GET with tight timeout, reuse global session adapter
    try:
        hdr = {"User-Agent": random.choice(_get_headers())["User-Agent"]}
        r = session.get(
            url,
            headers=hdr,
            proxies={"http": proxy, "https": proxy} if proxy else None,
            timeout=min(5.0, remaining_budget())
        )
        r.raise_for_status()
        downloaded = r.text
    except Exception:
        pass

    # Fallback 1: urllib with remaining budget
    if not downloaded and remaining_budget() > 0.2:
        try:
            with urllib.request.urlopen(url, timeout=min(4.0, remaining_budget())) as r2:
                downloaded = r2.read().decode("utf-8", errors="ignore")
        except Exception:
            pass

    # If we have HTML, prefer local extraction (no network)
    if downloaded:
        try:
            meta = trafilatura.extract_metadata(downloaded, default_url=url)
            text = trafilatura.extract(downloaded)
            if meta and text:
                return {
                    "source": urlparse(url).netloc.replace("www.",""),
                    "title": meta.title or "",
                    "author": meta.author or "",
                    "date": _format_date(getattr(meta, 'date', None)),
                    "text": text,
                    "link": url
                }
        except Exception:
            pass

    # Fallback 2: newspaper3k (respects timeout via config)
    if remaining_budget() > 0.2:
        try:
            RateLimiter.wait()
            art = Article(url, config=_NP_CFG)
            # Newspaper manages its own fetching; ensure we don't exceed budget by short-circuiting
            if remaining_budget() < 1.0:
                return None
            art.download()
            art.parse()
            return {
                "source": urlparse(url).netloc.replace("www.",""),
                "title": getattr(art, 'title', '') or "",
                "author": getattr(art, 'authors', []),
                "date": _format_date(getattr(art, 'publish_date', None)),
                "text": getattr(art, 'text', ''),
                "link": url
            }
        except Exception:
            return None

    # If we still have nothing or budget exceeded, give up
    return None

def extract_many_article_details(urls, workers=10):
    logger.info(f"Extracting details from {len(urls)} URLs...")
    results = []
    iterable = urls if IS_STREAMLIT else tqdm(urls, desc="Extracting")

    # To stay light in Streamlit, keep it sequential; otherwise, we could parallelize later.
    for u in iterable:
        d = extract_article_details(u)
        if d:
            results.append(d)
    return results

# @cache_data_decorator
def process_text_input_for_keyword(text: str) -> Dict[str, Any] | None:
    logger.info(f"Processing text input for keywords: {str(text)[:50]}")
    COMMON_TLDS = [
        'com', 'org', 'net', 'edu', 'gov', 'mil', 'int',
        'io', 'co', 'us', 'uk', 'de', 'jp', 'fr', 'au',
        'ca', 'cn', 'ru', 'ch', 'it', 'nl', 'se', 'no',
        'es', 'biz', 'info', 'mobi', 'name', 'ly',
        'xyz', 'online', 'site', 'tech', 'store', 'blog'
    ]
    stopwords = get_stopwords()
    methodology = -1
    keywords = None
    article = None
    text = text.strip()
    for tld in COMMON_TLDS:
        if "."+tld in text:
            if len(text.split()) == 1:
                if not validators.url(text):
                    text = normalize_url(text)
                if not validators.url(text):
                    return {"method": 1, "keywords": get_keywords(text), "extra_info": None}
                article = extract_article_details(text)
                if article and article.get('title'):
                    keywords = [word for word in article['title'].split() if word not in stopwords]
                else:
                    # Fall back to extracting keywords from the raw text if fetch failed
                    return {"method": 1, "keywords": get_keywords(text), "extra_info": None}
                methodology = 0
            break
    if keywords == None:
        methodology = 1
        if len(text.split()) <= 10:
            keywords = [word for word in text.split() if word not in stopwords]
        else:
            keywords = get_keywords(text)
    if not keywords:
        return None
    return {"method": methodology, "keywords": keywords, "extra_info": article}

def retrieve_information_online(keywords, link_num=10, diverse=True, extra_info=None):
    logger.info(f"Retrieving information online for keywords: {keywords}")
    # Ensure keywords is a list of tokens, not a single joined string
    links = retrieve_diverse_links(list(keywords), link_num) if diverse else retrieve_links(list(keywords), link_num)
    articles = []
    if links:
        fetched_articles = extract_many_article_details(links, workers=10)
        seen_links = set()
        seen_sources = set()
        seen_titles = set()
        for a in fetched_articles:
            source = a.get('source', '')
            title = a.get('title', '').strip().lower()
            link = a.get('link')
            if link and link not in seen_links and source not in seen_sources and title not in seen_titles:
                articles.append(a)
                seen_links.add(link)
                seen_sources.add(source)
                seen_titles.add(title)
        for a in fetched_articles:
            title = a.get('title', '').strip().lower()
            link = a.get('link')
            source = a.get('source', '')
            if link and link not in seen_links and title not in seen_titles and source not in seen_sources:
                articles.append(a)
                seen_links.add(link)
                seen_titles.add(title)
                seen_sources.add(source)
        for a in fetched_articles:
            link = a.get('link')
            if link and link not in seen_links:
                articles.append(a)
                seen_links.add(link)
        # Fallback: boost domain diversity with a larger pool if needed
        domains = {a.get('source','') for a in articles if a}
        if diverse and len(domains) < max(3, min(5, link_num//2)):
            more_links = retrieve_diverse_links(list(keywords), link_num * 3)
            more_links = [u for u in more_links if u not in seen_links]
            if more_links:
                more_articles = extract_many_article_details(more_links, workers=10)
                for a in more_articles:
                    if not a:
                        continue
                    source = a.get('source', '')
                    title = a.get('title', '').strip().lower()
                    link = a.get('link')
                    if link and link not in seen_links and source not in seen_sources and title not in seen_titles:
                        articles.append(a)
                        seen_links.add(link)
                        seen_sources.add(source)
                        seen_titles.add(title)
    if extra_info:
        articles.append(extra_info)
    if len(articles) > link_num:
        articles = random.sample(articles, link_num)
    return articles, links
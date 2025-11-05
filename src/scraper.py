import random
import time
import threading
from datetime import datetime
from urllib.parse import urlparse, parse_qs, quote_plus
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
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

def _try_google_news_html_scraping(keywords, amount=10):
    """Fallback: Try HTML scraping of Google News search results."""
    logger.info(f"HTML scraping fallback for keywords: {keywords}")
    headers_list = [
        {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"},
        {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"},
        {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
    ]
    search_query = '+'.join([quote_plus(k) for k in keywords])
    base_url = f"https://www.google.com/search?q={search_query}&gl=us&tbm=nws&num={amount}"
    logger.info(f"Constructed search URL: {base_url}")
    session = requests.Session()
    max_retries = 3
    for attempt in range(max_retries):
        logger.info(f"Attempt {attempt + 1}/{max_retries}")
        header = random.choice(headers_list)
        logger.debug(f"Using User-Agent: {header['User-Agent'][:50]}...")
        try:
            logger.debug("Sending GET request to Google News...")
            RateLimiter.wait()
            response = session.get(base_url, headers=header, timeout=10)
            logger.info(f"Response status code: {response.status_code}")
            if response.status_code != 200:
                logger.warning(f"Non-200 status code received: {response.status_code}")
                time.sleep(1)
                continue
            logger.debug(f"Response content length: {len(response.content)} bytes")
            soup = BeautifulSoup(response.content, "html.parser")
            all_links = soup.find_all('a', href=True)
            logger.info(f"Found {len(all_links)} total link elements in page")
            external_urls = set()
            sample_urls = [link.get('href', '')[:150] for link in all_links[:10] if link.get('href')]
            logger.info(f"Sample raw hrefs from page: {sample_urls}")
            for link_elem in all_links:
                raw_url = link_elem.get('href', '').strip()
                if not raw_url:
                    continue
                extracted_url = None
                if raw_url.startswith('https://'):
                    if any(domain in raw_url for domain in ['google.com', 'accounts.google', 'googleapis.com', 'gstatic.com']):
                        continue
                    if any(domain in raw_url for domain in ['facebook.com', 'twitter.com', 'x.com', 'reddit.com', 'youtube.com', 'instagram.com', 'linkedin.com']):
                        continue
                    extracted_url = raw_url
                elif raw_url.startswith('/url?'):
                    try:
                        parsed_query = parse_qs(urlparse(raw_url).query)
                        if 'q' in parsed_query:
                            candidate = parsed_query['q'][0]
                            if candidate.startswith('http') and 'google' not in candidate:
                                extracted_url = candidate
                                logger.debug(f"Extracted from redirect: {candidate[:80]}...")
                    except Exception as e:
                        logger.debug(f"Could not parse redirect URL {raw_url[:100]}: {e}")
                elif raw_url.startswith('/search?') or raw_url.startswith('?'):
                    continue
                if extracted_url:
                    try:
                        parsed = urlparse(extracted_url)
                        clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                        if parsed.query:
                            clean_url += f"?{parsed.query}"
                        if clean_url not in external_urls and len(clean_url) > 20:
                            external_urls.add(clean_url)
                            logger.debug(f"Added external URL: {clean_url[:100]}...")
                    except Exception as e:
                        logger.debug(f"Error parsing URL {extracted_url}: {e}")
                        continue
            results = list(external_urls)[:amount]
            logger.info(f"Extracted {len(results)} unique external links")
            if results:
                logger.info("Successfully extracted URLs:")
                for i, url in enumerate(results[:5], 1):
                    logger.info(f"  {i}. {url[:80]}...")
                return results
            logger.warning(f"No external links found on attempt {attempt + 1}")
            logger.warning("This might indicate Google is blocking the request or HTML structure changed")
        except Exception as e:
            logger.error(f"Exception on attempt {attempt + 1}: {type(e).__name__}: {str(e)}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
            if attempt < max_retries - 1:
                logger.info("Retrying after 2 seconds...")
                time.sleep(2)
            continue
    logger.warning(f"Failed to retrieve links after {max_retries} attempts, returning empty list")
    return []

def retrieve_links(keywords, amount=10):
    logger.info("Retrieving links for keywords: " + ", ".join(keywords))
    headers_list = _get_headers()
    search_query = '+'.join([quote_plus(k) for k in keywords])
    base_url = f"https://www.google.com/search?q={search_query}&gl=us&tbm=nws&num={amount}"
    session = requests.Session()
    max_retries = 8
    for _ in range(max_retries):
        header = random.choice(headers_list)
        RateLimiter.wait()
        try:
            r = session.get(base_url, headers=header, timeout=8)
            if r.status_code != 200 or "To continue, please type the characters" in r.text:
                time.sleep(random.uniform(1, 2))
                continue
            soup = BeautifulSoup(r.content, "html.parser")
            results = []
            for el in soup.select('div.Gx5Zad.xpd.EtOod.pkphOe a'):
                raw_url = el.get('href')
                if raw_url:
                    parsed = parse_qs(urlparse(raw_url).query).get('q')
                    if parsed:
                        candidate = parsed[0]
                        if candidate.startswith('http'):
                            results.append(candidate)
                            if len(results) >= amount:
                                return results
        except:
            time.sleep(random.uniform(1, 2))
            continue
    fallback_results = _try_google_news_html_scraping(keywords, amount)
    return fallback_results

def retrieve_diverse_links(keywords, amount=10):
    logger.info("Retrieving diverse links for keywords: " + ", ".join(keywords))
    raw = retrieve_links(keywords, amount * 5)
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
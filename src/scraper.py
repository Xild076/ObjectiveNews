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
from newspaper import Article
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.request
from dateutil.parser import parse
from utility import get_stopwords, DLog, get_keywords, normalize_url, cache_data_decorator
import validators

PROXIES = []
logger = DLog(name="FETCH_ARTICLE", level="DEBUG")

class RateLimiter:
    lock = threading.Lock()
    last_time = 0
    min_delay = 2
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

def _get_headers():
    uas = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Mozilla/5.0 (X11; Linux x86_64)",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
        "Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X)",
        "Mozilla/5.0 (iPad; CPU OS 14_0 like Mac OS X)"
    ]
    return [{"User-Agent": ua} for ua in uas]

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
            r = session.get(base_url, headers=header, timeout=10)
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
    return []

def retrieve_diverse_links(keywords, amount=10):
    logger.info("Retrieving diverse links for keywords: " + ", ".join(keywords))
    raw = retrieve_links(keywords, amount * 2)
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
    RateLimiter.wait()
    proxy = random.choice(PROXIES) if PROXIES else None
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        try:
            RateLimiter.wait()
            r = session.get(url, headers={"User-Agent": random.choice(_get_headers())["User-Agent"]}, proxies={"http": proxy, "https": proxy} if proxy else None, timeout=10)
            r.raise_for_status()
            downloaded = r.text
        except:
            pass
    if not downloaded:
        try:
            RateLimiter.wait()
            with urllib.request.urlopen(url, timeout=10) as r:
                downloaded = r.read().decode("utf-8", errors="ignore")
        except:
            pass
    if downloaded:
        meta = trafilatura.extract_metadata(downloaded, default_url=url)
        text = trafilatura.extract(downloaded)
        if meta and text:
            return {
                "source": urlparse(url).netloc.replace("www.",""),
                "title": meta.title or "",
                "author": meta.author or "",
                "date": _format_date(meta.date),
                "text": text,
                "link": url
            }
    try:
        RateLimiter.wait()
        art = Article(url)
        art.download()
        art.parse()
        return {
            "source": urlparse(url).netloc.replace("www.",""),
            "title": art.title or "",
            "author": art.authors,
            "date": _format_date(art.publish_date),
            "text": art.text,
            "link": url
        }
    except:
        return None

def extract_many_article_details(urls, workers=10):
    logger.info(f"Extracting details from {len(urls)} URLs...")
    results = []
    # Use tqdm for progress bar if running outside streamlit
    from utility import IS_STREAMLIT
    iterable = urls if IS_STREAMLIT else tqdm(urls, desc="Extracting")
    
    for u in iterable:
        d = extract_article_details(u) # This is now called on the main thread
        if d:
            results.append(d)
    return results

@cache_data_decorator
def process_text_input_for_keyword(text:str) -> str:
    logger.info(f"Processing text input for keywords: {str(text)[:50]}")
    COMMON_TLDS = [
        'com', 'org', 'net', 'edu', 'gov', 'mil', 'int',
        'io', 'co', 'us', 'uk', 'de', 'jp', 'fr', 'au',
        'ca', 'cn', 'ru', 'ch', 'it', 'nl', 'se', 'no',
        'es', 'mil', 'biz', 'info', 'mobi', 'name', 'ly',
        'xyz', 'online', 'site', 'tech', 'store', 'blog'
    ]
    stopwords = get_stopwords()
    methodology = -1
    keywords = None
    article = None
    text = text.strip()
    for tld in COMMON_TLDS:
        if "."+tld in text:
            if len(" ".split(text)) == 1:
                if not validators.url(text):
                    text = normalize_url(text)
                if not validators.url(text):
                    return get_keywords(text)
                article = extract_article_details(text)
                keywords = [word for word in article['title'].split() if word not in stopwords]
                methodology = 0
            break
    if keywords == None:
        methodology = 1
        if len(" ".split(text)) <= 10:
            keywords = [word for word in text.split() if word not in stopwords]
        else:
            keywords = get_keywords(text)
    if not keywords:
        return None
    return {"method": methodology, "keywords": keywords, "extra_info": article}

def retrieve_information_online(keywords, link_num=10, diverse=True, extra_info=None):
    logger.info(f"Retrieving information online for keywords: {keywords}")
    links = retrieve_diverse_links(" ".join(keywords), link_num) if diverse else retrieve_links(" ".join(keywords), link_num)
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
            if link and link not in seen_links and title not in seen_titles:
                articles.append(a)
                seen_links.add(link)
                seen_titles.add(title)
        for a in fetched_articles:
            link = a.get('link')
            if link and link not in seen_links:
                articles.append(a)
                seen_links.add(link)
    if extra_info:
        articles.append(extra_info)
    if len(articles) > link_num:
        articles = random.sample(articles, link_num)
    return articles, links
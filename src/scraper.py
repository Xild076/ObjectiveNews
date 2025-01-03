from utility import DLog

logger = DLog("Scraper", "DEBUG", "logs")

logger.info("Importing modules...")
import requests
import random
import time
from datetime import datetime
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs, quote_plus
import trafilatura
import urllib.request
from tqdm import tqdm
from newspaper import Article
import threading
import streamlit as st
logger.info("Modules imported...")

class RateLimiter:
    lock = threading.Lock()
    last_request_time = 0
    min_delay = 2
    @staticmethod
    def wait():
        with RateLimiter.lock:
            elapsed = time.time() - RateLimiter.last_request_time
            if elapsed < RateLimiter.min_delay:
                time.sleep(RateLimiter.min_delay - elapsed)
            RateLimiter.last_request_time = time.time()

@st.cache_data
def get_headers():
    return FetchArticle._get_headers()

class FetchArticle:
    @staticmethod
    def retrieve_links(keywords, amount=10):
        headers_list = get_headers()
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
                    time.sleep(random.uniform(5, 10))
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
                time.sleep(random.uniform(5, 10))
                continue
        return []

    @staticmethod
    def extract_article_details(url):
        if not url.startswith('http'):
            return None
        RateLimiter.wait()
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            try:
                time.sleep(random.uniform(2,5))
                r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
                r.raise_for_status()
                downloaded = r.text
            except:
                pass
        if not downloaded:
            try:
                time.sleep(random.uniform(2,5))
                with urllib.request.urlopen(url, timeout=10) as r:
                    downloaded = r.read().decode('utf-8', errors='ignore')
            except:
                pass
        if downloaded:
            m = trafilatura.extract_metadata(downloaded, default_url=url)
            t = trafilatura.extract(downloaded)
            if t and m:
                return {
                    "source": urlparse(url).netloc.replace('www.', ''),
                    "text": t,
                    "title": m.title or '',
                    "author": m.author or '',
                    "date": FetchArticle._format_date(m.date),
                    "link": url
                }
        try:
            time.sleep(random.uniform(2,5))
            a = Article(url)
            a.download()
            a.parse()
            return {
                "source": urlparse(url).netloc.replace('www.', ''),
                "text": a.text,
                "title": a.title,
                "author": a.authors,
                "date": FetchArticle._format_date(a.publish_date),
                "link": url
            }
        except:
            return None

    @staticmethod
    def extract_many_article_details(urls):
        data = []
        for url in urls:
            logger.info(f"Extracting article details from {url}")
            d = FetchArticle.extract_article_details(url)
            if d:
                data.append(d)
        return data

    @staticmethod
    def _get_headers():
        ua = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Mozilla/5.0 (X11; Linux x86_64)",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
            "Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X)",
            "Mozilla/5.0 (iPad; CPU OS 14_0 like Mac OS X)"
        ]
        return [{"User-Agent": x} for x in ua]

    @staticmethod
    def _format_date(dt):
        if not dt:
            return ""
        from datetime import datetime
        if isinstance(dt, datetime):
            return dt.strftime("%Y-%m-%d")
        try:
            from dateutil.parser import parse
            parsed = parse(dt)
            return parsed.strftime("%Y-%m-%d")
        except:
            return str(dt)


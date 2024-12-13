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

class FetchArticle:
    @staticmethod
    def retrieve_links(keywords, start_date, end_date, amount=10):
        headers_list = FetchArticle._get_headers()
        search_query = '+'.join([quote_plus(keyword) for keyword in keywords])
        formatted_start = datetime.strptime(start_date, "%Y-%m-%d").strftime("%m/%d/%Y")
        formatted_end = datetime.strptime(end_date, "%Y-%m-%d").strftime("%m/%d/%Y")
        base_url = (
            f"https://www.google.com/search?q={search_query}"
            f"&gl=us&tbm=nws&num={amount}"
            f"&tbs=cdr:1,cd_min:{formatted_start},cd_max:{formatted_end}"
        )
        session = requests.Session()
        max_retries = 8
        for _ in range(max_retries):
            header = random.choice(headers_list)
            time.sleep(random.uniform(2, 5))
            try:
                response = session.get(base_url, headers=header, timeout=10)
                if response.status_code != 200 or "To continue, please type the characters" in response.text:
                    continue
                soup = BeautifulSoup(response.content, "html.parser")
                news_results = []
                for el in soup.select('div.Gx5Zad.xpd.EtOod.pkphOe a'):
                    raw_url = el.get('href')
                    if raw_url:
                        parsed_url = parse_qs(urlparse(raw_url).query).get('q')
                        if parsed_url:
                            candidate = parsed_url[0]
                            if candidate.startswith('http'):
                                news_results.append(candidate)
                                if len(news_results) >= amount:
                                    return news_results
            except:
                continue
        return []

    @staticmethod
    def extract_article_details(url):
        if not url.startswith('http'):
            return None
        downloaded = trafilatura.fetch_url(url)
        if downloaded is None:
            headers = {'User-Agent': 'Mozilla/5.0'}
            try:
                time.sleep(random.uniform(2,5))
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                downloaded = response.text
            except:
                pass
        if downloaded is None:
            try:
                time.sleep(random.uniform(2,5))
                with urllib.request.urlopen(url, timeout=10) as response:
                    downloaded = response.read().decode('utf-8', errors='ignore')
            except:
                pass
        text = None
        metadata = None
        if downloaded:
            metadata = trafilatura.extract_metadata(downloaded, default_url=url)
            text = trafilatura.extract(downloaded)
        if text is None or metadata is None:
            try:
                time.sleep(random.uniform(2,5))
                article = Article(url)
                article.download()
                article.parse()
                text = article.text
                metadata = type('Meta', (object,), {
                    'title': article.title, 
                    'author': article.authors, 
                    'date': article.publish_date
                })()
            except:
                return None
        return {
            "source": urlparse(url).netloc.replace('www.', ''),
            "text": text if text else '',
            "title": metadata.title if metadata and hasattr(metadata, 'title') and metadata.title else '',
            "author": metadata.author if metadata and hasattr(metadata, 'author') and metadata.author else '',
            "date": metadata.date if metadata and hasattr(metadata, 'date') and metadata.date else '',
            "link": url
        }

    @staticmethod
    def extract_many_article_details(urls):
        extracted_text = []
        for url in tqdm(urls, desc="Extracting Articles"):
            article = FetchArticle.extract_article_details(url)
            if article:
                extracted_text.append(article)
        return extracted_text

    @staticmethod
    def _get_headers():
        user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Mozilla/5.0 (X11; Linux x86_64)",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
            "Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X)",
            "Mozilla/5.0 (iPad; CPU OS 14_0 like Mac OS X)"
        ]
        return [{"User-Agent": ua} for ua in user_agents]

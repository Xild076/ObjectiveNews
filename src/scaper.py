import requests
import random
import time
from datetime import datetime
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs, quote_plus
from newspaper import Article
import trafilatura

class FetchArticle:
    @staticmethod
    def retrieve_links(keywords, start_date, end_date, amount=10):
        headers_list = [{"User-Agent": "Mozilla/5.0"}]
        headers = random.choice(headers_list)
        max_retries = 3
        try:
            search_query = '+'.join([quote_plus(keyword) for keyword in keywords])
            formatted_start = datetime.strptime(start_date, "%Y-%m-%d").strftime("%m/%d/%Y")
            formatted_end = datetime.strptime(end_date, "%Y-%m-%d").strftime("%m/%d/%Y")
        except ValueError as ve:
            pass
        base_url = (
            f"https://www.google.com/search?q={search_query}"
            f"&gl=us&tbm=nws&num={amount}"
            f"&tbs=cdr:1,cd_min:{formatted_start},cd_max:{formatted_end}"
        )
        attempt = 0
        response = None
        while attempt < max_retries:
            try:
                response = requests.get(base_url, headers=headers, timeout=10)
                response.raise_for_status()
                break
            except requests.RequestException as e:
                attempt += 1
                wait_time = 2 ** attempt
                time.sleep(wait_time)
        if response is None:
            pass
        soup = BeautifulSoup(response.content, "html.parser")
        news_results = []
        for el in soup.find_all('div', class_='Gx5Zad xpd EtOod pkphOe'):
            link_tag = el.find('a', href=True)
            if link_tag:
                raw_url = link_tag['href']
                parsed_url = parse_qs(urlparse(raw_url).query).get('q')
                if parsed_url:
                    news_results.append(parsed_url[0])
        return news_results

    @staticmethod
    def extract_article_details(url):
        downloaded = trafilatura.fetch_url(url)
        if downloaded is None:
            return None
        metadata = trafilatura.extract_metadata(downloaded, default_url=url)
        if metadata is None:
            return None
        text = trafilatura.extract(downloaded)
        if text is None:
            return None
        return {
            "source": urlparse(url).netloc.replace('www.', ''),
            "text": text,
            "title": metadata.title,
            "author": metadata.author,
            "date": metadata.date
        }
    
    @staticmethod
    def extract_many_article_details(urls):
        extracted_text = []
        for url in urls:
            article = FetchArticle.extract_article_details(url)
            if article:
                extracted_text.append(article)
        return extracted_text
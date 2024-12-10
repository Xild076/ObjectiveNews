import requests
import random
import time
from datetime import datetime
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs, quote_plus
import trafilatura
import urllib.request
from newspaper import Article
import requests
from bs4 import BeautifulSoup
from scrapy import Selector
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from tqdm import tqdm


class FetchArticle:
    @staticmethod
    def retrieve_links(keywords, start_date, end_date, amount=10):
        methods = [
            FetchArticle._retrieve_with_requests_bs4,
            FetchArticle._retrieve_with_scrapy,
            FetchArticle._retrieve_with_selenium,
        ]

        for method in methods:
            try:
                news_results = method(keywords, start_date, end_date, amount)
                if news_results:
                    return news_results
            except Exception as e:
                print(f"Method {method.__name__} failed with error: {e}")
                continue
        return []

    @staticmethod
    def _retrieve_with_requests_bs4(keywords, start_date, end_date, amount):
        headers_list = [{"User-Agent": "Mozilla/5.0"}]
        headers = random.choice(headers_list)
        max_retries = 3
        search_query = '+'.join([quote_plus(keyword) for keyword in keywords])
        formatted_start = datetime.strptime(start_date, "%Y-%m-%d").strftime("%m/%d/%Y")
        formatted_end = datetime.strptime(end_date, "%Y-%m-%d").strftime("%m/%d/%Y")
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
            raise ConnectionError("Failed to retrieve data with requests after multiple attempts.")
        soup = BeautifulSoup(response.content, "html.parser")
        news_results = []
        for el in soup.find_all('div', class_='Gx5Zad xpd EtOod pkphOe'):
            link_tag = el.find('a', href=True)
            if link_tag:
                raw_url = link_tag['href']
                parsed_url = parse_qs(urlparse(raw_url).query).get('q')
                if parsed_url:
                    news_results.append(parsed_url[0])
                    if len(news_results) >= amount:
                        break
        return news_results

    @staticmethod
    def _retrieve_with_scrapy(keywords, start_date, end_date, amount):
        search_query = '+'.join([quote_plus(keyword) for keyword in keywords])
        formatted_start = datetime.strptime(start_date, "%Y-%m-%d").strftime("%m/%d/%Y")
        formatted_end = datetime.strptime(end_date, "%Y-%m-%d").strftime("%m/%d/%Y")
        base_url = (
            f"https://www.google.com/search?q={search_query}"
            f"&gl=us&tbm=nws&num={amount}"
            f"&tbs=cdr:1,cd_min:{formatted_start},cd_max:{formatted_end}"
        )
        response = requests.get(base_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        response.raise_for_status()
        selector = Selector(text=response.text)
        news_results = []
        for el in selector.css('div.Gx5Zad.xpd.EtOod.pkphOe'):
            link = el.css('a::attr(href)').get()
            if link:
                parsed_url = parse_qs(urlparse(link).query).get('q')
                if parsed_url:
                    news_results.append(parsed_url[0])
                    if len(news_results) >= amount:
                        break
        if not news_results:
            raise ValueError("Scrapy method did not find any results.")
        return news_results

    @staticmethod
    def _retrieve_with_selenium(keywords, start_date, end_date, amount):
        search_query = '+'.join([quote_plus(keyword) for keyword in keywords])
        formatted_start = datetime.strptime(start_date, "%Y-%m-%d").strftime("%m/%d/%Y")
        formatted_end = datetime.strptime(end_date, "%Y-%m-%d").strftime("%m/%d/%Y")
        base_url = (
            f"https://www.google.com/search?q={search_query}"
            f"&gl=us&tbm=nws&num={amount}"
            f"&tbs=cdr:1,cd_min:{formatted_start},cd_max:{formatted_end}"
        )
        options = Options()
        options.headless = True
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        driver = webdriver.Chrome(options=options)
        try:
            driver.get(base_url)
            wait = WebDriverWait(driver, 10)
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'div.Gx5Zad.xpd.EtOod.pkphOe')))
            elements = driver.find_elements(By.CSS_SELECTOR, 'div.Gx5Zad.xpd.EtOod.pkphOe')
            news_results = []
            for el in elements:
                link_tag = el.find_element(By.TAG_NAME, 'a')
                raw_url = link_tag.get_attribute('href')
                parsed_url = parse_qs(urlparse(raw_url).query).get('q')
                if parsed_url:
                    news_results.append(parsed_url[0])
                    if len(news_results) >= amount:
                        break
            if not news_results:
                raise ValueError("Selenium method did not find any results.")
            return news_results
        finally:
            driver.quit()
    
    @staticmethod
    def extract_article_details(url):
        downloaded = trafilatura.fetch_url(url)
        
        if downloaded is None:
            headers = {
                'User-Agent': 'Mozilla/5.0 (compatible; ArticleExtractor/1.0; +http://yourdomain.com/bot)'
            }
            try:
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                downloaded = response.text
            except requests.RequestException:
                pass
        
        if downloaded is None:
            try:
                with urllib.request.urlopen(url, timeout=10) as response:
                    downloaded = response.read().decode('utf-8', errors='ignore')
            except Exception:
                pass
        
        if downloaded is None:
            try:
                article = Article(url)
                article.download()
                article.parse()
                downloaded = article.text
            except Exception:
                pass
        
        if downloaded is None:
            return None
        
        metadata = trafilatura.extract_metadata(downloaded, default_url=url)
        if metadata is None:
            metadata = {}
        
        text = trafilatura.extract(downloaded)
        if text is None:
            return None
        
        return {
            "source": urlparse(url).netloc.replace('www.', ''),
            "text": text,
            "title": metadata.title if metadata.title else '',
            "author": metadata.author if metadata.author else '',
            "date": metadata.date if metadata.date else '',
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


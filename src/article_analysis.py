from summarizer import lexrank, summarize_text
from datetime import datetime
from objectify import objectify_text
from grouping import cluster_texts
from utility import clean_text, normalize_url, get_keywords, SentenceHolder, split_sentences
from typing import Dict, Literal, Optional
from scraper import FetchArticle
import validators
from nltk.corpus import stopwords

scraper = FetchArticle()

def process_text_input_for_keyword(text:str) -> str:
    COMMON_TLDS = [
        'com', 'org', 'net', 'edu', 'gov', 'mil', 'int',
        'io', 'co', 'us', 'uk', 'de', 'jp', 'fr', 'au',
        'ca', 'cn', 'ru', 'ch', 'it', 'nl', 'se', 'no',
        'es', 'mil', 'biz', 'info', 'mobi', 'name', 'ly',
        'xyz', 'online', 'site', 'tech', 'store', 'blog'
    ]
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
                article = scraper.extract_article_details(text)
                keywords = [word for word in article['title'].split() if word not in stopwords.words("english")]
                methodology = 0
            break
    if keywords == None:
        methodology = 1
        if len(" ".split(text)) <= 10:
            keywords = [word for word in text.split() if word not in stopwords.words("english")]
        else:
            keywords = get_keywords(text)
    if not keywords:
        return None
    return {"method": methodology, "keywords": keywords, "extra_info": article}

def retrieve_information_online(keywords, link_num=5, extra_info=None):
    articles = []
    links = []
    max_attempts = 2
    attempts = 0
    while not articles and attempts < max_attempts:
        links = scraper.retrieve_links(keywords, link_num)
        if links:
            fetched_articles = scraper.extract_many_article_details(links)
            articles.extend(fetched_articles)
        link_num += 1
        attempts += 1
    if extra_info:
        articles.append(extra_info)
    return articles, links

def group_individual_articles(article):
    rep_sentences = []
    text = article.get('text', '')
    sentences = split_sentences(text)
    sentences = [SentenceHolder(text=text, source=article['source'], author=article['author'], date=article['date']) for text in sentences]

    clusters = cluster_texts(sentences)

    for cluster in clusters:
        rep_sentences.append(cluster['representative'])
    return rep_sentences

def group_representative_sentences(rep_sentences):
    OPTIMAL_CLUSTERING_PARAMS = {
        "weights": 1.0,
        "context": False,
        "context_len": 2,
        "preprocess": False,
        "attention": False,
        "norm": 'max',
        "n_neighbors": 15,
        "n_components": 4,
        "umap_metric": 'cosine',
        "cluster_metric": 'euclidean',
        "algorithm": 'prims_kdtree',
        "cluster_selection_method": 'eom',
        "min_cluster_size": 3,
        "min_samples": 1
    }
    return cluster_texts(rep_sentences, OPTIMAL_CLUSTERING_PARAMS)

def summarize_clusters(clusters, level:Literal['fast', 'medium', 'slow'] = 'fast'):
    for cluster in clusters:
        if level == 'fast':
            summary = cluster['representative'].text
        elif level == 'medium':
            sentences = [sent.text for sent in cluster['sentences'] if sent.text.strip()]
            if not sentences or all(len(s.split()) == 0 for s in sentences):
                summary = cluster['representative'].text
            else:
                try:
                    text = lexrank(" ".join(sentences), n=max(1, round(len(sentences) // 3)), threshold=0.1)
                    summary = ""
                    for sent in text:
                        summary += "- " + sent + "\n"
                except ValueError:
                    summary = cluster['representative'].text
        else:
            summary = summarize_text(cluster['representative'].text, max_length=200, min_length=100, num_beams=4)
        cluster['summary'] = clean_text(summary)
    return clusters

def objectify_clusters(clusters):
    for cluster in clusters:
        cluster['summary'] = objectify_text(cluster['summary'])
    return clusters

def article_analysis(text: str, level:Literal['fast', 'medium', 'slow'] = 'fast') -> Dict[str, Optional[Dict]]:
    text = clean_text(text)
    print("[Info] Starting article analysis...")
    if not text:
        print("[Error] No valid text provided.")
        return {"error": "No valid text provided."}
    print("[Info] Processing text input for keywords...")
    keywords_info = process_text_input_for_keyword(text)
    if not keywords_info:
        print("[Error] No valid keywords found.")
        return {"error": "No valid keywords found."}
    print("[Info] Keywords found:", keywords_info['keywords'])
    articles, links = retrieve_information_online(keywords_info['keywords'], extra_info=keywords_info.get('extra_info'))
    if not articles:
        print("[Error] No articles found.")
        return {"error": "No articles found."}
    print(f"[Info] Retrieved {len(articles)} articles from {len(links)} links.")
    grouped_articles = []
    for article in articles:
        rep_sentences = group_individual_articles(article)
        grouped_articles.extend(rep_sentences)
    print(f"[Info] Grouped {len(grouped_articles)} representative sentences from articles.")
    clusters = group_representative_sentences(grouped_articles)
    print("[Info] Grouped representative sentences into clusters.")
    summarized_clusters = summarize_clusters(clusters, level)
    print("[Info] Summarized clusters.")
    objectified_clusters = objectify_clusters(summarized_clusters)
    print("[Info] Objectified clusters.")
    print("\n===== Article Analysis Result =====\n")

    return objectified_clusters

def visualize_article_analysis(analysis_result) -> None:
    for idx, cluster in enumerate(analysis_result, 1):
        print(f"\033[1mCluster {idx}\033[0m")
        print("\033[94mSummary:\033[0m")
        print(f"  {cluster['summary']}")
        print("\033[92mRepresentative Sentences:\033[0m")
        for sent in cluster['sentences']:
            print(f"    • {sent.text}\n      \033[90mSource: {sent.source} | Author: {sent.author} | Date: {sent.date}\033[0m")
        print("\033[90m" + "-"*60 + "\033[0m")

clusters = article_analysis("trump", level='medium')
visualize_article_analysis(clusters)
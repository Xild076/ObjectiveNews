import numpy as np
import nltk
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sentence_transformers import SentenceTransformer
from typing import Union, Type, List, Dict, Any, Literal
from nltk.stem import WordNetLemmatizer
from colorama import Fore, Style
import validators
from scraper import FetchArticle
from datetime import datetime, timedelta
from util import get_keywords, find_bias_rating
from grouping import cluster_text
from objectify_text import objectify_text
from summarizer import summarize_text


def cluster_articles(link, type: Literal['news', 'data']='news', link_num: int=10, debug_print: bool=False):
    if not validators.url(link):
        raise TypeError(f"Link {link} is not valid.")
    
    article = FetchArticle.extract_article_details(link)
    keywords = get_keywords(article['title'])

    if debug_print:
        print(Fore.GREEN + "Keywords Loaded: " + Fore.RESET + str(keywords))

    date = datetime.strptime(article['date'], '%Y-%m-%d')
    if type == 'news':
        date_range = 2
    elif type == 'data':
        date_range = 30
    start_date = (date - timedelta(days=date_range)).strftime('%Y-%m-%d')
    end_date = (date + timedelta(days=date_range)).strftime('%Y-%m-%d')

    if debug_print:
        print(Fore.GREEN + "Start Date: " + Fore.RESET + str(start_date) + Fore.GREEN + ", End Date: " + Fore.RESET + str(end_date))

    links = FetchArticle.retrieve_links(keywords, start_date, end_date, link_num)
    articles = FetchArticle.extract_many_article_details(links)
    articles.append(article)

    if debug_print:
        print(Fore.GREEN + "Links: " + Fore.RESET + str(links))

    rep_sentences = []
    for article in articles:
        text = article['text']
        sentences = nltk.sent_tokenize(text)
        clusters_article = cluster_text(sentences, context=True, context_weights={'single':0.4, 'context': 0.6}, score_weights={'sil': 0.45, 'db': 0.55, 'ch': 0.1}, clustering_method=AgglomerativeClustering)
        if clusters_article:
            for cluster in clusters_article['clusters']:
                rep_sentences.append({'text': cluster['representative_with_context'], 'source': article['source']})
    
    max_clusters = max(round(len(rep_sentences)/6), 15)

    if debug_print:
        print(Fore.GREEN + "Representative Sentences: " + Fore.RESET + str(len(rep_sentences)))
        print(Fore.GREEN + "Max Clusters: " + Fore.RESET + str(max_clusters))

    clusters = cluster_text(rep_sentences, score_weights={'sil': 0.5, 'db': 0.3, 'ch': 0.2}, clustering_method=AgglomerativeClustering, max_clusters=max_clusters)
    return clusters

def objectify_and_summarize(text_list: List[str]) -> str:
    if not text_list:
        return ""
    
    text = " ".join(text_list).replace("\'", "'")
    objectifed_text = objectify_text(text)
    
    num_sentences = len(text_list)
    
    if num_sentences <= 2:
        print(Fore.RED + "Warning: Few sentences in cluster. Using default summary lengths." + Fore.RESET)
        summary_length_min = 30
        summary_length_max = 130
    else:
        summary_length_min = round(len(objectifed_text) / (num_sentences + 2) / 2.5)
        summary_length_max = round(len(objectifed_text) / (num_sentences - 2) / 2.5)
        
    return summarize_text(objectifed_text, min_length=summary_length_min, max_length=summary_length_max)

result = cluster_articles('https://www.bbc.com/news/articles/c4gxplxy550o', link_num=10, debug_print=True)

for cluster in result['clusters']:
    print(Fore.YELLOW + Style.BRIGHT + f"Cluster {cluster['cluster_id']}" + Style.RESET_ALL)
    print(Fore.GREEN + f"Representative Sentence:", Fore.RESET + str(cluster['representative']))
    print(Fore.GREEN + f"Sentences:", Fore.RESET + objectify_and_summarize(cluster['sentences']))
    print(Fore.BLUE + f"Sources:", Fore.RESET + str(cluster['sources']))
    reliability = []
    for source in cluster['sources']:
        reliability.append(find_bias_rating(source))
    reliability_score = float(max(reliability) * 5 + sum(reliability)) / (len(reliability) + 5)
    print(Fore.BLUE + f"Reliability:", Fore.RESET + str(reliability_score))
    print("-" * 80)
print("Scores:", result['metrics'])
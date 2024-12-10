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
from text_fixer import clean_text
from textblob import TextBlob

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
import re
import langdetect
from concurrent.futures import ThreadPoolExecutor, as_completed


def is_cluster_valid(cluster, min_avg_sentence_length=50, alpha_ratio_threshold=0.7, min_sentences=2, debug_print=False):
    representative = cluster['representative']
    try:
        lang = langdetect.detect(representative)
        if lang != 'en':
            if debug_print:
                print(Fore.RED + "Cluster removed: Non-English representative sentence." + Fore.RESET)
            return False
    except:
        if debug_print:
            print(Fore.RED + "Cluster removed: Language detection failed." + Fore.RESET)
        return False
    alpha_ratio = len(re.findall(r'[A-Za-z]', representative)) / (len(representative) + 1)
    if alpha_ratio < alpha_ratio_threshold:
        if debug_print:
            print(Fore.RED + f"Cluster removed: Low alpha ratio ({alpha_ratio:.2f})." + Fore.RESET)
        return False
    non_content_phrases = [
        'error processing your request',
        'an email has been sent',
        'click here to',
        'subscribe to',
        'read more',
        '©',
        'all rights reserved',
        'cookies',
        'newsletter',
        'javascript',
        'privacy policy',
        'terms of service',
        '24/7',
        'log out',
        'sign in',
        'log in',
        'sign out',
        'coverage',
        'reporting',
        'news anchor'
    ]
    representative_lower = representative.lower()
    for phrase in non_content_phrases:
        if phrase in representative_lower:
            if debug_print:
                print(Fore.RED + f"Cluster removed: Non-content phrase '{phrase}' found." + Fore.RESET)
            return False
    sentences = cluster['sentences']
    if len(sentences) < min_sentences:
        if debug_print:
            print(Fore.RED + "Cluster removed: Insufficient number of sentences." + Fore.RESET)
        return False
    avg_sentence_length = sum(len(s) for s in sentences) / len(sentences)
    if avg_sentence_length < min_avg_sentence_length:
        if debug_print:
            print(Fore.RED + f"Cluster removed: Average sentence length too short ({avg_sentence_length:.2f})." + Fore.RESET)
        return False
    return True

def cluster_articles(link, type: Literal['news', 'data']='news', link_num: int=10, debug_print: bool=False):
    if not validators.url(link):
        raise TypeError(f"Link {link} is not valid.")
    article = FetchArticle.extract_article_details(link)
    keywords = get_keywords(article['title'])
    if debug_print:
        print(Fore.GREEN + "Keywords Loaded: " + Fore.RESET + str(keywords))
    date = datetime.strptime(article['date'], '%Y-%m-%d')
    date_range = 2 if type == 'news' else 30
    start_date = (date - timedelta(days=date_range)).strftime('%Y-%m-%d')
    end_date = (date + timedelta(days=date_range)).strftime('%Y-%m-%d')
    if debug_print:
        print(Fore.GREEN + f"Start Date: {start_date}, End Date: {end_date}" + Fore.RESET)
    links = FetchArticle.retrieve_links(keywords, start_date, end_date, link_num)
    articles = FetchArticle.extract_many_article_details(links)
    articles.append(article)
    if debug_print:
        print(Fore.GREEN + "Links: " + Fore.RESET + str(links))
    rep_sentences = []
    for article in articles:
        text = article['text']
        sentences = nltk.sent_tokenize(text)
        max_clusters = max(round(len(sentences) / 6), 10)
        clusters_article = cluster_text(sentences, context=True, context_weights={'single': 0.4, 'context': 0.6}, 
                                        score_weights={'sil': 0.45, 'db': 0.55, 'ch': 0.1}, 
                                        clustering_method=AgglomerativeClustering, max_clusters=max_clusters)
        if clusters_article:
            for cluster in clusters_article['clusters']:
                rep_sentences.append({'text': cluster['representative_with_context'], 'source': article['source']})
    max_clusters = max(round(len(rep_sentences) / 8), 15)
    if debug_print:
        print(Fore.GREEN + f"Representative Sentences: {len(rep_sentences)}, Max Clusters: {max_clusters}" + Fore.RESET)
    clusters = cluster_text(rep_sentences, score_weights={'sil': 0.5, 'db': 0.3, 'ch': 0.2}, 
                            clustering_method=AgglomerativeClustering, max_clusters=max_clusters)
    valid_clusters = []
    for cluster in clusters['clusters']:
        if is_cluster_valid(cluster, debug_print=debug_print):
            valid_clusters.append(cluster)
        else:
            if debug_print:
                print(Fore.RED + f"Removed Cluster {cluster['cluster_id']} due to invalid content." + Fore.RESET)
    clusters['clusters'] = valid_clusters
    return clusters

def objectify_and_summarize(text_list: List[str]) -> str:
    if not text_list:
        return ""
    
    text = " ".join(text_list).replace("\'", "'")
    objectifed_text = objectify_text(text)
    
    num_sentences = len(text_list)
    
    if num_sentences <= 2:
        print(Fore.RED + "Warning: Few sentences in cluster. Using default summary lengths." + Fore.RESET)
        summary_length_min = 50
        summary_length_max = 200
    else:
        summary_length_min = round(len(objectifed_text) / (num_sentences + 2) / 2.5)
        summary_length_max = round(len(objectifed_text) / (num_sentences - 2) / 2.5)
    
    return summarize_text(objectifed_text, min_length=summary_length_min, max_length=summary_length_max)

def calculate_reliability_and_summary(cluster):
    reliability = []
    for source in cluster['sources']:
        reliability.append(find_bias_rating(source))
    reliability_score = float(max(reliability) * 5 + sum(reliability)) / (len(reliability) + 5)
    objectivity = 0
    for sentence in cluster['sentences']:
        textblob = TextBlob(sentence)
        objectivity += textblob.subjectivity
    objectivity /= len(cluster['sentences'])
    objectivity = (objectivity / 2 + 0.75)
    cluster['reliability'] = reliability_score * objectivity
    summary = objectify_and_summarize(cluster['sentences'])
    cluster['summary'] = summary
    return cluster

def provide_metrics(result_dict: dict) -> dict:
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(calculate_reliability_and_summary, cluster) for cluster in result_dict['clusters']]
        for future in as_completed(futures):
            completed_cluster = future.result()
            for cluster in result_dict['clusters']:
                if cluster == completed_cluster:
                    cluster.update(completed_cluster)
    return result_dict

def organize_clusters(clusters:dict):
    clusters_organized = []
    for cluster in clusters['clusters']:
        sentences = []
        for sentence in cluster['sentences']:
            sentences.append(clean_text(sentence).strip())
        clusters_organized.append({'sentences': cluster['sentences'], 
                                   'representative': cluster['representative'].strip(), 
                                   'representative_with_context': clean_text(cluster['representative_with_context']).strip(),
                                   'sentences': sentences, 
                                   'summary': cluster.get('summary', None), 
                                   'reliability': cluster.get('reliability', None), 
                                   'sources': cluster['sources']})
    clusters_organized.sort(key=lambda x: len(x['summary']) if 'summary' in x and x['summary'] else 0)
    clusters_organized.reverse()
    return clusters_organized


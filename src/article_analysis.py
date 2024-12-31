from utility import DLog

logger = DLog(name="ArticleAnalysis", level="DEBUG", log_dir="logs")

logger.info("Importing modules...")
import validators
from utility import normalize_url, get_keywords, SentenceHolder, find_bias_rating, normalize_text
from scraper import FetchArticle
from nltk.corpus import stopwords
import nltk
nltk.download('punkt_tab')
from math import floor
from grouping import observe_best_cluster
from sklearn.cluster import AgglomerativeClustering, KMeans
from langdetect import detect, LangDetectException
from typing import Dict, List
from textblob import TextBlob
from reliability import calculate_date_relevancy, calculate_general_reliability
from objectify import objectify_text
from summarizer import summarize_text
import re
import time
from colorama import Fore, Style
from datetime import datetime
from keybert import KeyBERT
import streamlit as st
logger.info("Modules imported...")

@st.cache_data
def is_cluster_valid(cluster: Dict[str, any],
                    min_avg_sentence_length: int = 50,
                    alpha_ratio_threshold: float = 0.7,
                    min_sentences: int = 2,
                    debug: bool = False,
                    keywords: List[str] = None) -> bool:
    NON_CONTENT_PHRASES = {
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
        'news anchor'
    }
    ALPHA_REGEX = re.compile(r'[A-Za-z]')
    representative = cluster.get('representative', '').text
    sentences = cluster.get('sentences', [])

    if not representative or not isinstance(representative, str):
        if debug:
            logger.warning("Cluster removed: Representative is missing or not a string.")
        return False
    if not isinstance(sentences, list) or not all(isinstance(s.text, str) for s in sentences):
        if debug:
            logger.warning("Cluster removed: Sentences are missing or not all strings.")
        return False
    try:
        lang = detect(representative)
        if lang != 'en':
            if debug:
                logger.warning("Cluster removed: Non-English representative sentence.")
            return False
    except LangDetectException:
        if debug:
            logger.warning("Cluster removed: Language detection failed.")
        return False
    alpha_ratio = len(ALPHA_REGEX.findall(representative)) / (len(representative) + 1)
    if alpha_ratio < alpha_ratio_threshold:
        if debug:
            logger.warning(f"Cluster removed: Low alpha ratio ({alpha_ratio:.2f}).")
        return False
    representative_lower = representative.lower()
    if any(phrase in representative_lower for phrase in NON_CONTENT_PHRASES):
        if debug:
            logger.warning("Cluster removed: Contains non-content phrase.")
        return False
    if len(sentences) < min_sentences:
        if debug:
            logger.warning("Cluster removed: Insufficient number of sentences.")
        return False
    total_length = sum(len(sentence.text) for sentence in sentences)
    avg_length = total_length / len(sentences) if sentences else 0.0
    if avg_length < min_avg_sentence_length:
        if debug:
            logger.warning(f"Cluster removed: Average sentence length too short ({avg_length:.2f}).")
        return False
    if keywords:
        keyword_set = set(word.lower() for word in keywords)
        sentences_with_keywords = 0
        for sentence in sentences:
            sentence_words = set(word.lower() for word in re.findall(r'\b\w+\b', sentence.text))
            if sentence_words & keyword_set:
                sentences_with_keywords += 1
        if sentences_with_keywords < len(sentences) / 2:
            if debug:
                logger.warning("Cluster removed: Majority of sentences do not contain the specified keywords.")
            return False
    return True

@st.cache_data
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
                article = FetchArticle.extract_article_details(text)
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

@st.cache_data
def retrieve_information_online(keywords, link_num=10, extra_info=None):
    articles = []
    max_attempts = 5
    attempts = 0
    while not articles and attempts < max_attempts:
        links = FetchArticle.retrieve_links(keywords, link_num)
        fetched_articles = FetchArticle.extract_many_article_details(links)
        articles.extend(fetched_articles)
        link_num += 1
        attempts += 1
    if extra_info:
        articles.append(extra_info)
    return articles, links

@st.cache_data
def group_individual_article(article):
    rep_sentences = []

    text = article['text']
    sentences = nltk.sent_tokenize(text)
    sentences = [SentenceHolder(text=sent, source=article['source'], author=article['author'], date=article['date']) for sent in sentences]
    if len(sentences) <= 2:
        cluster_articles = [
            {
                "cluster_id": 0,
                "sentences": [sentences[0]] if sentences else [],
                "representative": sentences[0] if sentences else None,
                "representative_with_context": sentences[0] if sentences else None
            },
            {
                "cluster_id": 1,
                "sentences": [sentences[1]] if len(sentences) == 2 else [],
                "representative": sentences[1] if len(sentences) == 2 else None,
                "representative_with_context": sentences[1] if len(sentences) == 2 else None
            }
        ]
    else:
        max_clusters = max(floor(len(sentences) / 6), 8)
        cluster_articles = observe_best_cluster(
            sentences, max_clusters=max_clusters, context=True, context_len=1,
            weights={'single':0.7, 'context':0.3}, preprocess=True, attention=True,
            clustering_method=KMeans, score_weights={'sil':0.9, 'db':0.05, 'ch':0.05}
        )['clusters']
    if cluster_articles:
        for cluster in cluster_articles:
            rep_sentences.append(cluster['representative_with_context'])
    return rep_sentences

@st.cache_data
def group_representative_sentences(rep_sentences:List[SentenceHolder]):
    max_clusters = max(floor(len(rep_sentences) / 6), 10)
    cluster_articles = observe_best_cluster(rep_sentences, max_clusters=max_clusters, 
                                            context=False, context_len=1, weights={'single':0.7, 'context':0.3}, 
                                            preprocess=True, attention=True, clustering_method=KMeans, 
                                            score_weights={'sil':0.9, 'db':0.05, 'ch':0.05})['clusters']
    return cluster_articles

@st.cache_data
def calculate_reliability(clusters:list):
    dates = {}
    for i, cluster in enumerate(clusters):
        dates[i] = [sentence.date for sentence in cluster['sentences'] if sentence.date]
    date_relevancy = calculate_date_relevancy(dates, bounded_range=True)

    for i, cluster in enumerate(clusters):
        reliability_scores = []

        sentences = [sentence.text for sentence in cluster['sentences']]
        objectivity_scores = [(TextBlob(sentence).subjectivity / 2 + 0.75) for sentence in sentences]

        sources = [sentence.source for sentence in cluster['sentences'] if sentence.source]
        source_reliability_scores = [find_bias_rating(source) for source in sources]

        for j in range(len(sentences)):
            reliability_scores.append(source_reliability_scores[j] * objectivity_scores[j])
        
        general_reliability = calculate_general_reliability(reliability_scores)

        cluster['reliability'] = general_reliability * (date_relevancy[i]['relevancy'] - 0.1)
        cluster['sources'] = sources
        cluster['reliability_stats'] = {
            'objectivity': sum(objectivity_scores) / len(objectivity_scores),
            'source_reliability': sum(source_reliability_scores) / len(source_reliability_scores),
            'date_relevancy': date_relevancy[i]['relevancy'],
        }
    
    return clusters

@st.cache_data
def objectify_and_summarize(cluster:dict):
    i = 1
    sentences = [sentence.text for sentence in cluster['sentences']]
    max_input_length = 2**12
    text = ""
    current_length = 0

    for sentence in sentences:
        if current_length + len(sentence) > max_input_length:
            break
        text += sentence + " "
        current_length += len(sentence) + 1
        i += 1

    text = text.strip().replace("\'", "'")

    if i <= 3:
        logger.warning("Warning: Few sentences in cluster. Using default summary lengths.")
        summary_length_min = 50
        summary_length_max = 100
    else:
        summary_length_min = round(len(text) / (i + 3) / 3)
        summary_length_max = round(len(text) / (i - 3) / 3)
    
    summary_length_max = min(summary_length_max, 325)
    summary_length_max = min(summary_length_max, 275)
    
    cluster['summary'] = normalize_text(objectify_text(summarize_text(text, min_length=summary_length_min, max_length=summary_length_max)))

    return cluster

@st.cache_data
def article_analyse(text, link_num=10):
    processed_text = process_text_input_for_keyword(text)
    if not processed_text:
        return None
    

    keywords = processed_text['keywords']
    extra_info = processed_text['extra_info']
    articles, _ = retrieve_information_online(keywords, link_num=link_num, extra_info=extra_info)

    if extra_info: 
        kw_model = KeyBERT()
        keywords_bert = kw_model.extract_keywords(extra_info['text'], keyphrase_ngram_range=(1, 1), stop_words='english', top_n=10)
        keywords_bert = [kw[0] for kw in keywords_bert]
    else:
        keywords_bert = None

    rep_sentences = []
    for article in articles:
        rep_sentences.extend(group_individual_article(article))
        
    if len(rep_sentences) <= 2:
        cluster_articles = [
            {
                "cluster_id": idx,
                "sentences": [rep_sentences[idx]],
                "representative": rep_sentences[idx],
                "representative_with_context": rep_sentences[idx]
            }
            for idx in range(len(rep_sentences))
        ]
    else:
        cluster_articles = group_representative_sentences(rep_sentences)

    valid_clusters = []
    for cluster in cluster_articles:
        if is_cluster_valid(cluster, keywords=keywords_bert, debug=True):
            cluster = objectify_and_summarize(cluster)
            valid_clusters.append(cluster)
    valid_clusters = calculate_reliability(valid_clusters)
    return valid_clusters

@st.cache_data
def visualize_article_analysis(text, link_num=10):
    now = time.time()
    results = article_analyse(text, link_num)

    print(f"{Fore.YELLOW}{Style.BRIGHT}\n=== Article Analysis Results ===")

    for i, cluster in enumerate(results, 1):
        sources = set(sentence.source for sentence in cluster['sentences'] if sentence.source)
        dates = [sentence.date for sentence in cluster['sentences'] if sentence.date]
        parsed_dates = []
        for date_str in dates:
            try:
                parsed_dates.append(datetime.strptime(date_str, '%Y-%m-%d') if isinstance(date_str, str) else date_str)
            except ValueError:
                pass
        if len(parsed_dates) > 1:
            date_range = f"{min(parsed_dates).strftime('%Y-%m-%d')} to {max(parsed_dates).strftime('%Y-%m-%d')}"
        else:
            date_range = parsed_dates[0].strftime('%Y-%m-%d') if parsed_dates else 'N/A'

        print(f"\n{Fore.CYAN}{Style.BRIGHT}Cluster {i}:{Style.RESET_ALL}")
        print(f"{Fore.GREEN}Reliability Score:{Style.RESET_ALL} {cluster['reliability']:.2f}")
        print(f"{Fore.MAGENTA}Summary:{Style.RESET_ALL} {cluster['summary']}")
        print(f"{Fore.BLUE}Representative Sentence:{Style.RESET_ALL} {cluster['representative'].text}")
        print(f"{Fore.GREEN}Sources:{Style.RESET_ALL} {', '.join(sources) if sources else 'N/A'}")
        print(f"{Fore.RED}Date Range:{Style.RESET_ALL} {date_range}")
        print(f"{Fore.YELLOW}{'-' * 80}")

    print(f"\n{Fore.YELLOW}Analysis completed in {time.time() - now:.2f} seconds")



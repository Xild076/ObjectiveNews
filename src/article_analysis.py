import validators
from utility import normalize_url, get_keywords, SentenceHolder, find_bias_rating, normalize_text
from scraper import FetchArticle
from nltk.corpus import stopwords
from math import floor
from grouping import observe_best_cluster
from sklearn.cluster import KMeans
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
import nltk
nltk.download('punkt', quiet=True)

STOPWORDS = set(stopwords.words("english"))
ALPHA_RATIO_REGEX = re.compile(r'[A-Za-z]')
COMMON_TLDS = {
    'com','org','net','edu','gov','mil','int','io','co','us','uk','de','jp',
    'fr','au','ca','cn','ru','ch','it','nl','se','no','es','biz','info','mobi',
    'name','ly','xyz','online','site','tech','store','blog'
}
kw_model = KeyBERT()

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
    representative = cluster.get('representative', '').text
    sentences = cluster.get('sentences', [])

    if not representative or not isinstance(representative, str):
        if debug:
            print("Cluster removed: Representative is missing or not a string.")
        return False
    if not isinstance(sentences, list) or not all(isinstance(s.text, str) for s in sentences):
        if debug:
            print("Cluster removed: Sentences are missing or not all strings.")
        return False
    try:
        lang = detect(representative)
        if lang != 'en':
            if debug:
                print("Cluster removed: Non-English representative sentence.")
            return False
    except LangDetectException:
        if debug:
            print("Cluster removed: Language detection failed.")
        return False
    alpha_ratio = sum(c.isalpha() for c in representative) / (len(representative) + 1)
    if alpha_ratio < alpha_ratio_threshold:
        if debug:
            print(f"Cluster removed: Low alpha ratio ({alpha_ratio:.2f}).")
        return False
    rep_lower = representative.lower()
    if any(phrase in rep_lower for phrase in NON_CONTENT_PHRASES):
        if debug:
            print("Cluster removed: Contains non-content phrase.")
        return False
    if len(sentences) < min_sentences:
        if debug:
            print("Cluster removed: Insufficient number of sentences.")
        return False
    avg_length = sum(len(s.text) for s in sentences) / len(sentences) if sentences else 0.0
    if avg_length < min_avg_sentence_length:
        if debug:
            print(f"Cluster removed: Average sentence length too short ({avg_length:.2f}).")
        return False
    if keywords:
        keyword_set = set(word.lower() for word in keywords)
        sentences_with_keywords = sum(1 for s in sentences if set(re.findall(r'\b\w+\b', s.text.lower())) & keyword_set)
        if sentences_with_keywords < len(sentences) / 2:
            if debug:
                print("Cluster removed: Majority of sentences do not contain the specified keywords.")
            return False
    return True

def process_text_input_for_keyword(text):
    t = text.strip()
    if any(f".{td}" in t for td in COMMON_TLDS):
        if len(t.split()) == 1:
            if not validators.url(t):
                t = normalize_url(t)
            if validators.url(t):
                a = FetchArticle.extract_article_details(t)
                kw = [w for w in a['title'].split() if w.lower() not in STOPWORDS]
                return {"method":0,"keywords":kw,"extra_info":a}
    if len(t.split()) <= 10:
        kw = [w for w in t.split() if w.lower() not in STOPWORDS]
    else:
        kw = get_keywords(t)
    if not kw:
        return None
    return {"method":1,"keywords":kw,"extra_info":None}

def retrieve_information_online(keywords, link_num=10, extra_info=None):
    arts = []
    for _ in range(5):
        links = FetchArticle.retrieve_links(keywords, link_num)
        fetched = FetchArticle.extract_many_article_details(links)
        if fetched:
            arts.extend(fetched)
            break
        link_num += 1
    if extra_info:
        arts.append(extra_info)
    return arts, links

def group_individual_article(article):
    txt = article['text']
    sentences = [SentenceHolder(x, article['source'], article['author'], article['date']) for x in nltk.sent_tokenize(txt)]
    if len(sentences) <= 2:
        return [
            {
                "cluster_id":i,
                "sentences":[s],
                "representative":s,
                "representative_with_context":s
            }
            for i, s in enumerate(sentences)
        ]
    mx = max(floor(len(sentences)/6), 8)
    clusters = observe_best_cluster(
        sentences, mx, {'single':0.7,'context':0.3}, True, 1, True, True, KMeans, {'sil':0.9,'db':0.05,'ch':0.05}
    )['clusters']
    return [cc['representative_with_context'] for cc in clusters] if clusters else []

def group_representative_sentences(rep_sentences):
    mx = max(floor(len(rep_sentences)/6), 10)
    return observe_best_cluster(
        rep_sentences, mx, {'single':0.7,'context':0.3}, False, 1, True, True, KMeans, {'sil':0.9,'db':0.05,'ch':0.05}
    )['clusters']

def calculate_reliability(clusters):
    date_dict = {i: [x.date for x in c['sentences'] if x.date] for i, c in enumerate(clusters)}
    date_relevancy = calculate_date_relevancy(date_dict, True)
    for i, c in enumerate(clusters):
        s_texts = [x.text for x in c['sentences']]
        oscores = [TextBlob(x).subjectivity / 2 + 0.75 for x in s_texts]
        sources = [x.source for x in c['sentences'] if x.source]
        srs = [find_bias_rating(src) for src in sources]
        rs = [srs[j] * oscores[j] for j in range(len(srs))]
        gr = calculate_general_reliability(rs)
        c['reliability'] = gr * (date_relevancy[i]['relevancy'] - 0.1)
        c['sources'] = sources
        c['reliability_stats'] = {
            'objectivity': sum(oscores) / len(oscores) if oscores else 0,
            'source_reliability': sum(srs) / len(srs) if srs else 0,
            'date_relevancy': date_relevancy[i]['relevancy']
        }
    return clusters

def objectify_and_summarize(cluster, light=True):
    s = [x.text for x in cluster['sentences']]
    t = ' '.join(s[:min(len(s), 2**12 // max(len(se) + 1 for se in s))])
    t = t.strip().replace("'", "'")
    if len(s) <= 2:
        slmn, slmx = 50, 100
    else:
        slmn = min(round(len(t)/(len(s)+3)/3), 275)
        slmx = min(round(len(t)/(len(s)-3)/3), 325)
    sm = s[0] if light else summarize_text(t, slmx, slmn)
    cluster['summary'] = normalize_text(objectify_text(sm))
    return cluster

def article_analyse(text, link_num=10):
    p = process_text_input_for_keyword(text)
    
    print("Processing text input...")

    if not p:
        return None
    kw, extra_info = p['keywords'], p['extra_info']
    arts, _ = retrieve_information_online(kw, link_num, extra_info)

    print("Retrieved articles...")

    kb = list(kw_model.extract_keywords(extra_info['text'], keyphrase_ngram_range=(1, 1), stop_words='english', top_n=10)) if extra_info else None
    grouped = [grouped for a in arts for grouped in group_individual_article(a)]

    print("Grouped articles...")

    if len(grouped) <= 2:
        clusters = [
            {
                "cluster_id":i,
                "sentences":[s],
                "representative":s,
                "representative_with_context":s
            }
            for i, s in enumerate(grouped)
        ]
    else:
        clusters = group_representative_sentences(grouped)

    print("Grouped representative sentences...")

    valid_clusters = [objectify_and_summarize(cc, False) for cc in clusters if is_cluster_valid(cc, keywords=kb, debug=False)]

    print("Objectified and summarized clusters...")
    
    return calculate_reliability(valid_clusters) if valid_clusters else []

def visualize_article_analysis(text, link_num=10):
    start = time.time()
    res = article_analyse(text, link_num)
    if not res:
        print(f"{Fore.YELLOW}{Style.BRIGHT}\n=== No Valid Clusters Found ===")
        print(f"\n{Fore.YELLOW}Analysis completed in {time.time()-start:.2f} seconds")
        return
    print(f"{Fore.YELLOW}{Style.BRIGHT}\n=== Article Analysis Results ===")
    for i, c in enumerate(res, 1):
        so = set(x.source for x in c['sentences'] if x.source)
        ds = [x.date for x in c['sentences'] if x.date]
        pdz = []
        for d in ds:
            if isinstance(d, str):
                try:
                    pdz.append(datetime.strptime(d, '%Y-%m-%d'))
                except:
                    pass
            elif isinstance(d, datetime):
                pdz.append(d)
        dr = f"{min(pdz).strftime('%Y-%m-%d')} to {max(pdz).strftime('%Y-%m-%d')}" if len(pdz) > 1 else (pdz[0].strftime('%Y-%m-%d') if pdz else 'N/A')
        print(f"\n{Fore.CYAN}{Style.BRIGHT}Cluster {i}:{Style.RESET_ALL}")
        print(f"{Fore.GREEN}Reliability Score:{Style.RESET_ALL} {c['reliability']:.2f}")
        print(f"{Fore.MAGENTA}Summary:{Style.RESET_ALL} {c['summary']}")
        print(f"{Fore.BLUE}Representative Sentence:{Style.RESET_ALL} {c['representative'].text}")
        print(f"{Fore.GREEN}Sources:{Style.RESET_ALL} {', '.join(so) if so else 'N/A'}")
        print(f"{Fore.RED}Date Range:{Style.RESET_ALL} {dr}")
        print(f"{Fore.YELLOW}{'-'*80}")
    print(f"\n{Fore.YELLOW}Analysis completed in {time.time()-start:.2f} seconds")
import validators
from utility import normalize_url, get_keywords, SentenceHolder, find_bias_rating, normalize_text
from scraper import FetchArticle
from nltk.corpus import stopwords
import nltk
nltk.download('punkt')
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
import streamlit as st

STOPWORDS = set(stopwords.words("english"))
ALPHA_REGEX = re.compile(r'[A-Za-z]')
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
    ALPHA_REGEX = re.compile(r'[A-Za-z]')
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
    alpha_ratio = len(ALPHA_REGEX.findall(representative)) / (len(representative) + 1)
    if alpha_ratio < alpha_ratio_threshold:
        if debug:
            print(f"Cluster removed: Low alpha ratio ({alpha_ratio:.2f}).")
        return False
    representative_lower = representative.lower()
    if any(phrase in representative_lower for phrase in NON_CONTENT_PHRASES):
        if debug:
            print("Cluster removed: Contains non-content phrase.")
        return False
    if len(sentences) < min_sentences:
        if debug:
            print("Cluster removed: Insufficient number of sentences.")
        return False
    total_length = sum(len(sentence.text) for sentence in sentences)
    avg_length = total_length / len(sentences) if sentences else 0.0
    if avg_length < min_avg_sentence_length:
        if debug:
            print(f"Cluster removed: Average sentence length too short ({avg_length:.2f}).")
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
                print("Cluster removed: Majority of sentences do not contain the specified keywords.")
            return False
    return True

def process_text_input_for_keyword(text):
    COMMON_TLDS = [
        'com','org','net','edu','gov','mil','int','io','co','us','uk','de','jp',
        'fr','au','ca','cn','ru','ch','it','nl','se','no','es','biz','info','mobi',
        'name','ly','xyz','online','site','tech','store','blog'
    ]
    m=-1
    kw=None
    a=None
    t=text.strip()
    for td in COMMON_TLDS:
        if "."+td in t:
            if len(" ".split(t))==1:
                if not validators.url(t): t=normalize_url(t)
                if not validators.url(t): return get_keywords(t)
                a=FetchArticle.extract_article_details(t)
                kw=[w for w in a['title'].split() if w not in STOPWORDS]
                m=0
            break
    if kw is None:
        m=1
        if len(" ".split(t))<=10: kw=[w for w in t.split() if w not in STOPWORDS]
        else: kw=get_keywords(t)
    if not kw: return None
    return {"method":m,"keywords":kw,"extra_info":a}

def retrieve_information_online(keywords, link_num=10, extra_info=None):
    arts=[]
    mx=5
    att=0
    while not arts and att<mx:
        l=FetchArticle.retrieve_links(keywords, link_num)
        f=FetchArticle.extract_many_article_details(l)
        arts.extend(f)
        link_num+=1
        att+=1
    if extra_info: arts.append(extra_info)
    return arts,l

def group_individual_article(article):
    r=[]
    txt=article['text']
    s=[SentenceHolder(x,article['source'],article['author'],article['date']) for x in nltk.sent_tokenize(txt)]
    if len(s)<=2:
        c=[
            {
                "cluster_id":0,
                "sentences":[s[0]] if s else [],
                "representative":s[0] if s else None,
                "representative_with_context":s[0] if s else None
            },
            {
                "cluster_id":1,
                "sentences":[s[1]] if len(s)==2 else [],
                "representative":s[1] if len(s)==2 else None,
                "representative_with_context":s[1] if len(s)==2 else None
            }
        ]
    else:
        mx=max(floor(len(s)/6),8)
        c=observe_best_cluster(s, mx, {'single':0.7,'context':0.3}, True, 1, True, True, KMeans, {'sil':0.9,'db':0.05,'ch':0.05})['clusters']
    if c: 
        for cc in c:
            r.append(cc['representative_with_context'])
    return r

def group_representative_sentences(rep_sentences):
    mx=max(floor(len(rep_sentences)/6),10)
    return observe_best_cluster(
        rep_sentences, mx, {'single':0.7,'context':0.3}, False, 1, True, True, KMeans, {'sil':0.9,'db':0.05,'ch':0.05}
    )['clusters']

def calculate_reliability(clusters):
    d={}
    for i,c in enumerate(clusters):
        d[i]=[x.date for x in c['sentences'] if x.date]
    dr=calculate_date_relevancy(d,True)
    for i,c in enumerate(clusters):
        rs=[]
        s=[x.text for x in c['sentences']]
        oscores=[(TextBlob(x).subjectivity/2+0.75) for x in s]
        so=[x.source for x in c['sentences'] if x.source]
        srs=[find_bias_rating(xx) for xx in so]
        for j in range(len(s)):
            rs.append(srs[j]*oscores[j])
        gr=calculate_general_reliability(rs)
        c['reliability']=gr*(dr[i]['relevancy']-0.1)
        c['sources']=so
        c['reliability_stats']={
            'objectivity':sum(oscores)/len(oscores),
            'source_reliability':sum(srs)/len(srs),
            'date_relevancy':dr[i]['relevancy']
        }
    return clusters

def objectify_and_summarize(cluster, light=True):
    i=1
    s=[x.text for x in cluster['sentences']]
    mx=2**12
    t=""
    cl=0
    for se in s:
        if cl+len(se)>mx: break
        t+=se+" "
        cl+=len(se)+1
        i+=1
    t=t.strip().replace("'","'")
    if i<=3:
        slmn=50
        slmx=100
    else:
        slmn=round(len(t)/(i+3)/3)
        slmx=round(len(t)/(i-3)/3)
    slmx=min(slmx,325)
    slmn=min(slmn,275)
    if light: sm=s[0]
    else: sm=summarize_text(t, slmx, slmn)
    cluster['summary']=normalize_text(objectify_text(sm))
    return cluster

def article_analyse(text, link_num=10):
    print(f"Input text: {text}")
    p = process_text_input_for_keyword(text)
    if not p: 
        print("No keywords found, returning None.")
        return None
    kw = p['keywords']
    x = p['extra_info']
    print(f"Keywords: {kw}, Extra info: {x}")
    arts, _ = retrieve_information_online(kw, link_num, x)
    print(f"Retrieved articles: {arts}")
    if x: 
        kb = [k[0] for k in kw_model.extract_keywords(x['text'], (1, 1), 'english', 10)]
        print(f"Extracted keywords from extra info: {kb}")
    else: 
        kb = None
        print("No extra info provided.")
    r = []
    for a in arts:
        grouped = group_individual_article(a)
        print(f"Grouped individual article: {grouped}")
        r.extend(grouped)
    print(f"Grouped sentences: {r}")
    if len(r) <= 2:
        c = [
            {
                "cluster_id": i,
                "sentences": [r[i]],
                "representative": r[i],
                "representative_with_context": r[i]
            }
            for i in range(len(r))
        ]
        print(f"Clusters with <= 2 sentences: {c}")
    else:
        c = group_representative_sentences(r)
        print(f"Clustered representative sentences: {c}")
    v = []
    for cc in c:
        if is_cluster_valid(cc, keywords=kb, debug=True):
            print(f"Valid cluster before summarization")
            cc = objectify_and_summarize(cc, False)
            print(f"Objectified and summarized cluster")
            v.append(cc)
        else:
            print(f"Invalid cluster")
    result = calculate_reliability(v)
    print(f"Calculated reliability: {result}")
    return result

def visualize_article_analysis(text, link_num=10):
    n=time.time()
    res=article_analyse(text, link_num)
    print(f"{Fore.YELLOW}{Style.BRIGHT}\n=== Article Analysis Results ===")
    for i,c in enumerate(res,1):
        so=set(x.source for x in c['sentences'] if x.source)
        ds=[x.date for x in c['sentences'] if x.date]
        pdz=[]
        for d in ds:
            try:
                pdz.append(datetime.strptime(d,'%Y-%m-%d') if isinstance(d,str) else d)
            except: pass
        if len(pdz)>1: dr=f"{min(pdz).strftime('%Y-%m-%d')} to {max(pdz).strftime('%Y-%m-%d')}"
        else: dr=pdz[0].strftime('%Y-%m-%d') if pdz else 'N/A'
        print(f"\n{Fore.CYAN}{Style.BRIGHT}Cluster {i}:{Style.RESET_ALL}")
        print(f"{Fore.GREEN}Reliability Score:{Style.RESET_ALL} {c['reliability']:.2f}")
        print(f"{Fore.MAGENTA}Summary:{Style.RESET_ALL} {c['summary']}")
        print(f"{Fore.BLUE}Representative Sentence:{Style.RESET_ALL} {c['representative'].text}")
        print(f"{Fore.GREEN}Sources:{Style.RESET_ALL} {', '.join(so) if so else 'N/A'}")
        print(f"{Fore.RED}Date Range:{Style.RESET_ALL} {dr}")
        print(f"{Fore.YELLOW}{'-'*80}")
    print(f"\n{Fore.YELLOW}Analysis completed in {time.time()-n:.2f} seconds")


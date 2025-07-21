from summarizer import lexrank, summarize_text
from datetime import datetime
from objectify import objectify_text
from grouping import cluster_texts, observe_best_cluster
from utility import clean_text, normalize_url, get_keywords, SentenceHolder, split_sentences
from typing import Dict, Literal, Optional, List
from scraper import FetchArticle
import validators
from nltk.corpus import stopwords
from tqdm import tqdm
import random
from urllib.parse import urlparse, urlunparse
from math import floor
from collections import defaultdict
from sklearn.cluster import KMeans

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
        rep_sentences.append(cluster['representative_with_context'])
    return rep_sentences

"""def group_representative_sentences(rep_sentences):
    # Ensure reps_with_context are SentenceHolder objects, not strings
    reps_with_context = []
    reps_no_context = []
    for sentence in rep_sentences:
        # sentence[1] may be a SentenceHolder or a string, ensure it's a SentenceHolder
        if hasattr(sentence[1], 'text'):
            reps_with_context.append(sentence[1])
        else:
            # If it's a string, wrap it in a SentenceHolder using the representative's metadata
            rep = sentence[0]
            reps_with_context.append(SentenceHolder(sentence[1], rep.source, rep.author, rep.date))
        reps_no_context.append(sentence[0])
    OPTIMAL_CLUSTERING_PARAMS = {
        "preprocess": True,
        "n_neighbors": 15,
        "min_cluster_size": 2,
        "min_samples": 5,
        "context": False,
        "attention": False,
        "norm": 'l2',
        "n_components": 2,
        "umap_metric": 'cosine',
        "cluster_metric": 'euclidean',
        "algorithm": 'best',
        "cluster_selection_method": 'eom'
    }
    clusters = cluster_texts(reps_with_context, OPTIMAL_CLUSTERING_PARAMS)
    # Build a mapping from text to the no-context SentenceHolder
    text_to_no_context = {s.text: s for s in reps_no_context}
    for cluster in clusters:
        if cluster['representative'] is not None:
            cluster['representative'] = text_to_no_context.get(cluster['representative'].text, cluster['representative'])
        cluster['sentences'] = [text_to_no_context.get(sent.text, sent) for sent in cluster['sentences']]
    return clusters

"""

def group_representative_sentences(rep_sentences: List[SentenceHolder], min_cluster_size: int = 3, min_avg_sim: float = 0.15):
    reps_with_context = []
    reps_no_context = []
    for sentence in rep_sentences:
        if isinstance(sentence, tuple):
            if hasattr(sentence[1], 'text'):
                reps_with_context.append(sentence[1])
            else:
                rep = sentence[0]
                reps_with_context.append(SentenceHolder(sentence[1], rep.source, rep.author, rep.date))
            reps_no_context.append(sentence[0])
        else:
            reps_with_context.append(sentence)
            reps_no_context.append(sentence)

    min_avg_sim = max(min_avg_sim, 0.25)

    OPTIMAL_CLUSTERING_PARAMS = {
        "preprocess": True,
        "n_neighbors": 15,
        "min_cluster_size": min_cluster_size,
        "min_samples": min(5, len(reps_with_context)),
        "context": False,
        "attention": False,
        "norm": 'l2',
        "n_components": 2,
        "umap_metric": 'cosine',
        "cluster_metric": 'euclidean',
        "algorithm": 'best',
        "cluster_selection_method": 'eom'
    }
    clusters = cluster_texts(reps_with_context, OPTIMAL_CLUSTERING_PARAMS)
    text_to_no_context = {s.text: s for s in reps_no_context}

    filtered_clusters = []
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    for cluster in clusters:
        if len(cluster['sentences']) < min_cluster_size:
            continue
        sent_texts = [s.text for s in cluster['sentences']]
        if len(sent_texts) > 1:
            from grouping import model
            embeddings = model.encode(sent_texts, show_progress_bar=False)
            sim_matrix = cosine_similarity(embeddings)
            avg_sim = (sim_matrix.sum() - len(sent_texts)) / (len(sent_texts) * (len(sent_texts) - 1))
        else:
            avg_sim = 1.0
        if avg_sim < min_avg_sim:
            continue

        if cluster['representative'] is not None and len(sent_texts) > 1:
            rep_text = cluster['representative'].text
            try:
                rep_idx = sent_texts.index(rep_text)
            except ValueError:
                rep_idx = 0
            rep_emb = embeddings[rep_idx]
            sim_to_rep = cosine_similarity([rep_emb], embeddings)[0]
            keep_idxs = [i for i, sim in enumerate(sim_to_rep) if sim >= 0.5]
            if len(keep_idxs) < min_cluster_size:
                continue
            cluster['sentences'] = [cluster['sentences'][i] for i in keep_idxs]
            sent_texts = [s.text for s in cluster['sentences']]
            embeddings = [embeddings[i] for i in keep_idxs]

        sent_lens = np.array([len(s.text.split()) for s in cluster['sentences']])
        if len(sent_lens) > 2:
            med = np.median(sent_lens)
            mad = np.median(np.abs(sent_lens - med))
            keep_idxs = [i for i, l in enumerate(sent_lens) if abs(l - med) <= 2 * mad]
            if len(keep_idxs) < min_cluster_size:
                continue
            cluster['sentences'] = [cluster['sentences'][i] for i in keep_idxs]

        if cluster['representative'] is not None:
            cluster['representative'] = text_to_no_context.get(cluster['representative'].text, cluster['representative'])
        cluster['sentences'] = [text_to_no_context.get(sent.text, sent) for sent in cluster['sentences']]
        filtered_clusters.append(cluster)
    return filtered_clusters

def summarize_clusters(clusters, level:Literal['fast', 'medium', 'slow'] = 'fast'):
    import re
    from difflib import SequenceMatcher
    def normalize_sent(s):
        return re.sub(r'\W+', '', s.lower())

    def is_similar(a, b, threshold=0.85):
        return SequenceMatcher(None, a, b).ratio() >= threshold

    for cluster in clusters:
        if level == 'fast':
            summary = cluster['representative'].text
        elif level == 'medium':
            sentences = [sent.text for sent in cluster['sentences'] if sent.text.strip()]
            if not sentences or all(len(s.split()) == 0 for s in sentences):
                summary = cluster['representative'].text
            else:
                try:
                    n = max(1, round(len(sentences) // 3))
                    lexrank_sents = lexrank(" ".join(sentences), n=n, threshold=0.1)
                    # Deduplicate by normalized content and filter near-duplicates
                    normed_lexrank = [normalize_sent(s) for s in lexrank_sents]
                    ordered = []
                    for s in sentences:
                        norm_s = normalize_sent(s)
                        # Only add if it's a lexrank sentence and not too similar to any already added
                        if norm_s in normed_lexrank:
                            if not any(is_similar(norm_s, normalize_sent(o)) for o in ordered):
                                ordered.append(s)
                    # If summary is too short, fill with next most central, non-redundant sentences
                    if len(ordered) < n:
                        for s in sentences:
                            norm_s = normalize_sent(s)
                            if not any(is_similar(norm_s, normalize_sent(o)) for o in ordered):
                                ordered.append(s)
                            if len(ordered) >= n:
                                break
                    summary = " ".join(ordered) if ordered else cluster['representative'].text
                except Exception:
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

import itertools
import sys
from datetime import datetime


def test_clustering_parameters(text: str, level: Literal['fast', 'medium', 'slow'] = 'medium'):
    """
    Performs a curated grid search over an expanded set of clustering parameters.

    This function fetches and processes the articles once, then iterates through various
    parameter combinations for the clustering step. The results of every test run are
    saved to a single, timestamped text file for later analysis.

    Args:
        text (str): The initial keyword, text, or URL to start the analysis.
        level (str, optional): The summarization level. Defaults to 'medium'.
    """
    text = clean_text(text)
    print("[Info] Performing one-time data retrieval...")
    keywords_info = process_text_input_for_keyword(text)
    if not keywords_info:
        print("[Error] No valid keywords found.")
        return

    articles, links = retrieve_information_online(keywords_info['keywords'], extra_info=keywords_info.get('extra_info'))
    if not articles:
        print("[Error] No articles found.")
        return

    print(f"[Info] Retrieved {len(articles)} articles from {len(links)} links.")
    
    grouped_articles = []
    for article in articles:
        rep_sentences = group_individual_articles(article)
        grouped_articles.extend(rep_sentences)

    print(f"[Info] Grouped {len(grouped_articles)} representative sentences.")
    
    output_filename = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    print(f"[Info] Beginning parameter tests. All test output will be saved to: {output_filename}")

    original_stdout = sys.stdout
    with open(output_filename, 'w', encoding='utf-8') as f:
        sys.stdout = f
        try:
            # A curated grid search to test new parameters while keeping the run count manageable.
            param_grid = {
                'n_neighbors': [8, 15, 25],
                'min_samples': [2, 5],
                'n_components': [2, 5],
                'cluster_selection_method': ['eom', 'leaf'],
            }

            keys, values = zip(*param_grid.items())
            test_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

            for i, combo in enumerate(test_combinations):
                # Base parameters, updated by the current combination
                current_params = {
                    "attention": False,
                    "cluster_metric": 'euclidean',
                    "umap_metric": 'cosine',
                    "weights": 1.0,
                    "context": False,
                    "context_len": 2,
                    "preprocess": True,  # Locked in as 'True' based on prior results
                    "norm": 'l2',
                    "algorithm": 'best',
                }
                current_params.update(combo)

                print(f"\n\n{'='*25} TEST RUN {i+1}/{len(test_combinations)} {'='*25}")
                print("PARAMETERS:")
                # Print all parameters for complete reference
                for key, value in sorted(current_params.items()):
                     print(f"  - {key}: {value}")
                print('-'*60)

                try:
                    clusters = cluster_texts(grouped_articles, current_params)
                    summarized_clusters = summarize_clusters(clusters, level)
                    objectified_clusters = objectify_clusters(summarized_clusters)
                    
                    print("\n===== Article Analysis Result =====\n")
                    visualize_article_analysis(objectified_clusters)
                except Exception as e:
                    print(f"\n[ERROR] Test run failed with error: {e}\n")

            print(f"\n\n{'='*25} TESTING COMPLETE {'='*25}")
        
        finally:
            # Ensure stdout is restored even if an error occurs
            sys.stdout = original_stdout
            
    print(f"[Info] Testing complete. Results saved to {output_filename}")

article = article_analysis("Post office", level='medium')
visualize_article_analysis(article)
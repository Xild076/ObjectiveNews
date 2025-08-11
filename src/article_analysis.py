import json
import time
from colorama import Fore, Style
from datetime import datetime
import sys, os
try:
    from grouping.grouping import group_individual_articles, group_representative_sentences, merge_similar_clusters
except Exception:
    try:
        from src.grouping.grouping import group_individual_articles, group_representative_sentences, merge_similar_clusters
    except Exception:
        BASE_DIR = os.path.abspath(os.path.dirname(__file__))
        if BASE_DIR not in sys.path:
            sys.path.append(BASE_DIR)
        from src.grouping.grouping import group_individual_articles, group_representative_sentences, merge_similar_clusters
try:
    from objectify.objectify import objectify_clusters
except Exception:
    from src.objectify.objectify import objectify_clusters
try:
    from summarizer import summarize_clusters
except Exception:
    from src.summarizer import summarize_clusters
try:
    from reliability import calculate_reliability
except Exception:
    from src.reliability import calculate_reliability
try:
    from scraper import process_text_input_for_keyword, retrieve_information_online
except Exception:
    from src.scraper import process_text_input_for_keyword, retrieve_information_online
try:
    from utility import DLog, IS_STREAMLIT
except Exception:
    from src.utility import DLog, IS_STREAMLIT
import os
from typing import List, Dict, Any, Literal, Callable, Optional
from math import floor

logger = DLog(name="ARTICLE_ANALYSIS", level="DEBUG")

def article_analysis(text: str, link_n=5, diverse_links=True, summarize_level:Literal["fast", "medium", "slow"]="fast", progress_callback: Optional[Callable[[float, str], None]] = None):
    total_steps = 7
    def update_progress(step, message):
        if progress_callback:
            progress_callback(step / total_steps, message)

    update_progress(0, "Starting article analysis...")
    logger.info("Starting article analysis...")

    update_progress(1, "Processing text input for keywords...")
    logger.info("Processing text input for keywords...")
    keywords_info = process_text_input_for_keyword(text)
    if not keywords_info:
        logger.error("No valid keywords found.")
        return []
    logger.info(f"Keywords found: {keywords_info['keywords']}")

    update_progress(2, f"Retrieving up to {link_n} articles online...")
    logger.info("Retrieving information online...")
    articles, links = retrieve_information_online(keywords_info['keywords'], link_num=link_n, diverse=diverse_links)
    logger.info(f"Retrieved {len(articles)} articles from {len(links)} links.")
    if not articles:
        logger.error("No articles found.")
        return []
    
    update_progress(3, f"Distilling key sentences from {len(articles)} articles...")
    logger.info("Grouping individual articles into representative sentences...")
    all_representative_sentences = []
    article_source_map = {}
    
    for article in articles:
        reps = group_individual_articles(article)
        source = article.get('source', 'unknown')
        for rep in reps:
            if isinstance(rep, tuple):
                rep_sentence, context_sentence = rep
                if hasattr(rep_sentence, 'source'):
                    rep_sentence.source = source
                if hasattr(context_sentence, 'source'):
                    context_sentence.source = source
            elif hasattr(rep, 'source'):
                rep.source = source
        all_representative_sentences.extend(reps)
        
        if source not in article_source_map:
            article_source_map[source] = []
        article_source_map[source].extend(reps)

    if not all_representative_sentences:
        logger.warning("No representative sentences found after grouping articles.")
        return []
    
    if not IS_STREAMLIT:
        reps_dir = "src/aa"
        os.makedirs(reps_dir, exist_ok=True)
        with open(f"{reps_dir}/reps_{str(datetime.now())}.txt", "w") as f:
            f.write(str(all_representative_sentences))
    
    logger.info(f"Extracted {len(all_representative_sentences)} representative sentences from {len(article_source_map)} unique sources.")

    update_progress(4, f"Grouping {len(all_representative_sentences)} sentences into narratives...")
    logger.info("Grouping representative sentences into clusters...")
    
    filtered_sentences = []
    keyword_set = set(kw.lower() for kw in keywords_info['keywords'])
    
    for sent in all_representative_sentences:
        if isinstance(sent, tuple):
            rep, ctx = sent
            text = getattr(rep, 'text', '').lower()
            text_length = len(text.split())
            
            keyword_matches = sum(1 for kw in keyword_set if kw in text)
            relevance_score = keyword_matches / len(keyword_set) if keyword_set else 0
            
            if text_length >= 8 and text_length <= 50 and relevance_score >= 0.3:
                filtered_sentences.append(sent)
        elif hasattr(sent, 'text'):
            text = sent.text.lower()
            text_length = len(text.split())
            
            keyword_matches = sum(1 for kw in keyword_set if kw in text)
            relevance_score = keyword_matches / len(keyword_set) if keyword_set else 0
            
            if text_length >= 8 and text_length <= 50 and relevance_score >= 0.3:
                filtered_sentences.append(sent)
    
    if not filtered_sentences:
        filtered_sentences = all_representative_sentences[:min(len(all_representative_sentences), 10)]
    
    logger.info(f"Filtered to {len(filtered_sentences)} relevant sentences from original {len(all_representative_sentences)}")
    
    source_sentence_map = {}
    for sent in filtered_sentences:
        if isinstance(sent, tuple):
            source = getattr(sent[0], 'source', 'unknown')
        else:
            source = getattr(sent, 'source', 'unknown')
        
        if source not in source_sentence_map:
            source_sentence_map[source] = []
        source_sentence_map[source].append(sent)
    
    if len(source_sentence_map) < 2:
        logger.warning(f"Only {len(source_sentence_map)} unique sources found after filtering. Analysis may be limited.")
        
    min_cluster_size = max(2, min(len(source_sentence_map), 4))
    # weights,context,context_len,preprocess,norm,n_neighbors,n_components,umap_metric,cluster_metric,algorithm,cluster_selection_method,value
    # 1.0, False, 1, False,l1,10,5,correlation,manhattan,prims_kdtree,eom
    PARAMS = {
        "weights": 1.0,
        "context": False,
        "context_len": 1,
        "preprocess": False,
        "norm": "l1",
        "n_neighbors": 10,
        "n_components": 5,
        "umap_metric": "correlation",
        "cluster_metric": "manhattan",
        "algorithm": "prims_kdtree",
        "cluster_selection_method": "eom",
    }
    clusters = group_representative_sentences(filtered_sentences, min_cluster_size=min_cluster_size, params=PARAMS)
    raw_clusters = list(clusters)
    
    final_clusters = []
    for cluster in clusters:
        cluster_sources = set()
        for sent in cluster.get('sentences', []):
            source = getattr(sent, 'source', 'unknown')
            cluster_sources.add(source)
        # require at least 2 unique sources to consider this a robust narrative
        if len(cluster_sources) >= 2:
            final_clusters.append(cluster)
    
    clusters = final_clusters
    if not clusters:
        logger.warning("No multi-source clusters found; falling back to best single-source cluster.")
        if raw_clusters:
            largest = max(raw_clusters, key=lambda c: len(c.get('sentences', [])) if isinstance(c, dict) else 0)
            clusters = [largest]
            clusters[0]['meta'] = clusters[0].get('meta', {})
            clusters[0]['meta']['single_source_fallback'] = True
        else:
            logger.warning("No clusters produced at all.")
            return []
    clusters = merge_similar_clusters(clusters, threshold=0.74)
    logger.info(f"Grouped representative sentences into {len(clusters)} clusters.")

    update_progress(5, f"Summarizing {len(clusters)} narratives...")
    logger.info("Summarizing clusters...")
    try:
        summarized_clusters = summarize_clusters(clusters, level=summarize_level)
    except Exception as e:
        logger.error(f"Summarization failed: {e}")
        summarized_clusters = [{**c, 'summary': c.get('representative', None).text if c.get('representative') else ''} for c in clusters]
    if not summarized_clusters:
        logger.warning("No summarized clusters found.")
        summarized_clusters = clusters
    logger.info(f"Summarized {len(summarized_clusters)} clusters.")

    update_progress(6, "Making summaries objective...")
    logger.info("Objectifying clusters...")
    try:
        objectified_clusters = objectify_clusters(summarized_clusters)
    except Exception as e:
        logger.error(f"Objectification failed: {e}")
        objectified_clusters = summarized_clusters
    if not objectified_clusters:
        logger.warning("No objectified clusters found.")
        objectified_clusters = summarized_clusters
    logger.info(f"Objectified {len(objectified_clusters)} clusters.")

    update_progress(7, "Calculating reliability scores...")
    logger.info("Calculating reliability for clusters...")
    reliability_clusters = calculate_reliability(objectified_clusters)
    if not reliability_clusters:
        logger.warning("No reliability clusters found.")
        return []
    logger.info(f"Reliability clusters calculated.")
    
    update_progress(total_steps, "Analysis complete.")
    return reliability_clusters

def visualize_article_analysis(analysis_result) -> None:
    logger.info("Visualizing article analysis...")
    if not analysis_result:
        logger.warning("No analysis result to visualize.")
        return
    # Avoid unnecessary delays in Streamlit runtime
    # time.sleep(1)
    print("\n" + f"{Style.BRIGHT}{Fore.CYAN}===== ARTICLE ANALYSIS RESULT ====={Style.RESET_ALL}\n")
    for i, cluster in enumerate(analysis_result):
        print(f"{Style.BRIGHT}{Fore.GREEN}CLUSTER {i+1}{Style.RESET_ALL} {Fore.LIGHTBLACK_EX}| Reliability: {Fore.YELLOW}{cluster.get('reliability', 'N/A'):.2f}{Fore.RESET}")
        print(f"{Fore.BLUE}Summary:{Fore.RESET} {Style.BRIGHT}{cluster.get('summary', '')}{Style.RESET_ALL}")
        print(f"{Fore.LIGHTYELLOW_EX}Key Sentences:{Fore.RESET}")
        sents = cluster.get('sentences', [])
        count = 0
        for sent in sents:
            if count >= 3:
                break
            src = getattr(sent, 'source', '') or 'N/A'
            author = getattr(sent, 'author', '') or 'N/A'
            date = getattr(sent, 'date', '') or 'N/A'
            print(f"{Fore.WHITE}  - {sent.text}{Fore.LIGHTBLACK_EX} (Source: {src}, Author: {author}, Date: {date}){Fore.RESET}")
            count += 1
        if count == 0:
            print(f"{Fore.RED}  (No key sentences available){Fore.RESET}")
        print(f"{Fore.MAGENTA}Reliability Score:{Fore.RESET} {Style.BRIGHT}{cluster.get('reliability', 'N/A'):.2f}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Reliability Details:{Fore.RESET}")
        details = cluster.get('reliability_details', {})
        if not details:
            print(f"{Fore.RED}  (No reliability details available){Fore.RESET}")
        else:
            maxlen = max((len(str(k)) for k in details), default=0)
            for detail in sorted(details):
                val = details[detail]
                print(f"  {detail.replace('_',' ').capitalize():<{maxlen}} : {val}")
        print(f"{Fore.LIGHTBLACK_EX}" + "-"*60 + f"{Fore.RESET}\n")
    print(f"{Style.BRIGHT}{Fore.CYAN}===== END OF ANALYSIS ====={Style.RESET_ALL}\n")

# No top-level execution on import; functions are called by CLI/UI only.
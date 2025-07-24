import time
from colorama import Fore, Style
from grouping.grouping import group_individual_articles, group_representative_sentences
from objectify.objectify import objectify_clusters
from summarizer import summarize_clusters
from reliability import calculate_reliability
from scraper import process_text_input_for_keyword, retrieve_information_online
from utility import DLog
from typing import List, Dict, Any, Literal, Callable, Optional

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
    grouped_articles = [sent for article in articles for sent in group_individual_articles(article)]
    if not grouped_articles:
        logger.warning("No representative sentences found after grouping articles.")
        return []
    logger.info(f"Grouped {len(grouped_articles)} representative sentences from articles.")

    update_progress(4, f"Grouping {len(grouped_articles)} sentences into narratives...")
    logger.info("Grouping representative sentences into clusters...")
    clusters = group_representative_sentences(grouped_articles)
    if not clusters:
        logger.warning("No clusters found after grouping representative sentences.")
        return []
    logger.info(f"Grouped representative sentences into {len(clusters)} clusters.")

    update_progress(5, f"Summarizing {len(clusters)} narratives...")
    logger.info("Summarizing clusters...")
    summarized_clusters = summarize_clusters(clusters, level=summarize_level)
    if not summarized_clusters:
        logger.warning("No summarized clusters found.")
        return []
    logger.info(f"Summarized {len(summarized_clusters)} clusters.")

    update_progress(6, "Making summaries objective...")
    logger.info("Objectifying clusters...")
    objectified_clusters = objectify_clusters(summarized_clusters)
    if not objectified_clusters:
        logger.warning("No objectified clusters found.")
        return []
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
    time.sleep(1)
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
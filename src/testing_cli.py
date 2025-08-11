from typing import List, Dict, Optional

import argparse
import sys
import os
import subprocess
from colorama import Fore, Style
from article_analysis import article_analysis, visualize_article_analysis
from utility import DLog

logger = DLog(name="CLI")


def run_app():
    logger.info("Launching Streamlit app...")
    subprocess.run(["streamlit", "run", "app.py"], cwd=os.path.dirname(__file__))

def run_aa(text, link_n=5, diverse_links=True, summarize_level="fast"):
    logger.info("Running article analysis...")
    articles = article_analysis(text, link_n=link_n, diverse_links=diverse_links, summarize_level=summarize_level)
    visualize_article_analysis(articles)
    logger.info("Article analysis complete.")

def main():
    parser = argparse.ArgumentParser(description="Stock Prediction Model CLI")
    parser.add_argument('command', choices=['app', 'aa'], help="Command to execute")
    parser.add_argument('--text', type=str, help="Text input for article analysis")
    parser.add_argument('--link_n', type=int, default=5, help="Number of articles to fetch")
    parser.add_argument('--diverse_links', action='store_true', help="Use diverse sources for article retrieval")
    parser.add_argument('--summarize_level', type=str, choices=['fast', 'medium', 'slow'], default='fast', help="Level of summarization detail")
    args = parser.parse_args()

    if args.command == 'app':
        run_app()
    elif args.command == 'aa':
        run_aa(args.text, link_n=args.link_n, diverse_links=args.diverse_links, summarize_level=args.summarize_level)

if __name__ == "__main__":
    main()
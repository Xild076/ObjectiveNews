import functools
from keybert import KeyBERT
from urllib.parse import urlparse, urlunparse
from datetime import datetime
import re, spacy
from colorama import init, Fore, Style
import traceback
import os
from sentence_transformers import SentenceTransformer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import inflect

LEVEL_COLORS = {
    "DEBUG": Fore.CYAN,
    "INFO": Fore.GREEN,
    "WARNING": Fore.YELLOW,
    "ERROR": Fore.RED,
    "CRITICAL": Fore.RED + Style.BRIGHT
}

import functools
class DLog:
    def __init__(self, name="DLog", level="DEBUG", log_dir="logs"):
        init(autoreset=True)
        self.name = name
        self.levels = {"DEBUG":10,"INFO":20,"WARNING":30,"ERROR":40,"CRITICAL":50}
        self.log_level = self.levels.get(level.upper(),10)
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.current_date = datetime.now().strftime("%Y-%m-%d")
        self.log_path = os.path.join(self.log_dir,f"{self.current_date}.log")
        self.error_log_path = os.path.join(self.log_dir,f"errors_{self.current_date}.log")
        self.log_file = open(self.log_path,"a")
        self.error_file = open(self.error_log_path,"a")
    
    def _check_date(self):
        d = datetime.now().strftime("%Y-%m-%d")
        if d != self.current_date:
            self.log_file.close()
            self.error_file.close()
            self.current_date = d
            self.log_path = os.path.join(self.log_dir,f"{self.current_date}.log")
            self.error_log_path = os.path.join(self.log_dir,f"errors_{self.current_date}.log")
            self.log_file = open(self.log_path,"a")
            self.error_file = open(self.error_log_path,"a")
    
    def _log(self, level_str, msg):
        if self.levels[level_str]>=self.log_level:
            t=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            c=LEVEL_COLORS.get(level_str,"")
            out=f"{t} {Fore.RESET} [{self.name}] {c} [{level_str}] {Fore.RESET} {msg}"
            print(f"{out}")
            self.log_file.write(out+"\n")
            if self.levels[level_str]>=self.levels["ERROR"]:
                self.error_file.write(out+"\n")
            self.log_file.flush()
            self.error_file.flush()
    
    def debug(self,msg):
        self._check_date()
        self._log("DEBUG",msg)
    def info(self,msg):
        self._check_date()
        self._log("INFO",msg)
    def warning(self,msg):
        self._check_date()
        self._log("WARNING",msg)
    def error(self,msg):
        self._check_date()
        self._log("ERROR",msg)
    def critical(self,msg):
        self._check_date()
        self._log("CRITICAL",msg)
    def exception(self,msg):
        self._check_date()
        tb=traceback.format_exc()
        self._log("ERROR",f"{msg}\n{tb}")
    
    def __del__(self):
        self.log_file.close()
        self.error_file.close()

logger = DLog(name="UtilityLogger", level="DEBUG")

try:
    import streamlit as st
    IS_STREAMLIT = True
    cache_resource_decorator = st.cache_resource
    cache_data_decorator = st.cache_data
    logger.info("Streamlit is available. Using st.cache_resource and st.cache_data.")

except ImportError:
    IS_STREAMLIT = False
    cache_resource_decorator = functools.lru_cache(maxsize=1)
    cache_data_decorator = functools.lru_cache(maxsize=128)
    logger.info("Streamlit not available. Falling back to functools.lru_cache.")


@cache_resource_decorator
def load_nlp_ecws():
    logger.info("Loading spaCy model 'en_core_web_sm...'")
    return spacy.load("en_core_web_sm")

@cache_resource_decorator
def load_sent_transformer():
    logger.info("Loading SentenceTransformer model 'all-MiniLM-L6-v2...'")
    return SentenceTransformer("all-MiniLM-L6-v2")

@cache_resource_decorator
def load_lemma():
    logger.info("Loading WordNetLemmatizer...")
    return WordNetLemmatizer()

@cache_resource_decorator
def load_keybert():
    logger.info("Loading KeyBERT model...")
    return KeyBERT()

@cache_resource_decorator
def load_inflect():
    logger.info("Loading inflect engine...")
    return inflect.engine()

@cache_data_decorator
def clean_text(text: str) -> str:
    logger.info("Cleaning text...")
    if not text:
        return ""
    
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    
    text = re.sub(r'^\s*\d{1,2}:\d{2}(\s*[AP]M)?.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^(Updated|Published|By|Source):.*$', '', text, flags=re.IGNORECASE | re.MULTILINE)
    
    text = re.sub(r'Follow us on.*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Click here to.*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Read more:.*', '', text, flags=re.IGNORECASE)

    text = re.sub(r'^\s*[-*•–]\s*', '', text, flags=re.MULTILINE)
    
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

@cache_data_decorator
def split_sentences(text: str):
    logger.info("Splitting text into sentences...")
    text = clean_text(text)
    if not text:
        return []
    nlp = load_nlp_ecws()
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if len(sent.text.strip().split()) > 4 and len(sent.text.strip()) > 25]

@cache_data_decorator
def normalize_url(url:str, default_scheme='https', add_www=True):
    logger.info("Normalizing URL...")
    url = url.strip()
    
    parsed = urlparse(url)
    
    if not parsed.scheme:
        url = f"{default_scheme}://{url}"
        parsed = urlparse(url)
    
    netloc_parts = parsed.netloc.split('.')
    
    if add_www:
        if not netloc_parts[0].lower().startswith('www'):
            parsed = parsed._replace(netloc='www.' + parsed.netloc)
    
    normalized_url = urlunparse(parsed)
    
    return normalized_url

@cache_data_decorator
def get_keywords(text):
    logger.info("Extracting keywords from text...")
    kw_model = load_keybert()
    keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 2),
        stop_words='english',
        top_n=3,
        use_mmr=True,
        diversity=0.7
    )
    seen = set()
    return [word for keyword, _ in keywords for word in keyword.lower().split() if len(word) > 3 and not (word in seen or seen.add(word))]

@cache_resource_decorator
def get_stopwords():
    logger.info("Loading stopwords...")
    return list(set(stopwords.words("english")))

@cache_data_decorator
def normalize_values_minmax(values, reverse=False):
    logger.info("Normalizing values using min-max scaling...")
    mn = min(values)
    mx = max(values)
    if mn == mx:
        return [0.5] * len(values)
    rng = mx - mn
    if not reverse:
        return [(v - mn) / rng for v in values]
    else:
        return [(mx - v) / rng for v in values]
class SentenceHolder(object):
    def __init__(self, text, source=None, author=None, date=None):
        self.text = clean_text(text)
        self.source = source
        self.author = author
        self.date = date
    
    def __repr__(self):
        return f"SentH[text={self.text}]"
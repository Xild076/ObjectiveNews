import functools
from keybert import KeyBERT
from urllib.parse import urlparse, urlunparse
from datetime import datetime
import re, spacy
from colorama import init, Fore, Style
import traceback
import os
import sys
import logging
from sentence_transformers import SentenceTransformer
import inflect

LEVEL_COLORS = {"DEBUG": Fore.CYAN, "INFO": Fore.GREEN, "WARNING": Fore.YELLOW, "ERROR": Fore.RED, "CRITICAL": Fore.RED + Style.BRIGHT}

class DLog:
    def __init__(self, name="DLog", level="DEBUG", log_dir="logs", quiet=None):
        init(autoreset=True)
        self.name = name
        self.levels = {"DEBUG":10,"INFO":20,"WARNING":30,"ERROR":40,"CRITICAL":50}
        env_level = os.environ.get("DLOG_LEVEL")
        if env_level and env_level.upper() in self.levels:
            level = env_level.upper()
        self.log_level = self.levels.get(level.upper(),10)
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.current_date = datetime.now().strftime("%Y-%m-%d")
        self.log_path = os.path.join(self.log_dir,f"{self.current_date}.log")
        self.error_log_path = os.path.join(self.log_dir,f"errors_{self.current_date}.log")
        self.log_file = open(self.log_path,"a")
        self.error_file = open(self.error_log_path,"a")
        if quiet is None:
            self.quiet = False
        else:
            self.quiet = bool(quiet)
        if not logging.getLogger().handlers:
            logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        self._py_levels = {"DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARNING": logging.WARNING, "ERROR": logging.ERROR, "CRITICAL": logging.CRITICAL}
    
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
            if not self.quiet:
                print(f"{out}")
            self.log_file.write(out+"\n")
            if self.levels[level_str]>=self.levels["ERROR"]:
                self.error_file.write(out+"\n")
            self.log_file.flush()
            self.error_file.flush()
            try:
                logging.getLogger(self.name).log(self._py_levels.get(level_str, logging.INFO), str(msg))
            except Exception:
                pass

    def set_level(self, level:str):
        lvl = self.levels.get(level.upper())
        if lvl is not None:
            self.log_level = lvl

    def set_quiet(self, quiet:bool=True):
        self.quiet = bool(quiet)
    
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
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx  # type: ignore
        _st_ctx = get_script_run_ctx()
    except Exception:
        _st_ctx = None
    if _st_ctx is not None:
        IS_STREAMLIT = True
        cache_resource_decorator = st.cache_resource
        cache_data_decorator = st.cache_data
    else:
        raise ImportError("Streamlit not running")
except ImportError:
    IS_STREAMLIT = False
    cache_resource_decorator = functools.lru_cache(maxsize=1)
    cache_data_decorator = functools.lru_cache(maxsize=128)

# Ensure project paths are importable in both local and Streamlit runtimes
try:
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    SRC_DIR = os.path.abspath(os.path.dirname(__file__))
    if ROOT_DIR not in sys.path:
        sys.path.append(ROOT_DIR)
    if SRC_DIR not in sys.path:
        sys.path.append(SRC_DIR)
except Exception:
    pass


def _import_nltk_safely():
    import importlib, sys
    if 'nltk' in sys.modules and not hasattr(sys.modules['nltk'], 'data'):
        del sys.modules['nltk']
    return importlib.import_module('nltk')

def ensure_nltk_data():
    nltk = _import_nltk_safely()
    try:
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        data_dir = os.path.join(root, 'nltk_data')
        os.makedirs(data_dir, exist_ok=True)
        if data_dir not in nltk.data.path:
            nltk.data.path.insert(0, data_dir)
    except Exception:
        data_dir = None
    to_check = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/sentiwordnet", "sentiwordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]
    for path, pkg in to_check:
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                logger.info(f"Downloading NLTK package: {pkg}")
                if data_dir:
                    nltk.download(pkg, download_dir=data_dir, quiet=True)
                else:
                    nltk.download(pkg, quiet=True)
            except Exception as e:
                logger.warning(f"Failed to download NLTK package {pkg}: {e}")


@cache_resource_decorator
def load_nlp_ecws():
    logger.info("Loading spaCy NLP pipeline")
    try:
        return spacy.load("en_core_web_sm")
    except Exception:
        try:
            nlp = spacy.blank("en")
            if "sentencizer" not in nlp.pipe_names:
                nlp.add_pipe("sentencizer")
            logger.warning("Using spaCy blank('en') with sentencizer fallback")
            return nlp
        except Exception:
            raise

@cache_resource_decorator
def load_sent_transformer():
    logger.info("Loading SentenceTransformer model 'all-MiniLM-L6-v2...'")
    return SentenceTransformer("all-MiniLM-L6-v2")

# Lightweight per-sentence embedding cache to avoid repeated encodes
_SBERT_CACHE: dict[str, list[float]] = {}
_SBERT_CACHE_MAX = 2000 if IS_STREAMLIT else 8000

def encode_sentences_cached(texts: list[str]):
    model = load_sent_transformer()
    to_encode = []
    indices = []
    for i, t in enumerate(texts):
        if t not in _SBERT_CACHE:
            to_encode.append(t)
            indices.append(i)
    if to_encode:
        new_embs = model.encode(to_encode, show_progress_bar=False)
        for j, emb in enumerate(new_embs):
            _SBERT_CACHE[to_encode[j]] = emb.tolist()
        if len(_SBERT_CACHE) > _SBERT_CACHE_MAX:
            k = len(_SBERT_CACHE) - _SBERT_CACHE_MAX
            for _ in range(k):
                try:
                    _SBERT_CACHE.pop(next(iter(_SBERT_CACHE)))
                except Exception:
                    break
    out = []
    for t in texts:
        e = _SBERT_CACHE.get(t)
        if e is None:
            e = model.encode([t], show_progress_bar=False)[0].tolist()
            _SBERT_CACHE[t] = e
        out.append(e)
    import numpy as _np
    return _np.array(out)

@cache_resource_decorator
def load_lemma():
    logger.info("Loading WordNetLemmatizer...")
    try:
        ensure_nltk_data()
        from nltk.stem import WordNetLemmatizer as _WNL
        return _WNL()
    except Exception:
        class _IdentityLemma:
            def lemmatize(self, w):
                return w
        logger.warning("Falling back to identity lemmatizer (WordNet unavailable).")
        return _IdentityLemma()

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
    try:
        ensure_nltk_data()
        from nltk.corpus import stopwords as _stops
        return list(set(_stops.words("english")))
    except Exception:
        fallback = {
            'the','a','an','and','or','but','if','while','with','to','from','by','on','in','for','of','at','as','is','are','was','were','be','been','being','it','this','that','these','those','i','you','he','she','they','we','them','us','our','your','his','her','their'
        }
        logger.warning("Falling back to built-in stopwords list (NLTK stopwords unavailable).")
        return list(fallback)

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
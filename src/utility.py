import os
from datetime import datetime
import traceback
from colorama import Fore, Style, init
from urllib.parse import urlparse, urlunparse
from keybert import KeyBERT
import pandas as pd
import nltk
import streamlit as st
from nltk.corpus import stopwords, wordnet as wn
from nltk.stem import WordNetLemmatizer
from typing import List, Dict, Any
import re

nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('stopwords', quiet=True)

@st.cache_resource
def get_stopwords():
    return set(stopwords.words("english"))

lemmatizer = WordNetLemmatizer()

def preprocess_text(text: str, language: str = 'english') -> str:
    from nltk.tokenize import word_tokenize
    s = get_stopwords()
    t = re.sub(r'[\n\t\r]', ' ', text.lower())
    tk = [lemmatizer.lemmatize(w) for w in word_tokenize(t) if w not in s]
    return ' '.join(tk)

def normalize_values_minmax(values: List[float], reverse: bool = False) -> List[float]:
    mn, mx = min(values), max(values)
    if mn == mx:
        return [0.5] * len(values)
    r = mx - mn
    return [(mx - v)/r if reverse else (v - mn)/r for v in values]

def dictionary_pos_to_wordnet(pos_str: str) -> str:
    return {"noun": "n", "verb": "v", "adverb": "r", "adjective": "a"}.get(pos_str.lower(), "s" if "satellite" in pos_str.lower() else None)

def get_pos_full_text(pos_str: str) -> str:
    return {"VERB": "verb", "ADJ": "adjective", "ADV": "adverb"}.get(pos_str, None)

def normalize_text(text: str) -> str:
    replacements = [
        (" ,", ","), (" .", "."), (" !", "!"), (" ?","?"), (" ;",";"), (" :", "."),
        (" %","%"), (" *","*"), (" n't","n't"), (" n’t","n’t"), (" 's","'s"),
        (" ’s","’s"), (" 'm","'m"), (" ’m","’m"), (" ’","’"), ("( ","("),
        ("[ ","["), ("{ ","{"), (" )",")"), (" ]","]"), (" }","}"), ("$ ","$"),
        ("# ","#"), (" – ","–"), (" - ","-"), (" -","-"), (" –","–")
    ]
    for a, b in replacements:
        text = text.replace(a, b)
    if len(text) > 3 and text[-1] in {'"', '”'} and text[-2] not in {".", "!", "?"}:
        text += "."
    return text.capitalize() if text else text

def fix_space_newline(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()

def normalize_url(url: str, default_scheme: str = 'https', add_www: bool = True) -> str:
    url = url.strip()
    p = urlparse(url)
    if not p.scheme:
        url = f"{default_scheme}://{url}"
        p = urlparse(url)
    nl = p.netloc.split('.')
    if add_www and not nl[0].lower().startswith('www'):
        p = p._replace(netloc='www.' + p.netloc)
    return urlunparse(p)

@st.cache_resource
def load_kw_model():
    return KeyBERT()

@st.cache_resource
def load_bias_data():
    return pd.read_csv('data/bias.csv')

def get_keywords(text: str) -> List[str]:
    m = load_kw_model()
    kw = m.extract_keywords(text, keyphrase_ngram_range=(1,2), stop_words='english', top_n=3, use_mmr=True, diversity=0.7)
    seen = set()
    return [w for k, _ in kw for w in k.lower().split() if len(w) > 3 and not (w in seen or seen.add(w))]

def find_bias_rating(url: str) -> float:
    df = load_bias_data()
    r = df[df['url'].str.contains(url, na=False)]
    if not r.empty:
        b, a = r.iloc[0]['bias_rating'], r.iloc[0]['factual_reporting_rating']
        m = 0.5 if a == 'VERY HIGH' else 1 if a == 'HIGH' else 1.25
        return abs(b) * m
    return 30.0

LEVEL_COLORS = {
    "DEBUG": Fore.CYAN,
    "INFO": Fore.GREEN,
    "WARNING": Fore.YELLOW,
    "ERROR": Fore.RED,
    "CRITICAL": Fore.RED + Style.BRIGHT
}

class DLog:
    def __init__(self, name: str = "DLog", level: str = "DEBUG", log_dir: str = "logs"):
        init(autoreset=True)
        self.name = name
        self.levels = {"DEBUG":10, "INFO":20, "WARNING":30, "ERROR":40, "CRITICAL":50}
        self.log_level = self.levels.get(level.upper(), 10)
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.current_date = datetime.now().strftime("%Y-%m-%d")
        self.log_path = os.path.join(log_dir, f"{self.current_date}.log")
        self.error_log_path = os.path.join(log_dir, f"errors_{self.current_date}.log")
        self.log_file = open(self.log_path, "a")
        self.error_file = open(self.error_log_path, "a")

    def _check_date(self):
        d = datetime.now().strftime("%Y-%m-%d")
        if d != self.current_date:
            self.log_file.close()
            self.error_file.close()
            self.current_date = d
            self.log_path = os.path.join(self.log_dir, f"{self.current_date}.log")
            self.error_log_path = os.path.join(self.log_dir, f"errors_{self.current_date}.log")
            self.log_file = open(self.log_path, "a")
            self.error_file = open(self.error_log_path, "a")

    def _log(self, level_str: str, msg: str):
        if self.levels[level_str] >= self.log_level:
            t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            c = LEVEL_COLORS.get(level_str, "")
            o = f"{t} [{self.name}] [{level_str}] {msg}"
            print(f"{c}{o}")
            self.log_file.write(o + "\n")
            if self.levels[level_str] >= self.levels["ERROR"]:
                self.error_file.write(o + "\n")
            self.log_file.flush()
            self.error_file.flush()

    def debug(self, msg: str):
        self._check_date()
        self._log("DEBUG", msg)

    def info(self, msg: str):
        self._check_date()
        self._log("INFO", msg)

    def warning(self, msg: str):
        self._check_date()
        self._log("WARNING", msg)

    def error(self, msg: str):
        self._check_date()
        self._log("ERROR", msg)

    def critical(self, msg: str):
        self._check_date()
        self._log("CRITICAL", msg)

    def exception(self, msg: str):
        self._check_date()
        tb = traceback.format_exc()
        self._log("ERROR", f"{msg}\n{tb}")

    def __del__(self):
        self.log_file.close()
        self.error_file.close()

class SentenceHolder:
    __slots__ = ['text', 'source', 'author', 'date']
    def __init__(self, text: str, source: str = None, author: str = None, date: str = None):
        self.text = fix_space_newline(text)
        self.source = source
        self.author = author
        self.date = date

    def __repr__(self):
        return f"SentH[text={self.text}]"
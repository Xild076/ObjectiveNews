import os
from datetime import datetime
import traceback
from colorama import Fore, Style, init
from urllib.parse import urlparse, urlunparse
from keybert import KeyBERT
import pandas as pd
import nltk
import streamlit as st

nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

@st.cache_resource
def get_stopwords():
    return set(stopwords.words("english"))

lemmatizer = WordNetLemmatizer()

def preprocess_text(text: str, language='english') -> str:
    from nltk.tokenize import word_tokenize
    stop_words = get_stopwords()
    text = text.lower().replace("\n", " ").replace("\t", " ").replace("\r", " ")
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def normalize_values_minmax(values, reverse=False):
    mn = min(values)
    mx = max(values)
    if mn == mx:
        return [0.5] * len(values)
    rng = mx - mn
    if not reverse:
        return [(v - mn) / rng for v in values]
    else:
        return [(mx - v) / rng for v in values]

def dictionary_pos_to_wordnet(pos_str: str) -> str:
    pos_str = pos_str.lower()
    if pos_str == "noun":
        return "n"
    elif pos_str == "verb":
        return "v"
    elif pos_str == "adverb":
        return "r"
    elif pos_str == "adjective":
        return "a"
    elif "satellite" in pos_str:
        return "s"
    return None

def get_pos_full_text(pos_str: str) -> str:
    if pos_str == "VERB":
        return "verb"
    elif pos_str == "ADJ":
        return "adjective"
    elif pos_str == "ADV":
        return "adverb"
    else:
        return None

def normalize_text(text:str) -> str:
    text = text.strip()

    text = text.replace(" ,", ",")
    text = text.replace(" .", ".")
    text = text.replace(" !", "!")
    text = text.replace(" ?", "?")
    text = text.replace(" ;", ";")
    text = text.replace(" :", ":")
    text = text.replace(" %", "%")
    text = text.replace(" *", "*")
    text = text.replace(" n't", "n't")
    text = text.replace(" n’t", "n’t")
    text = text.replace(" 's", "'s")
    text = text.replace(" ’s", "’s")
    text = text.replace(" 'm", "'m")
    text = text.replace(" ’m", "’m")
    text = text.replace(" ’", "’")

    text = text.replace("( ", "(")
    text = text.replace("[ ", "[")
    text = text.replace("{ ", "{")
    text = text.replace(" )", ")")
    text = text.replace(" ]", "]")
    text = text.replace(" }", "}")

    text = text.replace("$ ", "$")
    text = text.replace("# ", "#")

    text = text.replace(" – ", "–")
    text = text.replace(" - ", "-")

    text = text.replace(" -", "-")
    text = text.replace(" –", "–")
    
    if len(text) > 3:
        if text[-1] == '"' or text[-1] == "”":
            if text[-2] not in {".", "!", "?"}:
                text = text + "."
    
    text = text[0].upper() + text[1:]

    return text

def fix_space_newline(text:str):
    text = text.replace("\n", " ")
    text = text.replace("  ", " ")
    text = text.replace("   ", " ")
    return text.strip()

def normalize_url(url:str, default_scheme='https', add_www=True):
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

def get_keywords(text):
    kw_model = KeyBERT()
    keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 2),
        stop_words='english',
        top_n=3,
        use_mmr=True,
        diversity=0.7
    )
    unique_words = []
    seen = set()
    for keyword, _ in keywords:
        for word in keyword.lower().split():
            if len(word) > 3 and word not in seen:
                unique_words.append(word)
                seen.add(word)
    return unique_words

def find_bias_rating(url):
    df = pd.read_csv('data/bias.csv')
    result = df[df['url'].str.contains(url, na=False)]
    if not result.empty:
        bias_rating = result.iloc[0]['bias_rating']
        accuracy = result.iloc[0]['factual_reporting_rating']
        if accuracy == 'VERY HIGH':
            multiplier = 0.5
        elif accuracy == 'HIGH':
            multiplier = 1
        else:
            multiplier = 1.25
        return abs(bias_rating) * multiplier
    else:
        return 30

LEVEL_COLORS = {
    "DEBUG": Fore.CYAN,
    "INFO": Fore.GREEN,
    "WARNING": Fore.YELLOW,
    "ERROR": Fore.RED,
    "CRITICAL": Fore.RED + Style.BRIGHT
}

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
            out=f"{t} [{self.name}] [{level_str}] {msg}"
            print(f"{c}{out}")
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

class SentenceHolder(object):
    def __init__(self, text, source=None, author=None, date=None):
        self.text = fix_space_newline(text)
        self.source = source
        self.author = author
        self.date = date
    
    def __repr__(self):
        return f"SentH[text={self.text}]"
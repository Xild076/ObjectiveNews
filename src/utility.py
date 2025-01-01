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

def preprocess_text(text, language='english'):
    from nltk.tokenize import word_tokenize
    s = get_stopwords()
    t = text.lower().replace("\n"," ").replace("\t"," ").replace("\r"," ")
    tk = word_tokenize(t)
    tk = [lemmatizer.lemmatize(w) for w in tk if w not in s]
    return ' '.join(tk)

def normalize_values_minmax(values, reverse=False):
    mn, mx = min(values), max(values)
    if mn == mx: return [0.5]*len(values)
    r = mx - mn
    if not reverse: return [(v - mn)/r for v in values]
    return [(mx - v)/r for v in values]

def dictionary_pos_to_wordnet(pos_str):
    p = pos_str.lower()
    if p=="noun": return "n"
    elif p=="verb": return "v"
    elif p=="adverb": return "r"
    elif p=="adjective": return "a"
    elif "satellite" in p: return "s"

def get_pos_full_text(pos_str):
    if pos_str=="VERB": return "verb"
    elif pos_str=="ADJ": return "adjective"
    elif pos_str=="ADV": return "adverb"

def normalize_text(text):
    t = text.strip()
    r = [
        (" ,", ","),(" .","."),(" !","!"),(" ?","?"),(" ;",";"),(" :","."),
        (" %","%"),(" *","*"),(" n't","n't"),(" n’t","n’t"),(" 's","'s"),
        (" ’s","’s"),(" 'm","'m"),(" ’m","’m"),(" ’","’"),("( ","("),
        ("[ ","["),("{ ","{"),(" )",")"),(" ]","]"),(" }","}"),("$ ","$"),
        ("# ","#"),(" – ","–"),(" - ","-"),(" -","-"),(" –","–")
    ]
    for a,b in r: t = t.replace(a,b)
    if len(t)>3 and (t[-1]=='"' or t[-1]=='”') and t[-2] not in {".","!","?"}: t+="."
    t = t[0].upper()+t[1:] if t else t
    return t

def fix_space_newline(text):
    t = text.replace("\n"," ").replace("  "," ").replace("   "," ")
    return t.strip()

def normalize_url(url, default_scheme='https', add_www=True):
    url = url.strip()
    p = urlparse(url)
    if not p.scheme:
        url = f"{default_scheme}://{url}"
        p = urlparse(url)
    nl = p.netloc.split('.')
    if add_www and not nl[0].lower().startswith('www'):
        p = p._replace(netloc='www.'+p.netloc)
    return urlunparse(p)

@st.cache_resource
def load_kw_model():
    return KeyBERT()

def get_keywords(text):
    m = load_kw_model()
    kw = m.extract_keywords(text,(1,2),'english',3,True,0.7)
    u, s = [], set()
    for k,_ in kw:
        for w in k.lower().split():
            if len(w)>3 and w not in s:
                u.append(w)
                s.add(w)
    return u

def find_bias_rating(url):
    df = pd.read_csv('data/bias.csv')
    r = df[df['url'].str.contains(url,na=False)]
    if not r.empty:
        b = r.iloc[0]['bias_rating']
        a = r.iloc[0]['factual_reporting_rating']
        if a=='VERY HIGH': m=0.5
        elif a=='HIGH': m=1
        else: m=1.25
        return abs(b)*m
    return 30

LEVEL_COLORS={"DEBUG":Fore.CYAN,"INFO":Fore.GREEN,"WARNING":Fore.YELLOW,"ERROR":Fore.RED,"CRITICAL":Fore.RED+Style.BRIGHT}
class DLog:
    def __init__(self,name="DLog",level="DEBUG",log_dir="logs"):
        init(autoreset=True)
        self.name=name
        self.levels={"DEBUG":10,"INFO":20,"WARNING":30,"ERROR":40,"CRITICAL":50}
        self.log_level=self.levels.get(level.upper(),10)
        self.log_dir=log_dir
        if not os.path.exists(log_dir): os.makedirs(log_dir)
        self.current_date=datetime.now().strftime("%Y-%m-%d")
        self.log_path=os.path.join(self.log_dir,f"{self.current_date}.log")
        self.error_log_path=os.path.join(self.log_dir,f"errors_{self.current_date}.log")
        self.log_file=open(self.log_path,"a")
        self.error_file=open(self.error_log_path,"a")
    def _check_date(self):
        d=datetime.now().strftime("%Y-%m-%d")
        if d!=self.current_date:
            self.log_file.close()
            self.error_file.close()
            self.current_date=d
            self.log_path=os.path.join(self.log_dir,f"{self.current_date}.log")
            self.error_log_path=os.path.join(self.log_dir,f"errors_{self.current_date}.log")
            self.log_file=open(self.log_path,"a")
            self.error_file=open(self.error_log_path,"a")
    def _log(self,level_str,msg):
        if self.levels[level_str]>=self.log_level:
            t=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            c=LEVEL_COLORS.get(level_str,"")
            o=f"{t} [{self.name}] [{level_str}] {msg}"
            print(f"{c}{o}")
            self.log_file.write(o+"\n")
            if self.levels[level_str]>=self.levels["ERROR"]:
                self.error_file.write(o+"\n")
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

class SentenceHolder:
    def __init__(self,text,source=None,author=None,date=None):
        self.text=fix_space_newline(text)
        self.source=source
        self.author=author
        self.date=date
    def __repr__(self):
        return f"SentH[text={self.text}]"
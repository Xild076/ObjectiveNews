from keybert import KeyBERT
from urllib.parse import urlparse, urlunparse
from datetime import datetime
from typing import Dict, Optional

def clean_text(text):
    text = text.replace("\n", " ").replace("\t", " ").replace("\r", " ")
    text = text.replace("  ", " ").strip()
    return text

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

def split_sentences(text):
    abbreviations = {'Mr.', 'Mrs.', 'Dr.', 'Jr.', 'Sr.', 'vs.', 'etc.', 'i.e.', 'e.g.', 'U.S.'}
    opener_to_closer = {'"': '"', '“': '”'}
    closers = {v: k for k, v in opener_to_closer.items()}
    stack = []
    sentences = []
    buffer = []
    i = 0
    n = len(text)

    while i < n:
        ch = text[i]
        buffer.append(ch)

        if ch in opener_to_closer and not stack:
            stack.append(opener_to_closer[ch])
        elif stack and ch == stack[-1]:
            stack.pop()

        if ch in '.?!' and not stack:
            j = i + 1
            while j < n and text[j] in '.?!':
                buffer.append(text[j])
                j += 1

            tail = ''.join(buffer).strip().split()[-1]
            if tail not in abbreviations:
                if j == n or text[j].isspace():
                    sentences.append(''.join(buffer).strip())
                    buffer = []
                    i = j - 1

        i += 1

    rem = ''.join(buffer).strip()
    if rem:
        sentences.append(rem)

    return sentences

class SentenceHolder(object):
    def __init__(self, text, source=None, author=None, date=None):
        self.text = clean_text(text)
        self.source = source
        self.author = author
        self.date = date
    
    def __repr__(self):
        return f"SentH[text={self.text}]"
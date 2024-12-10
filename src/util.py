import spacy
import ssl
import yake
import re
import nltk
from nltk.corpus import stopwords, wordnet, words
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
import pandas as pd
from keybert import KeyBERT
import tqdm
from functools import wraps
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import sentiwordnet as swn
import numpy as np
import inflect
from pyinflect import getInflection


ssl._create_default_https_context = ssl._create_unverified_context

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words('english'))
nlp = spacy.load('en_core_web_sm')
english_vocab = set(w.lower() for w in words.words())

lemmatizer = WordNetLemmatizer()

def split_paragraph(text):
    """
    Functions: splits paragraphs into lists

    def split_paragraph(text)
    - text: the full paragraph text

    return re.split(r'\n{2,}', text)
    """
    return re.split(r'\n{2,}', text)

def with_progress_bar(desc=None, **tqdm_kwargs):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs_inner):
            result = func(*args, **kwargs_inner)
            bar_desc = desc if desc else func.__name__
            if hasattr(result, '__iter__') and not isinstance(result, (str, bytes)):
                iterable = tqdm(result, desc=bar_desc, **tqdm_kwargs)
                return iterable
            else:
                return result
        return wrapper
    return decorator

def get_keywords(text):
    """
    Extracts and filters keywords from the provided text using KeyBERT.

    Parameters:
    - text (str): The input text from which to extract keywords.

    Returns:
    - list: A list of filtered, unique, non-overlapping keywords in lowercase.
    """
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

def preprocess_text(text):
    """
    Function: to process text for easy analysis

    def preprocess_text(text)
    - text: full text

    return paragraph.strip()
    - paragraph: the preprocessed paragraphs
    """
    processed_sentences = []
    sentences = sent_tokenize(text)
    for sentence in sentences:
        words = word_tokenize(sentence)
        processed_sentences.append([lemmatizer.lemmatize(word) for word in words if word not in stop_words])
    
    paragraph = ""
    for sentence in processed_sentences:
        formatted_sentence = ""
        for i, token in enumerate(sentence):
            if i > 0 and token not in {",", ".", "!", "?"}:
                formatted_sentence += " "
            formatted_sentence += token
        formatted_sentence = formatted_sentence.capitalize()
        paragraph += formatted_sentence + (" " if formatted_sentence[-1] in {".", "!", "?"} else ". ")
    return paragraph.strip()    

def find_bias_rating(url):
    """
    Function: retrieves the bias of a certain source

    def find_bias_rating(url)
    - url: the base url of a source

    return abs(bias_rating) * multiplier
    """
    df = pd.read_csv('data/bias.csv')
    result = df[df['url'] == url]
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
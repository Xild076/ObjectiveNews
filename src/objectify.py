import logging
import re
import os
import requests
from functools import lru_cache

import spacy
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from rich import print

from synonym import get_synonyms

logging.basicConfig(level=logging.CRITICAL)
nlp = spacy.load("en_core_web_sm")

def simple_tokenizer(text: str) -> list[str]:
    return re.findall(r'\b\w+\b', text.lower())

@lru_cache(maxsize=2048)
def get_objectivity_from_swn(word: str) -> float:
    obj_scores = [s.obj_score() for s in swn.senti_synsets(word.lower())]
    return sum(obj_scores) / len(obj_scores) if obj_scores else 0.5

def get_objectivity_from_textblob(sentence: str) -> float:
    return 1.0 - TextBlob(sentence).subjectivity

try:
    vader_analyzer = SentimentIntensityAnalyzer()
    def get_objectivity_from_vader(sentence: str) -> float:
        return vader_analyzer.polarity_scores(sentence)['neu']
except Exception:
    def get_objectivity_from_vader(sentence: str) -> float:
        raise RuntimeError("VADER is not available.")

MPQA_LEXICON_PATH = "subjclueslen1-all.tff"
MPQA_LEXICON_URL = "http://mpqa.cs.pitt.edu/lexicons/subj_lexicon/subjclueslen1-all.tff"

def _download_mpqa_lexicon():
    if not os.path.exists(MPQA_LEXICON_PATH):
        try:
            response = requests.get(MPQA_LEXICON_URL)
            response.raise_for_status()
            with open(MPQA_LEXICON_PATH, 'w') as f:
                f.write(response.text)
        except requests.exceptions.RequestException:
            return False
    return True

@lru_cache(maxsize=1)
def _load_mpqa_lexicon() -> dict:
    lexicon = {}
    if not _download_mpqa_lexicon():
        return lexicon
    try:
        with open(MPQA_LEXICON_PATH, 'r') as f:
            for line in f:
                parts = dict(item.split('=') for item in line.strip().split())
                word = parts['word1']
                polarity = 'objective' if parts['priorpolarity'] == 'neutral' else 'subjective'
                lexicon[word] = polarity
    except Exception:
        return {}
    return lexicon

MPQA_LEXICON = _load_mpqa_lexicon()

def get_objectivity_from_mpqa(sentence: str) -> float:
    if not MPQA_LEXICON:
        raise RuntimeError("MPQA Lexicon is not available.")
    
    words = simple_tokenizer(sentence)
    if not words:
        return 0.5

    subjective_count = 0
    objective_count = 0
    
    for word in words:
        polarity = MPQA_LEXICON.get(word)
        if polarity == 'subjective':
            subjective_count += 1
        elif polarity == 'objective':
            objective_count += 1
            
    total_found = subjective_count + objective_count
    if total_found == 0:
        return 0.5

    return objective_count / total_found

def calculate_objectivity(
    text: str,
    ensemble_weights: dict = None
) -> float:
    if not text.strip():
        return 0.5

    default_weights = {'swn': 0.20, 'textblob': 0.25, 'vader': 0.25, 'mpqa': 0.30}
    weights = ensemble_weights if ensemble_weights is not None else default_weights

    scores = {}
    
    try:
        tokens = simple_tokenizer(text)
        if tokens:
            swn_scores = [get_objectivity_from_swn(token) for token in tokens]
            scores['swn'] = sum(swn_scores) / len(swn_scores)
    except Exception:
        pass

    try:
        scores['textblob'] = get_objectivity_from_textblob(text)
    except Exception:
        pass

    try:
        scores['vader'] = get_objectivity_from_vader(text)
    except Exception:
        pass
        
    try:
        scores['mpqa'] = get_objectivity_from_mpqa(text)
    except Exception:
        pass

    if not scores:
        return 0.5
    
    # --- BUG FIX IS HERE ---
    # Use .get(key, 0) to safely handle custom weights that don't include all methods.
    weighted_sum = sum(scores[key] * weights.get(key, 0) for key in scores)
    total_weight = sum(weights.get(key, 0) for key in scores)

    if total_weight == 0:
        # Fallback to simple average if no valid weights are provided for the successful methods.
        return sum(scores.values()) / len(scores)

    return weighted_sum / total_weight

def get_objective_synonym(word, pos=None):
    synonyms = get_synonyms(word, pos, include_external=True)
    if not synonyms:
        return {'word': word}
    objectivity_scores = [calculate_objectivity(synonym['word']) for synonym in synonyms]
    if not objectivity_scores:
        return synonyms[0]
    max_score = max(objectivity_scores, default=0)
    if max_score in objectivity_scores:
        return synonyms[objectivity_scores.index(max_score)]
    else:
        return synonyms[0]

def extract_amod(sentence):
    doc = nlp(sentence)
    return [t for t in doc if t.dep_ == "amod"]

def objectify_text(sentence, remove_dependents=False, objectivity_threshold=0.5):
    doc = nlp(sentence)
    skip, replace = set(), {}
    for t in doc:
        score = calculate_objectivity(t.text)
        if t.dep_ == "amod" and score < objectivity_threshold:
            if remove_dependents:
                skip.update(u.i for u in t.subtree)
            else:
                skip.add(t.i)
        elif t.pos_ in {"ADJ", "VERB", "NOUN"} and score < objectivity_threshold:
            rep = get_objective_synonym(t.text, pos=t.pos_)
            if rep:
                replace[t.i] = rep['word']
    for tok in doc:
        if tok.dep_ == "punct" and (tok.i - 1 in skip or tok.i + 1 in skip or tok.head.i in skip):
            skip.add(tok.i)
    out = []
    for t in doc:
        if t.i in skip:
            continue
        out.append((replace.get(t.i, t.text)) + t.whitespace_)
    text = "".join(out)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s([?.!,;:])", r"\1", text)
    return text.strip()

def test_objectify_text(text):
    objectified = objectify_text(text, remove_dependents=True)
    print(f"Original: {text}")
    print(f"Objectivity Score: {calculate_objectivity(text)}")
    print(f"Objectified: {objectified}")
    print(f"Objectivity Score (Objectified): {calculate_objectivity(objectified)}\n")
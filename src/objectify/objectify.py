import logging
import re
import os
import requests
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.corpus import sentiwordnet as swn
from utility import DLog, load_nlp_ecws, cache_resource_decorator, cache_data_decorator, load_inflect

from objectify.synonym import get_synonyms

logger = DLog("OBJECTIFY")

def simple_tokenizer(text: str) -> list[str]:
    logger.info("Simple tokenizer...")
    return re.findall(r'\b\w+\b', text.lower())

@cache_data_decorator
def get_objectivity_from_swn(word: str) -> float:
    logger.info("Getting objectivity from swn...")
    obj_scores = [s.obj_score() for s in swn.senti_synsets(word.lower())]
    return sum(obj_scores) / len(obj_scores) if obj_scores else 0.5

@cache_data_decorator
def get_objectivity_from_textblob(sentence: str) -> float:
    logger.info("Getting textblob objectivity...")
    return 1.0 - TextBlob(sentence).subjectivity

def get_objectivity_from_vader(sentence: str) -> float:
    try:
        analyzer = SentimentIntensityAnalyzer()
        return analyzer.polarity_scores(sentence)['neu']
    except Exception:
        return 0.5

MPQA_LEXICON_PATH = "subjclueslen1-all.tff"
MPQA_LEXICON_URL = "http://mpqa.cs.pitt.edu/lexicons/subj_lexicon/subjclueslen1-all.tff"

def _download_mpqa_lexicon():
    logger.info("_Downloading MPQA lexicon...")
    if not os.path.exists(MPQA_LEXICON_PATH):
        try:
            response = requests.get(MPQA_LEXICON_URL, timeout=5)
            response.raise_for_status()
            with open(MPQA_LEXICON_PATH, 'w') as f:
                f.write(response.text)
        except requests.exceptions.RequestException:
            logger.error("Error fetching MPQA: requests exception...")
            return False
    return True

@cache_resource_decorator
def _load_mpqa_lexicon() -> dict:
    logger.info("_Loading MPQA lexicon...")
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
    except Exception as e:
        logger.error(f"Error in loading error: {str(e)[:50]}")
        return {}
    return lexicon


@cache_resource_decorator
def get_mpqa_lexicon():
    return _load_mpqa_lexicon()

@cache_data_decorator
def get_objectivity_from_mpqa(sentence: str) -> float:
    logger.info("Getting objectivity from MPQA...")
    lexicon = get_mpqa_lexicon()
    if not lexicon:
        return 0.5
    words = simple_tokenizer(sentence)
    if not words:
        return 0.5
    subjective_count = 0
    objective_count = 0
    for word in words:
        polarity = lexicon.get(word)
        if polarity == 'subjective':
            subjective_count += 1
        elif polarity == 'objective':
            objective_count += 1
    total_found = subjective_count + objective_count
    if total_found == 0:
        return 0.5
    return objective_count / total_found

@cache_data_decorator
def calculate_objectivity(
    text: str,
    ensemble_weights: dict = None
) -> float:
    logger.info("Calculating objectivity...")
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
    weighted_sum = sum(scores[key] * weights.get(key, 0) for key in scores)
    total_weight = sum(weights.get(key, 0) for key in scores)
    if total_weight == 0:
        return sum(scores.values()) / len(scores)
    return weighted_sum / total_weight

@cache_data_decorator
def get_objective_synonym(word, pos=None):
    logger.info("Getting most objective synonyms...")
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

nlp = load_nlp_ecws()
inflect_engine = load_inflect()

@cache_data_decorator
def extract_amod(sentence):
    logger.info("Extracting amod...")
    doc = nlp(sentence)
    return [t for t in doc if t.dep_ == "amod"]

def objectify_text(sentence, remove_dependents=False, objectivity_threshold=0.5):
    logger.info("Objectifying text...")
    doc = nlp(sentence)
    p = inflect_engine
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
            rep_word = rep['word'] if rep else t.text
            if t.pos_ == "NOUN":
                if t.tag_ == "NNS":
                    rep_word = p.plural(rep_word)
                elif t.tag_ == "NN": 
                    rep_word = p.singular_noun(rep_word) or rep_word
            elif t.pos_ == "VERB":
                if t.tag_ == "VBD":
                    rep_word = t._.inflect("VBD") if hasattr(t._, "inflect") else rep_word
                elif t.tag_ == "VBG":
                    rep_word = t._.inflect("VBG") if hasattr(t._, "inflect") else rep_word
                elif t.tag_ == "VBZ":
                    rep_word = t._.inflect("VBZ") if hasattr(t._, "inflect") else rep_word
                elif t.tag_ == "VBN":
                    rep_word = t._.inflect("VBN") if hasattr(t._, "inflect") else rep_word
                elif t.tag_ == "VB":
                    rep_word = t._.inflect("VB") if hasattr(t._, "inflect") else rep_word
            elif t.pos_ == "ADJ":
                if hasattr(t._, "inflect"):
                    rep_word = t._.inflect(t.tag_) or rep_word
            replace[t.i] = rep_word
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

def objectify_clusters(clusters):
    for cluster in clusters:
        cluster['summary'] = objectify_text(cluster['summary'])
    return clusters

def test_objectify_text(text):
    logger.info("Testing objectify text...")
    objectified = objectify_text(text, remove_dependents=True)
    print(f"Original: {text}")
    print(f"Objectivity Score: {calculate_objectivity(text)}")
    print(f"Objectified: {objectified}")
    print(f"Objectivity Score (Objectified): {calculate_objectivity(objectified)}\n")
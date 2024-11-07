import spacy
from collections import Counter
from fuzzywuzzy import process, fuzz
import nltk
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import yake
from rake_nltk import Rake

nlp = spacy.load("en_core_web_sm")

def get_keywords(text):
    spacy_kw = [ent.text for ent in nlp(text).ents]

    yake_extractor = yake.KeywordExtractor(lan='en', features=None)
    yake_kw = [kw[0] for kw in yake_extractor.extract_keywords(text)]

    overlapping_keywords = list(set(spacy_kw).intersection(yake_kw))
    overlapping_keywords.remove('Link Copied')

    return overlapping_keywords
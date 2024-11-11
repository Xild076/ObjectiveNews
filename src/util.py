import spacy
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import yake
from rake_nltk import Rake
import pandas as pd

nlp = spacy.load("en_core_web_sm")

def get_keywords(text):
    spacy_kw = [ent.text for ent in nlp(text).ents]

    yake_extractor = yake.KeywordExtractor(lan='en', features=None)
    yake_kw = [kw[0] for kw in yake_extractor.extract_keywords(text)]

    overlapping_keywords = list(set(spacy_kw).intersection(yake_kw))
    overlapping_keywords.remove('Link Copied') if 'Link Copied' in overlapping_keywords else None

    return overlapping_keywords

def find_bias_rating(url):
    df = pd.read_csv('data/bias.csv')
    result = df[df['url'] == url]
    if not result.empty:
        bias_rating = result.iloc[0]['bias_rating']
        accuracy = result.iloc[0]['factual_reporting_rating']
        if accuracy == 'VERY HIGH':
            multiplier = 0.75
        elif accuracy == 'HIGH':
            multiplier = 1.25
        else:
            multiplier = 2
        return abs(bias_rating) * multiplier
    else:
        return 10
import ftfy
import regex as re
import smartypants
from autocorrect import Speller
import html

spell = Speller(lang='en')

def unicode_normalization(text):
    text = ftfy.fix_text(text)
    return text

def whitespace_normalization(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def punctuation_standardization(text):
    text = smartypants.smartypants(text)
    text = html.unescape(text)
    return text

def dash_standardization(text):
    text = re.sub(r'--+', '—', text)
    text = re.sub(r'(\d)\s*-\s*(\d)', r'\1–\2', text)
    text = re.sub(r'\s*—\s*', ' — ', text)
    return text

def spelling_correction(text):
    return spell(text)

def custom_rules(text):
    text = re.sub(r'\bUSA\b', 'United States of America', text)
    return text

def clean_text(text, correct_spelling=False):
    text = unicode_normalization(text)
    text = whitespace_normalization(text)
    text = punctuation_standardization(text)
    text = dash_standardization(text)
    if correct_spelling:
        text = spelling_correction(text)
    text = custom_rules(text)
    return text

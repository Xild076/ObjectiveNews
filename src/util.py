import spacy
import ssl
import yake
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd

ssl._create_default_https_context = ssl._create_unverified_context

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def split_paragraph(text):
    """
    Functions: splits paragraphs into lists

    def split_paragraph(text)
    - text: the full paragraph text

    return re.split(r'\n{2,}', text)
    """
    return re.split(r'\n{2,}', text)

def get_keywords(text):
    """
    Function: retrieves keywords of texts
    
    def get_keywords(text)
    - text: full text

    return overlapping_keywords
    - overlapping_keywords: a list of important keywords
    """
    spacy_kw = [ent.text for ent in nlp(text).ents]
    yake_extractor = yake.KeywordExtractor(lan='en', features=None)
    yake_kw = [kw[0] for kw in yake_extractor.extract_keywords(text)]
    overlapping_keywords = list(set(spacy_kw).intersection(yake_kw))
    if 'Link Copied' in overlapping_keywords:
        overlapping_keywords.remove('Link Copied')
    return overlapping_keywords

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
            multiplier = 0.75
        elif accuracy == 'HIGH':
            multiplier = 1
        else:
            multiplier = 1.25
        return abs(bias_rating) * multiplier
    else:
        return 10

from utility import DLog

logger = DLog("Objectify", "DEBUG", "logs")

logger.info("Importing modules...")
from textblob import TextBlob
import stanza
import re
import nltk
from nltk.corpus import wordnet as wn, sentiwordnet as swn
from nltk.stem import WordNetLemmatizer
import logging
import numpy as np
from synonym import get_contextual_synonyms, get_synonyms
from colorama import Fore
import warnings
from typing import Literal
from utility import get_pos_full_text, normalize_text
import streamlit as st
logger.info("Modules imported...")

stanza_logger = logging.getLogger("stanza")
stanza_logger.setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=FutureWarning)

logger.info("Downloading NLTK...")
nltk.download('wordnet')
nltk.download('sentiwordnet')
nltk.download('omw-1.4')
from nltk.corpus import wordnet as wn, sentiwordnet as swn
from nltk.stem import WordNetLemmatizer
logger.info("NLTK downloaded...")

logger.info("Establishing pipeline...")
lemmatizer = WordNetLemmatizer()

@st.cache_resource
def load_nlp_pipeline():
    nlp = stanza.Pipeline('en')
    return nlp

nlp = load_nlp_pipeline()
logger.info("Pipeline established...")

alphabet = 'abcdefghijklmnopqrstuvwxyz'
vowel = 'aeiou'
consonant = ''.join([char for char in alphabet if char not in vowel])

def _debug_nlp(text):
    doc = nlp(text)
    print(doc)
    return doc

@st.cache_data
def get_word_info(word, context):
    doc = nlp(context)
    for sent in doc.sentences:
        for _word in sent.words:
            if (_word.text == word):
                return {
                    'word': _word.text,
                    'pos': _word.upos.lower()
                }
    return {
        'word': word,
        'pos': None
    }

@st.cache_data
def calc_objectivity_word(word, pos=None):
    end = False
    if pos==-1:
        pos = None
        end = True
    obj_scores = []
    synsets = wn.synsets(word, pos=pos)
    for syn in synsets:
        try:
            swn_syn = swn.senti_synset(syn.name())
            obj_scores.append(swn_syn.obj_score())
        except Exception as e:
            pass
    if obj_scores:
        obj_swn = sum(obj_scores)/len(obj_scores)
    else:
        obj_swn = 0.5
    if obj_swn == 0.0:
        if not end:
            return calc_objectivity_word(word, pos=-1)
        else:
            return 1.0
    return obj_swn

@st.cache_data
def calc_objectivity_sentence(sentence):
    blob = TextBlob(sentence)
    return 1 - blob.subjectivity

@st.cache_data
def get_pos_wn(pos):
    if pos == 'ADJ':
        return wn.ADJ
    elif pos == 'ADV':
        return wn.ADV
    elif pos == 'NOUN':
        return wn.NOUN
    elif pos == 'VERB':
        return wn.VERB
    else:
        return None

@st.cache_data
def split_text_on_quotes(text):
    pattern = r'“(.*?)”|"(.*?)"'
    segments = []
    last_end = 0
    for m in re.finditer(pattern, text):
        start, end = m.span()
        if start > last_end:
            segments.append(('text', text[last_end:start]))
        if m.group(1) is not None:
            segments.append(('quote', m.group()))
        elif m.group(2) is not None:
            segments.append(('quote', m.group()))
        last_end = end
    if last_end < len(text):
        segments.append(('text', text[last_end:]))
    return segments

@st.cache_data
def get_objective_synonym(word, context, synonym_search_methodology:Literal["transformer", "wordnet"]="transformer"):
    pos = get_pos_full_text(get_word_info(word, context)['pos'])
    if synonym_search_methodology == "transformer":
        synonyms = get_contextual_synonyms(word, context)
    else:
        synonyms = [syn['word'] for syn in get_synonyms(word, pos)]
    objectivity = []
    for synonym in synonyms:
        objectivity.append(calc_objectivity_word(lemmatizer.lemmatize(synonym), pos))
    return synonyms[np.argmax(objectivity)]

@st.cache_data
def objectify_text(text: str, objectivity_threshold=0.75, synonym_search_methodology:Literal["transformer", "wordnet"]="transformer"):
    text = text.strip()
    segments = split_text_on_quotes(text)
    processed_segments = []
    for seg_idx, (seg_type, segment) in enumerate(segments):
        if seg_type == 'quote':
            processed_segments.append(segment)
        else:
            doc = nlp(segment)
            processed_sentences = []
            for sentence in doc.sentences:
                word_managed = []
                for word in sentence.words:
                    word_text = word.text
                    word_pos = word.upos
                    word_xpos = word.xpos
                    word_objectivity = calc_objectivity_word(word_text, get_pos_wn(word_pos))
                    word_managed.append({'text': word_text, 'pos': word_pos, 'xpos': word_xpos, 'objectivity': word_objectivity, 'removed': False, 'synonym': False})
                for i, word in enumerate(word_managed):
                    if word['pos'] == "ADJ":
                        if word['objectivity'] < objectivity_threshold:
                            n = i + 1
                            while n < len(word_managed):
                                if word_managed[n]['pos'] in {"NOUN", "PROPN"}:
                                    word_managed[i]['removed'] = True
                                    break
                                if word_managed[n]['pos'] not in {"CCONJ", "CONJ", "PUNCT", "ADJ", "ADV"}:
                                    break
                                n += 1
                            if i >= 1:
                                if word_managed[i-1]['pos'] == "AUX":
                                    word_managed[i]['synonym'] = True
                                    word_managed[i]['removed'] = False
                    elif word['pos'] == "ADV":
                        if word['objectivity'] < objectivity_threshold:
                            word_managed[i]['removed'] = True
                for i, word in enumerate(word_managed):
                    if word['pos'] == "ADV":
                        if i + 1 < len(word_managed) - 1:
                            if word_managed[i+1]['removed'] == True:
                                word_managed[i]['removed'] = True
                    elif word['pos'] == "PUNCT":
                        if 1 <= i <= len(word_managed) - 2:
                            if word['text'] not in {":", ";", "(", ")", "[", "]", "{", "}"}:
                                if not word_managed[i-1]['pos'] in {"AUX", "NOUN", "PROPN"}:
                                    if word_managed[i+1]['removed'] == True or word_managed[i-1]['removed'] == True:
                                        word_managed[i]['removed'] = True
                    elif word['pos'] in {"CCONJ", "CONJ"}:
                        if 2 <= i <= len(word_managed) - 2:
                            if word_managed[i+1]['removed'] == True and (word_managed[i-2]['removed'] == True or word_managed[i-1]['removed'] == True):
                                word_managed[i]['removed'] = True
                    elif word['pos'] in {"NOUN"}:
                        if 2 <= i <= len(word_managed) - 1:
                            if word_managed[i-1]['xpos'] == "HYPH" and word_managed[i-2]['removed'] == True:
                                word_managed[i]['removed'] = True
                for word in word_managed:
                    if not word['removed']:
                        if word['synonym']:
                            synonym = get_objective_synonym(word['text'], sentence.text, synonym_search_methodology=synonym_search_methodology)
                            processed_sentences.append(synonym)
                        else:
                            processed_sentences.append(word['text'])
                post_processed_sentences = []
                for i, word in enumerate(processed_sentences):
                    new_word = word
                    if i < len(processed_sentences) - 1:
                        if new_word.lower() == 'a' and processed_sentences[i+1][0] in vowel:
                            new_word = 'an'
                        elif new_word.lower() == 'an' and processed_sentences[i+1][0] in consonant:
                            new_word = 'a'
                    if 0 < i < len(processed_sentences):
                        if processed_sentences[i-1] == '.':
                            new_word = new_word.capitalize()
                    if i == 0:
                        if seg_idx >= 1 and segments[seg_idx-1][0] == 'quote':
                            prior_quote = segments[seg_idx-1][1].strip()
                            if len(prior_quote) > 1 and prior_quote[-2] in {'.', '!', '?'}:
                                new_word = new_word.capitalize()
                        else:
                            new_word = new_word.capitalize()
                    post_processed_sentences.append(new_word)
            full_sentence = " ".join(post_processed_sentences)
            full_sentence_normalized = normalize_text(full_sentence)
            processed_segments.append(full_sentence_normalized)
    return " ".join(processed_segments).strip()

@st.cache_data
def visualize_objectivity(text, objectivity_threshold=0.75, synonym_search_methodology:Literal["transformer", "wordnet"]="transformer"):
    print(Fore.BLUE + "Text: " + Fore.RESET + text.strip())
    print(Fore.BLUE + "Objectivity: " + Fore.RESET + str(calc_objectivity_sentence(text)))
    objectified_text = objectify_text(text, objectivity_threshold, synonym_search_methodology)
    print(Fore.GREEN + "Objectified Text: " + Fore.RESET + objectified_text.strip())
    print(Fore.GREEN + "Objectivity: " + Fore.RESET + str(calc_objectivity_sentence(objectified_text)))


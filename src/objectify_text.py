from nltk.corpus import wordnet as wn, sentiwordnet as swn
import nltk
import numpy as np
from transformers import pipeline, AutoTokenizer, logging as transformers_logging
from nltk.stem import WordNetLemmatizer
import warnings
import stanza
import logging
import re
from textblob import TextBlob
import language_tool_python
from synonym import get_contextual_synonyms
from text_fixer import clean_text
import ssl

nltk.download('wordnet')
nltk.download('omw-1.4')

ssl._create_default_https_context = ssl._create_unverified_context

transformers_logging.set_verbosity_error()
stanza_logger = logging.getLogger("stanza")
stanza_logger.setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=FutureWarning)
lemmatizer = WordNetLemmatizer()

stanza.download('en')

alphabet = 'abcdefghijklmnopqrstuvwxyz'
vowel = 'aeiou'
consonant = ''.join([char for char in alphabet if char not in vowel])

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

def calc_objectivity_word(word, pos=None):
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
    return obj_swn

def get_objective_synonym(word, context):
    synonyms = get_contextual_synonyms(word, context)
    objectivity = []
    for synonym in synonyms:
        objectivity.append(calc_objectivity_word(synonym))
    return synonyms[np.argmax(objectivity)]

def join_list_to_text(post_processed_sentence):
    sent_processed_text = " ".join(post_processed_sentence)

    sent_processed_text = re.sub(r"(\w)\s+([’'])\s*(\w)", r"\1\2\3", sent_processed_text)
    sent_processed_text = re.sub(r"([(\[{“‘'«])\s+", r'\1', sent_processed_text)
    sent_processed_text = re.sub(r'\s+([)\]}.,!?;:%”’"»])', r'\1', sent_processed_text)
    sent_processed_text = re.sub(r'\s*([-–—])\s*', r' \1 ', sent_processed_text)
    sent_processed_text = re.sub(r',([^ \n])', r', \1', sent_processed_text)
    sent_processed_text = re.sub(r'([.,!?;:])\'(\w)', r'\1 \' \2', sent_processed_text)
    sent_processed_text = re.sub(r'\s+\'([.,!?;:%”’"»])', r"'\1", sent_processed_text)
    sent_processed_text = re.sub(r'\s+\'$', "'", sent_processed_text)
    sent_processed_text = re.sub(r'\s{2,}', ' ', sent_processed_text)

    sent_processed_text = sent_processed_text.strip()

    return sent_processed_text

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

def normalize_contractions(text):
    text = re.sub(r"(\b\w+)\s+('t|'ve|'ll|'re|'m|'d|'s|n't|n’t)", r"\1\2", text)
    return text

def correct_text(text):
    tool = language_tool_python.LanguageTool('en-US')
    matches = tool.check(text)
    corrected_text = language_tool_python.utils.correct(text, matches)
    return corrected_text

def objectify_text(text: str, objectivity_threshold=0.75):
    nlp = stanza.Pipeline('en')
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
                processed_sentence = []
                for i, word in enumerate(sentence.words):
                    add = 0 # add is 0, remove is 1, synonym is 2
                    word_text = word.text
                    word_pos = word.upos
                    word_objectivity = calc_objectivity_word(word_text, get_pos_wn(word_pos))
                    if word_pos == "ADJ":
                        if word_objectivity < objectivity_threshold: # If it is above the threshold, add normally, no need to change
                            if sentence.words[i-1].upos == "AUX": # If preceding is AUX (is, was, etc. -> They were ____), add synonym
                                add = 2
                            else:
                                n = i + 1
                                while n < len(sentence.words):
                                    if sentence.words[n].upos in {"NOUN", "PROPN"}: # If subsequent word is Noun/Proper Noun, remove it
                                        add = 1
                                        break
                                    if sentence.words[n].upos not in {"CCONJ", "CONJ", "PUNCT", "ADJ"}: # If subsequent word is not connecting (conj, punct) or adjective, remove it
                                        break
                                    n += 1
                    elif word_pos == "VERB":
                        if word_objectivity < objectivity_threshold:# If it is above the threshold, add normally, no need to change
                            add = 2
                    elif word_pos == "ADV":
                        if 1 <= i <= len(sentence.words) - 2:
                            prev_token = sentence.words[i-1].upos
                            next_token = sentence.words[i+1].upos
                            if prev_token != "AUX":
                                if next_token == "ADJ":
                                    if calc_objectivity_word(sentence.words[i+1].text, get_pos_wn(next_token)) < objectivity_threshold:
                                        add = 1
                                elif word_objectivity < objectivity_threshold:
                                    add = 1
                    elif word_pos in {"CONJ", "CCONJ"}:
                        if 1 <= i <= len(sentence.words) - 2:
                            prev_token = sentence.words[i-1].upos
                            next_token = sentence.words[i+1].upos
                            if prev_token in {"ADJ", "PUNCT"} and next_token in {"ADJ"}: # If it is surrounded by lists of adjectives
                                if (calc_objectivity_word(sentence.words[i-1].text, get_pos_wn(prev_token)) < objectivity_threshold or 
                                    calc_objectivity_word(sentence.words[i+1].text, get_pos_wn(prev_token)) < objectivity_threshold): # Either or: Conj is less important in grammatical fluency
                                    add = 1
                    elif word_pos in {"PUNCT"}:
                        if 1 <= i <= len(sentence.words) - 2:
                            prev_token = sentence.words[i-1].upos
                            next_token = sentence.words[i+1].upos
                            if prev_token in {"ADJ", "PUNCT"} and next_token in {"ADJ"}: # If it is surrounded by lists of adjectives
                                if (calc_objectivity_word(sentence.words[i-1].text, get_pos_wn(prev_token)) < objectivity_threshold and 
                                    calc_objectivity_word(sentence.words[i+1].text, get_pos_wn(prev_token)) < objectivity_threshold): # And: Conj is less important in grammatical fluency
                                    add = 1
                            if prev_token in {"ADV"}: # If an adverb is before it
                                if calc_objectivity_word(sentence.words[i-1].text, get_pos_wn(prev_token)) < objectivity_threshold:
                                    add = 1
                    
                    if add == 0:
                        processed_sentence.append(word_text)
                    elif add == 2:
                        processed_sentence.append(get_objective_synonym(word_text, sentence.text))
                post_processed_sentence = []
                for i, word in enumerate(processed_sentence):
                    new_word = word
                    if i <= len(processed_sentence) - 2:
                        if new_word.lower() == 'a':
                            if processed_sentence[i+1][0] in vowel:
                                new_word = 'an'
                        if new_word.lower() == 'an':
                            if processed_sentence[i+1][0] in consonant:
                                new_word = 'a'
                    if 1 <= i <= len(processed_sentence) - 1:
                        if processed_sentence[i-1] == '.':
                            new_word = new_word.capitalize()
                    if i == 0:
                        if seg_idx >= 1 and segments[seg_idx-1][0] == 'quote':
                            if segments[seg_idx-1][1].strip()[-2] in {'.', '!', '?'}:
                                new_word.capitalize()
                        else:
                            new_word.capitalize()
                    post_processed_sentence.append(new_word)
                sent_processed_text = join_list_to_text(post_processed_sentence)
                processed_sentences.append(sent_processed_text)
            processed_text = " ".join(processed_sentences).strip()
            processed_text = normalize_contractions(processed_text)
            processed_segments.append(processed_text)
    joined_text = join_list_to_text(processed_segments)
    return joined_text

def visualize_objectivity(text):
    objectified = objectify_text(text)
    textblob_old = TextBlob(text)
    textblob_new = TextBlob(objectified)
    print(objectified)
    print("Unobjectified subjectivity:", textblob_old.subjectivity)
    print("Objectified subjectivity:", textblob_new.subjectivity)

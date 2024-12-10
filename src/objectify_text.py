from nltk.corpus import wordnet as wn, sentiwordnet as swn
import spacy
import numpy as np
from transformers import pipeline, AutoTokenizer, logging as transformers_logging
from nltk import word_tokenize, sent_tokenize
import nltk
import warnings
import stanza
from stanza.models.common.doc import Word
import logging
import re
from textblob import TextBlob
import language_tool_python

transformers_logging.set_verbosity_error()
stanza_logger = logging.getLogger("stanza")
stanza_logger.setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=FutureWarning)

stanza.download('en')
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
synonym_generator = pipeline("fill-mask", model="bert-base-uncased")

alphabet = 'abcdefghijklmnopqrstuvwxyz'
vowel = 'aeiou'
consonant = ''.join([char for char in alphabet if char not in vowel])

def get_contextual_synonyms(word, context):
    masked_sentence = context.replace(word, tokenizer.mask_token, 1)
    results = synonym_generator(masked_sentence)
    synonyms = [res['token_str'].strip() for res in results]
    synonyms.append(word)
    return synonyms

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
    sent_processed_text = re.sub(r'([(\[{“‘"«])\s+', r'\1', sent_processed_text)
    sent_processed_text = re.sub(r'\s+([)\]}.,!?;:%”’"»])', r'\1', sent_processed_text)
    sent_processed_text = re.sub(r'\s*([-–—])\s*', r' \1 ', sent_processed_text)
    sent_processed_text = re.sub(r'\s{2,}', ' ', sent_processed_text)
    sent_processed_text = sent_processed_text.strip()
    return sent_processed_text

def split_text_on_quotes(text):
    pattern = r'(["“”])(.*?)(\1)'
    segments = []
    last_end = 0
    for m in re.finditer(pattern, text):
        start, end = m.span()
        if start > last_end:
            segments.append(('text', text[last_end:start]))
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

def objectify_text(text):
    nlp = stanza.Pipeline('en')
    segments = split_text_on_quotes(text)
    processed_segments = []
    for seg_type, segment in segments:
        if seg_type == 'quote':
            processed_segments.append(segment)
        else:
            doc = nlp(segment)
            processed_sentences = []
            for sentence in doc.sentences:
                processed_sentence = []
                for i, word in enumerate(sentence.words):
                    word_text = word.text
                    pos = word.upos
                    objectivity = calc_objectivity_word(word_text, get_pos_wn(pos))
                    if pos == "ADJ":
                        if i >= 1:
                            if sentence.words[i-1].upos not in {"AUX", "NOUN"}:
                                n = i + 1
                                while n < len(sentence.words):
                                    if sentence.words[i].upos == "NOUN":
                                        processed_sentence.append(word.text)
                                        break
                                    if sentence.words[i].upos not in {"PUNCT", "ADJ", "CCONJ", "CONJ"}:
                                        if objectivity < 0.75:
                                            processed_sentence.append(get_objective_synonym(word_text, sentence.text))
                                        else:
                                            processed_sentence.append(word.text)
                                        break
                                    n += 1
                            else:
                                if objectivity < 0.75:
                                    processed_sentence.append(get_objective_synonym(word_text, sentence.text))
                                else:
                                    processed_sentence.append(word.text)
                    elif pos == "VERB":
                        if objectivity < 0.8:
                            processed_sentence.append(get_objective_synonym(word_text, sentence.text))
                        else:
                            processed_sentence.append(word.text)
                    elif pos == "PUNCT":
                        add = True
                        if 1 <= i <= len(sentence.words) - 2:
                            prev_token = sentence.words[i-1].upos
                            next_token = sentence.words[i+1].upos
                            if (prev_token in {"ADJ", "ADV", "VERB"} and next_token in {"ADJ", "ADV", "CONJ", "CCONJ", "PUNCT"}):
                                add = False
                            if (prev_token in {"ADJ", "ADV"} and next_token in {"VERB", "NOUN", "PNOUN", "PRON"}):
                                add = False
                        if add:
                            processed_sentence.append(word_text)
                    elif pos == "ADP":
                        add = True
                        if 1 <= i <= len(sentence.words) - 2:
                            prev_token = sentence.words[i-1].upos
                            next_token = sentence.words[i+1].upos
                            if (prev_token == "ADV" and next_token == "ADV"):
                                add = False
                        if add:
                            processed_sentence.append(word_text)
                    elif pos in {"CONJ", "CCONJ"}:
                        if 1 <= i <= len(sentence.words) - 2:
                            prev_token = sentence.words[i-1].upos
                            next_token = sentence.words[i+1].upos
                            if not ((prev_token in {"ADJ", "ADV", "PUNCT"}) and (next_token in {"ADJ", "ADV"})):
                                processed_sentence.append(word_text)
                        else:
                            processed_sentence.append(word_text)
                    elif pos == "ADV":
                        if 1 <= i <= len(sentence.words) - 2:
                            prev_token = sentence.words[i-1].upos
                            next_token = sentence.words[i+1].upos
                            if (prev_token == "AUX" and next_token == "PUNCT"):
                                processed_sentence.append(word_text)
                    else:
                        processed_sentence.append(word.text)
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
                        new_word = new_word.capitalize()
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


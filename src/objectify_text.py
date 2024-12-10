import nltk
from nltk.corpus import wordnet as wn, sentiwordnet as swn
import numpy as np
from transformers import pipeline, AutoTokenizer, logging as transformers_logging
import warnings
import stanza
import logging
import re
from textblob import TextBlob
import language_tool_python
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

transformers_logging.set_verbosity_error()
stanza_logger = logging.getLogger("stanza")
stanza_logger.setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=FutureWarning)

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('sentiwordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

stanza.download('en')

nlp = stanza.Pipeline('en')
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
synonym_generator = pipeline("fill-mask", model="distilbert-base-uncased", top_k=5)
tool = language_tool_python.LanguageToolPublicAPI('en-US')

vowels = set('aeiouAEIOU')
consonants = set('bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ')

def get_contextual_synonyms(word, context):
    masked_sentence = context.replace(word, tokenizer.mask_token, 1)
    results = synonym_generator(masked_sentence)
    synonyms = [res['token_str'].strip() for res in results]
    synonyms.append(word)
    return synonyms

def get_pos_wn(pos):
    pos_map = {
        'ADJ': wn.ADJ,
        'ADV': wn.ADV,
        'NOUN': wn.NOUN,
        'VERB': wn.VERB
    }
    return pos_map.get(pos, None)

def calc_objectivity_word(word, pos=None):
    obj_scores = []
    synsets = wn.synsets(word, pos=pos)
    for syn in synsets:
        try:
            swn_syn = swn.senti_synset(syn.name())
            obj_scores.append(swn_syn.obj_score())
        except:
            pass
    if obj_scores:
        obj_swn = sum(obj_scores) / len(obj_scores)
    else:
        obj_swn = 0.5
    return obj_swn

def get_objective_synonym(word, context):
    synonyms = get_contextual_synonyms(word, context)
    objectivity_scores = []
    for synonym in synonyms:
        obj_score = calc_objectivity_word(synonym)
        objectivity_scores.append(obj_score)
    max_index = np.argmax(objectivity_scores)
    return synonyms[max_index]

def join_list_to_text(word_list):
    text = " ".join(word_list)
    text = re.sub(r"\s([.,!?;:)](?:\s|$))", r"\1", text)
    text = re.sub(r"([({])\s", r"\1", text)
    text = re.sub(r"\s([)}])", r"\1", text)
    return text.strip()

def split_text_on_quotes(text):
    pattern = r'(["“”\'‘’])(.*?)(\1)'
    segments = []
    last_end = 0
    for match in re.finditer(pattern, text):
        start, end = match.span()
        if start > last_end:
            segments.append(('text', text[last_end:start]))
        segments.append(('quote', match.group()))
        last_end = end
    if last_end < len(text):
        segments.append(('text', text[last_end:]))
    return segments

def objectify_text(text):
    segments = split_text_on_quotes(text)
    processed_segments = []
    for seg_type, segment in segments:
        if seg_type == 'quote':
            processed_segments.append(segment)
            continue
        doc = nlp(segment)
        processed_sentences = []
        for sentence in doc.sentences:
            words = sentence.words
            processed_sentence = []
            num_words = len(words)
            for i, word in enumerate(words):
                word_text = word.text
                pos = word.upos
                pos_wn = get_pos_wn(pos)
                obj_score = calc_objectivity_word(word_text, pos=pos_wn)
                
                replace_word = False
                skip_word = False
                
                if pos == 'ADJ':
                    modifies_noun = False
                    if i < num_words - 1 and words[i + 1].upos == 'NOUN':
                        modifies_noun = True
                    if i > 0 and words[i - 1].upos == 'NOUN':
                        modifies_noun = True
                    if not modifies_noun and obj_score < 0.75:
                        replace_word = True
                elif pos == 'ADV':
                    if obj_score < 0.75:
                        skip_word = True
                elif pos == 'VERB':
                    if obj_score < 0.8:
                        replace_word = True
                
                if skip_word:
                    continue
                if replace_word:
                    new_word = get_objective_synonym(word_text, sentence.text)
                    processed_sentence.append(new_word)
                else:
                    processed_sentence.append(word_text)
            for i in range(len(processed_sentence) - 1):
                word = processed_sentence[i]
                next_word = processed_sentence[i + 1]
                if word.lower() in ('a', 'an'):
                    if next_word and next_word[0].lower() in vowels:
                        processed_sentence[i] = 'an'
                    else:
                        processed_sentence[i] = 'a'
            if processed_sentence:
                processed_sentence[0] = processed_sentence[0].capitalize()
            processed_text = join_list_to_text(processed_sentence)
            processed_sentences.append(processed_text)
        processed_segment = " ".join(processed_sentences)
        processed_segments.append(processed_segment)
    final_text = " ".join(processed_segments)
    final_text = tool.correct(final_text)
    return final_text.strip()

def visualize_objectivity(text):
    objectified = objectify_text(text)
    textblob_old = TextBlob(text)
    textblob_new = TextBlob(objectified)
    print("Original text:")
    print(text)
    print("\nObjectified text:")
    print(objectified)
    print("\nUnobjectified subjectivity:", round(textblob_old.subjectivity, 2))
    print("Objectified subjectivity:", round(textblob_new.subjectivity, 2))
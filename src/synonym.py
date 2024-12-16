import logging
import warnings
import stanza
from transformers import AutoTokenizer, pipeline
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import spacy
from datamuse import datamuse
from sentence_transformers import SentenceTransformer, util

stanza_logger = logging.getLogger("stanza")
stanza_logger.setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=FutureWarning)

nltk.download('wordnet')
nltk.download('omw-1.4')
lemmatizer = WordNetLemmatizer()

stanza.download('en')
tokenizer = AutoTokenizer.from_pretrained("roberta-large")
mask_synonym_generator = pipeline("fill-mask", model="roberta-large")
text_synonym_generator = pipeline("text-generation", model="gpt2")
nlp = spacy.load('en_core_web_sm')
datamuse_client = datamuse.Datamuse()
model = SentenceTransformer('all-MiniLM-L6-v2')

alphabet = 'abcdefghijklmnopqrstuvwxyz'
vowel = 'aeiou'
consonant = ''.join([char for char in alphabet if char not in vowel])

def get_wordnet_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace('_', ' ').lower()
            if synonym != word.lower():
                synonyms.add(lemmatizer.lemmatize(synonym))
    return synonyms

def get_datamuse_synonyms(word):
    results = datamuse_client.words(rel_syn=word, max=10)
    synonyms = set([res['word'].lower() for res in results])
    return synonyms

def get_gpt_synonyms(word, context, top_n=10):
    prompt = f"Find {top_n} synonyms for the word '{word}' in the context of the sentence: \"{context}\"."
    response = text_synonym_generator(prompt, max_length=100, num_return_sequences=1)
    synonyms = set()
    for res in response:
        text = res['generated_text']
        parts = text.split(':')[-1].split(',')
        synonyms.update([syn.strip().lower() for syn in parts])
    return synonyms

def get_word_tag(word, context):
    doc = nlp(context)
    for token in doc:
        if token.text.lower() == word.lower():
            return token.tag_
    return 'VB'

def get_contextual_synonyms(word, context):
    masked_sentence = context.replace(word, tokenizer.mask_token, 1)
    results = mask_synonym_generator(masked_sentence)
    contextual_synonyms = set(res['token_str'].strip().lower() for res in results)
    contextual_synonyms.add(word.lower())
    wordnet_synonyms = get_wordnet_synonyms(word)
    datamuse_synonyms = get_datamuse_synonyms(word)
    gpt_synonyms = get_gpt_synonyms(word, context)
    all_synonyms = contextual_synonyms.union(wordnet_synonyms, datamuse_synonyms, gpt_synonyms)
    word_tag = get_word_tag(word, context)
    inflected_synonyms = set()
    for syn in all_synonyms:
        doc_syn = nlp(syn)
        if doc_syn:
            syn_token = doc_syn[0]
            inflected = syn_token._.inflect(word_tag)
            if inflected:
                inflected_synonyms.add(inflected.lower())
    valid_synonyms = all_synonyms.union(inflected_synonyms)
    return list(valid_synonyms) if valid_synonyms else [word.lower()]
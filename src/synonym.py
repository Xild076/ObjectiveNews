from utility import DLog, dictionary_pos_to_wordnet
logger = DLog(name="Synonym", level="DEBUG", log_dir="logs")

logger.info("Importing modules...")
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import logging
import requests
import nltk
import streamlit as st
import torch
logger.info("Modules imported...")

logger.info("Downloading NLTK...")
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import wordnet as wn
logger.info("NLTK downloaded...")

logging.getLogger("transformers").setLevel(logging.ERROR)

logger.info("Establishing pipeline...")
@st.cache_resource
def load_model():
    if torch.cuda.is_available():
        return pipeline("fill-mask", model="albert-base-v2", device=0, torch_dtype=torch.bfloat16)
    return pipeline("fill-mask", model="albert-base-v2", device=-1, torch_dtype=torch.bfloat16)
unmasker = load_model()
logger.info("Pipeline established...")

def get_contextual_synonyms(original_word: str, original_sentence: str, top_n: int = 3, top_k: int = 50):
    if original_word.lower() not in original_sentence.lower():
        masked_sentence = original_sentence + " [MASK]"
    else:
        index = original_sentence.lower().index(original_word.lower())
        masked_sentence = ("Find the most objective synonym: " + original_sentence[:index] + "[MASK]" + 
                           original_sentence[index+len(original_word):])
    predictions = unmasker(masked_sentence, top_k=top_k)
    candidate_words = [p["token_str"].strip() for p in predictions if p["token_str"].strip().lower() != original_word.lower()]
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embedding_original = model.encode(original_word, convert_to_tensor=True)
    embeddings_candidates = model.encode(candidate_words, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(embedding_original, embeddings_candidates)[0]
    top_indices = similarities.argsort(descending=True)[:top_n]
    return [candidate_words[i] for i in top_indices]

def get_synonyms(word: str, pos: str = None, deep_search: bool = False):
    synonyms_set = set()
    synonyms_set.add((word, pos))
    try:
        url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}"
        response = requests.get(url)
        data = response.json()
        if isinstance(data, list):
            for entry in data:
                meanings = entry.get('meanings', [])
                for meaning in meanings:
                    part_of_speech = meaning.get('partOfSpeech', 'N/A')
                    if pos and part_of_speech != pos:
                        continue
                    syns = meaning.get('synonyms', [])
                    for syn in syns:
                        if syn.lower() != word.lower():
                            synonyms_set.add((syn, part_of_speech))
        else:
            raise ValueError("DictionaryAPI response not a list.")
    except Exception as e:
        logger.warning(f"DictionaryAPI failed for '{word}' due to exception {e}. Falling back to WordNet.")
        wn_pos = dictionary_pos_to_wordnet(pos) if pos else None
        for synset in wn.synsets(word, pos=wn_pos):
            for lemma in synset.lemmas():
                if lemma.name().lower() != word.lower():
                    synonyms_set.add((lemma.name().replace('_', ' '), synset.pos()))

    if deep_search:
        for s, p in list(synonyms_set):
            deeper = get_synonyms(s, p, False)
            for d in deeper:
                synonyms_set.add((d["word"], p))

    return [{"word": w, "pos": p} for (w, p) in synonyms_set]


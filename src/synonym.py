import os
import torch
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import logging
import requests
import nltk
import streamlit as st
from utility import dictionary_pos_to_wordnet, normalize_text
from nltk.corpus import wordnet as wn
from typing import List, Dict
import numpy as np

os.environ["OMP_NUM_THREADS"] = "1"
torch.set_num_threads(1)
logging.getLogger("transformers").setLevel(logging.ERROR)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

@st.cache_resource
def load_sentence_transformer():
    return SentenceTransformer('all-MiniLM-L6-v2', device='cpu', device_map='auto', load_in_8bit=True)

model = load_sentence_transformer()

@st.cache_resource
def load_synonym_pipeline():
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("fill-mask", model="albert-base-v2", device=device, torch_dtype=torch.float16)

unmasker = load_synonym_pipeline()

def get_contextual_synonyms(original_word: str, original_sentence: str, top_n: int = 3, top_k: int = 50) -> List[str]:
    if original_word.lower() not in original_sentence.lower():
        masked_sentence = original_sentence + " [MASK]"
    else:
        index = original_sentence.lower().index(original_word.lower())
        masked_sentence = f"Find the most objective synonym: {original_sentence[:index]}[MASK]{original_sentence[index+len(original_word):]}"
    predictions = unmasker(masked_sentence, top_k=top_k)
    candidate_words = [p["token_str"].strip() for p in predictions if p["token_str"].strip().lower() != original_word.lower()]
    if not candidate_words:
        return []
    embedding_original = model.encode(original_word, convert_to_tensor=True)
    embeddings_candidates = model.encode(candidate_words, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(embedding_original, embeddings_candidates)[0]
    top_indices = similarities.argsort(descending=True)[:top_n]
    return [candidate_words[i] for i in top_indices]

def get_synonyms(word: str, pos: str = None, deep_search: bool = False) -> List[Dict[str, str]]:
    synonyms_set = {(word, pos)}
    try:
        response = requests.get(f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}", timeout=5)
        data = response.json()
        if isinstance(data, list):
            for entry in data:
                for meaning in entry.get('meanings', []):
                    part_of_speech = meaning.get('partOfSpeech')
                    if pos and part_of_speech != pos:
                        continue
                    for syn in meaning.get('synonyms', []):
                        if syn.lower() != word.lower():
                            synonyms_set.add((syn, part_of_speech))
        else:
            raise ValueError
    except:
        wn_pos = dictionary_pos_to_wordnet(pos) if pos else None
        for synset in wn.synsets(word, pos=wn_pos):
            for lemma in synset.lemmas():
                syn_word = lemma.name().replace('_', ' ')
                if syn_word.lower() != word.lower():
                    synonyms_set.add((syn_word, synset.pos()))
    if deep_search:
        for s, p in list(synonyms_set):
            deeper = get_synonyms(s, p, False)
            synonyms_set.update((d["word"], p) for d in deeper)
    return [{"word": w, "pos": p} for w, p in synonyms_set]
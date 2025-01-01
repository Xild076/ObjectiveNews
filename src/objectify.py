import logging
logging.getLogger("stanza").setLevel(logging.ERROR)
import warnings
import stanza
import nltk
nltk.download("wordnet", quiet=True)
nltk.download("sentiwordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
from nltk.corpus import wordnet as wn, sentiwordnet as swn
from nltk.stem import WordNetLemmatizer
import re
import numpy as np
from textblob import TextBlob
from synonym import get_contextual_synonyms, get_synonyms
from colorama import Fore
from typing import Literal
from utility import get_pos_full_text, normalize_text
import streamlit as st
from functools import lru_cache

warnings.filterwarnings("ignore", category=FutureWarning)
lemmatizer = WordNetLemmatizer()
alphabet = "abcdefghijklmnopqrstuvwxyz"
vowel = "aeiou"
consonant = "".join([c for c in alphabet if c not in vowel])

@st.cache_resource
def load_nlp_pipeline():
    return stanza.Pipeline("en", processors='tokenize,pos', logging_level='ERROR')

nlp = load_nlp_pipeline()

def _debug_nlp(text):
    return nlp(text)

@lru_cache(maxsize=None)
def get_word_info(word, context):
    doc = nlp(context)
    for sent in doc.sentences:
        for w in sent.words:
            if w.text == word:
                return {"word": w.text, "pos": w.upos.lower()}
    return {"word": word, "pos": None}

@lru_cache(maxsize=None)
def calc_objectivity_word(word, pos=None):
    if pos == -1:
        pos = None
    scores = [swn.senti_synset(s.name()).obj_score() for s in wn.synsets(word, pos=pos) if swn.senti_synset(s.name())]
    if not scores and pos != -1:
        return calc_objectivity_word(word, pos=-1)
    return sum(scores)/len(scores) if scores else 0.5 if pos == -1 else 1.0

def calc_objectivity_sentence(sentence):
    return 1 - TextBlob(sentence).subjectivity

def get_pos_wn(pos):
    return {"ADJ": wn.ADJ, "ADV": wn.ADV, "NOUN": wn.NOUN, "VERB": wn.VERB}.get(pos, None)

def split_text_on_quotes(text):
    pattern = r'“(.*?)”|"(.*?)"'
    segs, last = [], 0
    for m in re.finditer(pattern, text):
        s, e = m.span()
        if s > last:
            segs.append(("text", text[last:s]))
        segs.append(("quote", m.group()))
        last = e
    if last < len(text):
        segs.append(("text", text[last:]))
    return segs

def get_objective_synonym(word, context, method="transformer"):
    pos = get_pos_full_text(get_word_info(word, context)["pos"])
    syns = get_contextual_synonyms(word, context) if method == "transformer" else [s["word"] for s in get_synonyms(word, pos)]
    syns = syns or [word]
    scores = [calc_objectivity_word(lemmatizer.lemmatize(s), get_pos_wn(get_word_info(s, context)["pos"])) for s in syns]
    return syns[np.argmax(scores)] if syns else word

def objectify_text(text, threshold=0.75, method="transformer"):
    text = text.strip()
    segs = split_text_on_quotes(text)
    out = []
    for tp, seg in segs:
        if tp == "quote":
            out.append(seg)
        else:
            doc = nlp(seg)
            final = []
            for sent in doc.sentences:
                wm = [{"text": w.text, "pos": w.upos, "xpos": w.xpos, "obj": calc_objectivity_word(w.text, get_pos_wn(w.upos)), "r": False, "syn": False} for w in sent.words]
                for idx, w in enumerate(wm):
                    if w["pos"] == "ADJ" and w["obj"] < threshold:
                        n = idx + 1
                        while n < len(wm) and wm[n]["pos"] in {"CCONJ", "CONJ", "PUNCT", "ADJ", "ADV"}:
                            if wm[n]["pos"] in {"NOUN", "PROPN"}:
                                wm[idx]["r"] = True
                                break
                            n += 1
                        if idx > 0 and wm[idx-1]["pos"] == "AUX":
                            wm[idx]["syn"] = True
                            wm[idx]["r"] = False
                    elif w["pos"] == "ADV" and w["obj"] < threshold:
                        wm[idx]["r"] = True
                for idx, w in enumerate(wm):
                    if w["pos"] == "ADV" and idx + 1 < len(wm) and wm[idx+1]["r"]:
                        wm[idx]["r"] = True
                    elif w["pos"] == "PUNCT" and 0 < idx < len(wm)-1 and w["text"] not in {":", ";", "(", ")", "[", "]", "{", "}"}:
                        if wm[idx-1]["pos"] not in {"AUX", "NOUN", "PROPN"} and wm[idx+1]["r"] or wm[idx-1]["r"]:
                            wm[idx]["r"] = True
                    elif w["pos"] in {"CCONJ", "CONJ"} and 1 < idx < len(wm)-1:
                        if wm[idx+1]["r"] and (wm[idx-2]["r"] if idx >= 2 else False or wm[idx-1]["r"]):
                            wm[idx]["r"] = True
                    elif w["pos"] == "NOUN" and 1 < idx < len(wm) and wm[idx-1]["xpos"] == "HYPH" and wm[idx-2]["r"]:
                        wm[idx]["r"] = True
                for w in wm:
                    if not w["r"]:
                        final.append(get_objective_synonym(w["text"], seg, method) if w["syn"] else w["text"])
                pps = []
                for idx, w in enumerate(final):
                    nw = w
                    if idx < len(final)-1:
                        if nw.lower() == "a" and final[idx+1][0] in vowel:
                            nw = "an"
                        elif nw.lower() == "an" and final[idx+1][0] in consonant:
                            nw = "a"
                    if idx > 0 and final[idx-1] == ".":
                        nw = nw.capitalize()
                    if idx == 0:
                        if out and segs[out.index(("quote", segs[out.index(("quote", segs[idx-1][0]))][1]))][0] == "quote":
                            pq = segs[idx-1][1].strip()
                            if len(pq) > 1 and pq[-2] in {".", "!", "?"}:
                                nw = nw.capitalize()
                        else:
                            nw = nw.capitalize()
                    pps.append(nw)
                fs = " ".join(pps)
                out.append(normalize_text(fs))
    return " ".join(out).strip()

def visualize_objectivity(text, threshold=0.75, method="transformer"):
    print(Fore.BLUE + "Text: " + Fore.RESET + text.strip())
    print(Fore.BLUE + "Objectivity: " + Fore.RESET + str(calc_objectivity_sentence(text)))
    o = objectify_text(text, threshold, method)
    print(Fore.GREEN + "Objectified Text: " + Fore.RESET + o.strip())
    print(Fore.GREEN + "Objectivity: " + Fore.RESET + str(calc_objectivity_sentence(o)))
import logging
logging.getLogger("stanza").setLevel(logging.ERROR)
import warnings
import stanza
import nltk
nltk.download("wordnet")
nltk.download("sentiwordnet")
nltk.download("omw-1.4")
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

warnings.filterwarnings("ignore", category=FutureWarning)
lemmatizer = WordNetLemmatizer()

@st.cache_resource
def load_nlp_pipeline():
    return stanza.Pipeline("en")

nlp = load_nlp_pipeline()
alphabet = "abcdefghijklmnopqrstuvwxyz"
vowel = "aeiou"
consonant = "".join([c for c in alphabet if c not in vowel])

def _debug_nlp(text):
    return nlp(text)

def get_word_info(word, context):
    doc = nlp(context)
    for sent in doc.sentences:
        for w in sent.words:
            if w.text == word:
                return {"word": w.text, "pos": w.upos.lower()}
    return {"word": word, "pos": None}

def calc_objectivity_word(word, pos=None):
    end = False
    if pos == -1:
        pos = None
        end = True
    scores = []
    syns = wn.synsets(word, pos=pos)
    for s in syns:
        try:
            swn_syn = swn.senti_synset(s.name())
            scores.append(swn_syn.obj_score())
        except:
            pass
    if scores:
        o = sum(scores) / len(scores)
    else:
        o = 0.5
    if o == 0.0:
        if not end:
            return calc_objectivity_word(word, pos=-1)
        else:
            return 1.0
    return o

def calc_objectivity_sentence(sentence):
    b = TextBlob(sentence)
    return 1 - b.subjectivity

def get_pos_wn(pos):
    if pos == "ADJ": return wn.ADJ
    elif pos == "ADV": return wn.ADV
    elif pos == "NOUN": return wn.NOUN
    elif pos == "VERB": return wn.VERB

def split_text_on_quotes(text):
    p = r'“(.*?)”|"(.*?)"'
    segs = []
    last_end = 0
    for m in re.finditer(p, text):
        s, e = m.span()
        if s > last_end:
            segs.append(("text", text[last_end:s]))
        if m.group(1) is not None:
            segs.append(("quote", m.group()))
        elif m.group(2) is not None:
            segs.append(("quote", m.group()))
        last_end = e
    if last_end < len(text):
        segs.append(("text", text[last_end:]))
    return segs

def get_objective_synonym(word, context, synonym_search_methodology="transformer"):
    pos = get_pos_full_text(get_word_info(word, context)["pos"])
    if synonym_search_methodology == "transformer":
        syns = get_contextual_synonyms(word, context)
    else:
        syns = [s["word"] for s in get_synonyms(word, pos)]
    scores = []
    for s in syns:
        scores.append(calc_objectivity_word(lemmatizer.lemmatize(s), pos))
    return syns[np.argmax(scores)]

def objectify_text(text, objectivity_threshold=0.75, synonym_search_methodology="transformer"):
    text = text.strip()
    segs = split_text_on_quotes(text)
    out = []
    for i, (tp, seg) in enumerate(segs):
        if tp == "quote":
            out.append(seg)
        else:
            d = nlp(seg)
            ps = []
            for sentence in d.sentences:
                wm = []
                for w in sentence.words:
                    wt = w.text
                    wp = w.upos
                    wx = w.xpos
                    wo = calc_objectivity_word(wt, get_pos_wn(wp))
                    wm.append({"text": wt, "pos": wp, "xpos": wx, "obj": wo, "r": False, "syn": False})
                for idx, w in enumerate(wm):
                    if w["pos"] == "ADJ":
                        if w["obj"] < objectivity_threshold:
                            n = idx + 1
                            while n < len(wm):
                                if wm[n]["pos"] in {"NOUN", "PROPN"}:
                                    wm[idx]["r"] = True
                                    break
                                if wm[n]["pos"] not in {"CCONJ", "CONJ", "PUNCT", "ADJ", "ADV"}:
                                    break
                                n += 1
                            if idx >= 1 and wm[idx-1]["pos"] == "AUX":
                                wm[idx]["syn"] = True
                                wm[idx]["r"] = False
                    elif w["pos"] == "ADV":
                        if w["obj"] < objectivity_threshold:
                            wm[idx]["r"] = True
                for idx, w in enumerate(wm):
                    if w["pos"] == "ADV":
                        if idx + 1 < len(wm) - 1 and wm[idx+1]["r"] == True:
                            wm[idx]["r"] = True
                    elif w["pos"] == "PUNCT":
                        if 1 <= idx <= len(wm) - 2:
                            if w["text"] not in {":", ";", "(", ")", "[", "]", "{", "}"}:
                                if wm[idx-1]["pos"] not in {"AUX", "NOUN", "PROPN"}:
                                    if wm[idx+1]["r"] == True or wm[idx-1]["r"] == True:
                                        wm[idx]["r"] = True
                    elif w["pos"] in {"CCONJ", "CONJ"}:
                        if 2 <= idx <= len(wm) - 2:
                            if wm[idx+1]["r"] == True and (wm[idx-2]["r"] == True or wm[idx-1]["r"] == True):
                                wm[idx]["r"] = True
                    elif w["pos"] == "NOUN":
                        if 2 <= idx <= len(wm) - 1:
                            if wm[idx-1]["xpos"] == "HYPH" and wm[idx-2]["r"] == True:
                                wm[idx]["r"] = True
                final = []
                for w in wm:
                    if not w["r"]:
                        if w["syn"]:
                            syn = get_objective_synonym(w["text"], sentence.text, synonym_search_methodology)
                            final.append(syn)
                        else:
                            final.append(w["text"])
                pps = []
                for idx, w in enumerate(final):
                    nw = w
                    if idx < len(final) - 1:
                        if nw.lower() == "a" and final[idx+1][0] in vowel:
                            nw = "an"
                        elif nw.lower() == "an" and final[idx+1][0] in consonant:
                            nw = "a"
                    if idx > 0:
                        if final[idx-1] == ".":
                            nw = nw.capitalize()
                    if idx == 0:
                        if i >= 1 and segs[i-1][0] == "quote":
                            pq = segs[i-1][1].strip()
                            if len(pq) > 1 and pq[-2] in {".", "!", "?"}:
                                nw = nw.capitalize()
                        else:
                            nw = nw.capitalize()
                    pps.append(nw)
            fs = " ".join(pps)
            fsn = normalize_text(fs)
            out.append(fsn)
    return " ".join(out).strip()

def visualize_objectivity(text, objectivity_threshold=0.75, synonym_search_methodology="transformer"):
    print(Fore.BLUE + "Text: " + Fore.RESET + text.strip())
    print(Fore.BLUE + "Objectivity: " + Fore.RESET + str(calc_objectivity_sentence(text)))
    o = objectify_text(text, objectivity_threshold, synonym_search_methodology)
    print(Fore.GREEN + "Objectified Text: " + Fore.RESET + o.strip())
    print(Fore.GREEN + "Objectivity: " + Fore.RESET + str(calc_objectivity_sentence(o)))
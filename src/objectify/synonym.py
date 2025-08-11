import nltk
import requests
from nltk.corpus import wordnet as wn, sentiwordnet as swn
from utility import DLog, cache_data_decorator, load_nlp_ecws, load_inflect

nlp = load_nlp_ecws()
inflect_engine = load_inflect()

logger = DLog("SYNONYM")

_COARSE_POS = {"n", "v", "a", "r"}

# @cache_data_decorator
def _normalize_pos(pos):
    logger.info("_Normalizing part-of-speech...")
    if not pos:
        return None
    p = pos.lower()
    if p in ("noun", "n"):
        return "n"
    if p in ("verb", "v"):
        return "v"
    if p in ("adverb", "adv", "r"):
        return "r"
    if p in ("adjective", "adj", "a"):
        return "a"
    if "satellite" in p or p == "s":
        return "a"
    return None

# @cache_data_decorator
def _synset_coarse_pos(_synset):
    logger.info("_Synset coarse part-of-speech...")
    p = _synset.pos()
    return "a" if p == "s" else p

# @cache_data_decorator
def _collect_wordnet_synonyms(word, pos=None, deep=False, sentiment_filter=False):
    logger.info("_Collecting wordnet synonyms...")
    cpos = _normalize_pos(pos)
    synsets = wn.synsets(word, pos=cpos) if cpos else wn.synsets(word)
    if cpos:
        synsets = [s for s in synsets if _synset_coarse_pos(s) == cpos]

    if not synsets:
        return {}

    target_sent = None
    if sentiment_filter:
        vals = []
        for s in synsets:
            try:
                sw = swn.senti_synset(s.name())
                vals.append(sw.pos_score() - sw.neg_score())
            except:
                pass
        if vals:
            target_sent = sum(vals) / len(vals)

    candidates = {}
    for s in synsets:
        c = _synset_coarse_pos(s)
        if cpos and c != cpos:
            continue
        for lemma in s.lemmas():
            name = lemma.name().replace("_", " ")
            if name.lower() == word.lower():
                continue
            score = lemma.count()
            keep = True
            if sentiment_filter and target_sent is not None:
                try:
                    sw = swn.senti_synset(s.name())
                    sent = sw.pos_score() - sw.neg_score()
                    if target_sent == 0:
                        keep = abs(sent) < 0.25
                    else:
                        keep = (sent == 0) or (sent * target_sent > 0) or (abs(sent - target_sent) < 0.25)
                except:
                    pass
            if keep:
                prev = candidates.get(name)
                if not prev or score > prev["score"]:
                    candidates[name] = {
                        "word": name,
                        "pos": c,
                        "score": prev["score"] + score if prev else score,
                        "definition": s.definition(),
                        "sense": s.name(),
                    }
                else:
                    candidates[name]["score"] += score

        if deep:
            related = []
            if c == "a":
                related.extend(s.similar_tos())
            related.extend(s.also_sees())
            for lemma in s.lemmas():
                related.extend(lemma.pertainyms())
            for rs in related:
                rc = _synset_coarse_pos(rs)
                if cpos and rc != cpos:
                    continue
                for lemma in rs.lemmas():
                    name = lemma.name().replace("_", " ")
                    if name.lower() == word.lower():
                        continue
                    score = max(1, lemma.count() // 2)
                    keep = True
                    if sentiment_filter and target_sent is not None:
                        try:
                            sw = swn.senti_synset(rs.name())
                            sent = sw.pos_score() - sw.neg_score()
                            if target_sent == 0:
                                keep = abs(sent) < 0.25
                            else:
                                keep = (sent == 0) or (sent * target_sent > 0) or (abs(sent - target_sent) < 0.25)
                        except:
                            pass
                    if keep:
                        prev = candidates.get(name)
                        if not prev or score > prev["score"]:
                            candidates[name] = {
                                "word": name,
                                "pos": rc,
                                "score": prev["score"] + score if prev else score,
                                "definition": rs.definition(),
                                "sense": rs.name(),
                            }
                        else:
                            candidates[name]["score"] += score

    return candidates

# @cache_data_decorator
def _collect_external_synonyms(word, pos=None):
    logger.info("_Collect external_synonyms...")
    cpos = _normalize_pos(pos)
    out = {}
    try:
        resp = requests.get(f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}", timeout=5)
        data = resp.json()
        if isinstance(data, list):
            for entry in data:
                for meaning in entry.get("meanings", []):
                    mpos = _normalize_pos(meaning.get("partOfSpeech"))
                    if cpos and mpos and mpos != cpos:
                        continue
                    if cpos and not mpos:
                        continue
                    for syn in meaning.get("synonyms", []):
                        name = syn.strip()
                        if name.lower() == word.lower():
                            continue
                        out[name] = {
                            "word": name,
                            "pos": mpos if mpos else cpos if cpos else None,
                            "score": 1,
                            "definition": None,
                            "sense": None,
                        }
    except Exception as e:
        logger.error(f"Error collecting external synonyms: {str(e)[:50]}")
    return out

# @cache_data_decorator
def get_synonyms(word, pos=None, deep=False, sentiment_filter=False, include_external=False, topn=None, simple=False):
    logger.info("Getting synonyms...")
    wn_cands = _collect_wordnet_synonyms(word, pos=pos, deep=deep, sentiment_filter=sentiment_filter)
    if include_external:
        ext_cands = _collect_external_synonyms(word, pos=pos)
        for k, v in ext_cands.items():
            if k in wn_cands:
                wn_cands[k]["score"] += v["score"]
            else:
                wn_cands[k] = v
    cands = list(wn_cands.values())
    cands.sort(key=lambda d: (-d["score"], d["word"]))
    if topn is not None:
        cands = cands[:topn]
    if simple:
        return [{"word": d["word"], "pos": d["pos"]} for d in cands]
    return cands


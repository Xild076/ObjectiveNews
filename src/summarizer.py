import nltk
nltk.download('punkt')
from utility import DLog, clean_text, split_sentences, load_sent_transformer, cache_resource_decorator
from transformers import pipeline
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Literal, Dict, Any
import re

device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger = DLog(name="SUMMARIZER", level="DEBUG")

@cache_resource_decorator
def load_summarizer():
    summarizer = pipeline(
        "summarization",
        model="facebook/bart-large-cnn",
        device=0 if torch.cuda.is_available() else -1,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    )
    return summarizer

def grammar_correct(text: str) -> str:
    logger.info("Correcting grammar...")
    text = re.sub(r'\s+', ' ', text.strip())
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    if text and text[-1] not in '.!?':
        text += '.'
    return text

def extractive_summarize(texts: List[str], top_k: int = 3) -> str:
    logger.info("Performing extractive summarization...")
    if not texts:
        return ""
    embed_model = load_sent_transformer()
    embs = embed_model.encode(texts, show_progress_bar=False)
    centroid = embs.mean(axis=0, keepdims=True)
    sims = cosine_similarity(centroid, embs)[0]
    idxs = sims.argsort()[::-1][:min(top_k, len(texts))]
    return " ".join(texts[i] for i in idxs)

def chunked_summarize(text: str, max_length: int = 200, min_length: int = 100, num_beams: int = 4) -> str:
    logger.info("Performing chunked summarization...")
    summarizer = load_summarizer()
    sents = split_sentences(text)
    chunks = (" ".join(sents[i:i+20]) for i in range(0, len(sents), 20))
    combined = " ".join(
        summarizer("summarize: " + chunk, max_length=max_length, min_length=min_length, num_beams=num_beams)[0]["summary_text"]
        for chunk in chunks
    )
    final = summarizer("summarize: " + combined, max_length=max_length, min_length=min_length, num_beams=num_beams)[0]["summary_text"]
    return final

def summarize(texts: List[str], level: Literal["fast", "medium", "slow"] = "fast") -> str:
    logger.info(f"Summarizing with level: {level}")
    if level == "fast":
        summary = texts[0] if texts else ""
    elif level == "medium":
        summary = extractive_summarize(texts, top_k=3)
    else:
        summary = chunked_summarize(" ".join(texts))
    try:
        summary = grammar_correct(summary)
    except Exception as e:
        logger.error(f"Grammar correction failed: {str(e)[:50]}")
    return clean_text(summary)

def summarize_clusters(clusters: List[Dict[str, Any]], level: Literal["fast", "medium", "slow"] = "fast"):
    logger.info(f"Summarizing clusters with level: {level}")
    for c in clusters:
        texts = (sent.text for sent in c["sentences"] if sent.text.strip())
        source = list(texts) or [c["representative"].text]
        c["summary"] = summarize(source, level)
    return clusters

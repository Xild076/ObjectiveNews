import re
import torch
import numpy as np
from typing import List, Literal, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import google.generativeai as genai
from utility import DLog, clean_text, split_sentences, load_sent_transformer, cache_resource_decorator, IS_STREAMLIT, encode_sentences_cached

device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger = DLog(name="SUMMARIZER")

_sent_model = None
def _get_sent_model():
    global _sent_model
    if _sent_model is None:
        _sent_model = load_sent_transformer()
    return _sent_model

 

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
    embs = encode_sentences_cached(texts)
    centroid = embs.mean(axis=0, keepdims=True)
    sims = cosine_similarity(centroid, embs)[0]
    idxs = sims.argsort()[::-1][:min(top_k, len(texts))]
    return " ".join(texts[i] for i in idxs)

def ai_summary(text: str, max_length: int = 200, min_length: int = 100) -> str:
    logger.info("Performing AI summarization...")
    model = genai.GenerativeModel("gemma-3-27b-it")
    api_key = None
    if IS_STREAMLIT:
        try:
            import streamlit as st
            api_key = st.secrets["gemma_api_key"]
        except Exception:
            pass
    if not api_key:
        try:
            with open("secrets/gemma.txt", "r") as f:
                api_key = f.read().strip()
        except Exception:
            api_key = None
    if not api_key:
        logger.warning("Gemma API key not configured; falling back to extractive summarization.")
        return extractive_summarize(split_sentences(text), top_k=3)
    genai.configure(api_key=api_key)
    generation_config = genai.types.GenerationConfig(
            temperature=0.1,
            max_output_tokens=5000,
            top_p=0.9,
            top_k=40
    )
    try:
        response = model.generate_content(
            f"""You are an unbaised and objective summarizer. Summarize the following text in a concise and objective manner, focusing on the key points and avoiding any subjective opinions or interpretations. The summary should be between {min_length} and {max_length} characters long. Respond with only the summary text, without any additional commentary or explanations.
            Text: {text}""",
            generation_config=generation_config
        )
        return response.text.strip()
    except:
        logger.error("AI summarization failed. Falling back to extractive summarization.")
        return extractive_summarize(split_sentences(text), top_k=3)

def summarize(texts: List[str], level: Literal["fast", "medium", "slow"] = "fast") -> str:
    logger.info(f"Summarizing with level: {level}")
    if level == "fast":
        summary = texts[0] if texts else ""
    elif level == "medium":
        summary = extractive_summarize(texts, top_k=3)
    else:
        summary = ai_summary(texts[0], max_length=1000, min_length=200) if texts else ""
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


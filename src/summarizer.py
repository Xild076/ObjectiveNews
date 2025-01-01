from utility import DLog

logger = DLog(name="Summarizer", level="DEBUG", log_dir="logs")

logger.info("Importing modules...")
import logging
logging.getLogger('transformers').setLevel(logging.ERROR)
from transformers import pipeline
from utility import normalize_text
import streamlit as st
import torch
logger.info("Modules imported...")

logger.info("Establishing pipeline...")
torch.set_num_threads(1)
# @st.cache_resource
def load_model():
    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        "summarization",
        model="google/flan-t5-small",
        device=device,
        torch_dtype=torch.bfloat16
    )
summarizer = load_model()
logger.info("Pipeline established...")

def summarize_text(text, max_length=200, min_length=100, num_beams=4):
    output = summarizer("summarize: " + text, max_length=max_length, min_length=min_length, num_beams=num_beams)
    text = output[0]['summary_text']
    text = text.replace("summarize:", "").strip()
    text = normalize_text(text)
    return text


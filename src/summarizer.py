from utility import DLog

logger = DLog(name="Summarizer", level="DEBUG", log_dir="logs")

logger.info("Importing modules...")
import logging
logging.getLogger('transformers').setLevel(logging.ERROR)
from transformers import pipeline
from utility import normalize_text
logger.info("Modules imported...")

logger.info("Establishing pipeline...")
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=-1)
logger.info("Pipeline established...")

def summarize_text(text, max_length=200, min_length=100, num_beams=4):
    output = summarizer("summarize: " + text, max_length=max_length, min_length=min_length, num_beams=num_beams)
    text = output[0]['summary_text']
    text = text.replace("summarize:", "").strip()
    text = normalize_text(text)
    return text


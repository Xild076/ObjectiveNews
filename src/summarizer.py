import nltk

from utility import clean_text, split_sentences
nltk.download('punkt')
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
import torch
from transformers import pipeline
from utility import clean_text

device = 0 if torch.cuda.is_available() else -1
summarizer = pipeline(
    "summarization",
    model="google/flan-t5-small",
    device=device,
    torch_dtype=torch.bfloat16
)

def preprocess(text,min_len=8):
    return [s for s in split_sentences(text) if len(s.split())>=min_len]

def pagerank(M,eps=1e-6,d=0.85):
    import warnings
    n=M.shape[0]
    S=M.sum(axis=1)
    P=np.divide(M,S[:,None],where=S[:,None]!=0)
    v=np.ones(n)/n
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        while True:
            v_new=(1-d)/n + d*P.T.dot(v)
            if np.isnan(v_new).any() or np.isnan(v).any():
                break
            if np.linalg.norm(v_new-v)<eps:
                return v_new
            v=v_new
    return v

def lexrank(text,n=2,threshold=0.1):
    sents=preprocess(text)
    v=TfidfVectorizer(stop_words='english')
    M=v.fit_transform(sents).toarray()
    norms=np.linalg.norm(M,axis=1)
    sim=np.matmul(M,M.T)/norms[:,None]/norms[None,:]
    sim[sim<threshold]=0
    np.fill_diagonal(sim,0)
    scores=pagerank(sim)
    idx=np.argsort(scores)[::-1][:n]
    return [sents[i] for i in idx]

def summarize_text(text, max_length=200, min_length=100, num_beams=4):
    output = summarizer("summarize: " + text, max_length=max_length, min_length=min_length, num_beams=num_beams)
    text = output[0]['summary_text']
    text = text.replace("summarize:", "").strip()
    text = clean_text(text)
    return text

import numpy as np
import torch
import torch.nn as nn
from math import floor, log
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, cosine_similarity
from sentence_transformers import SentenceTransformer
import nltk
nltk.download("punkt", quiet=True)
from nltk import sent_tokenize
from typing import List, Dict, Any, Union, Type
from colorama import Fore
import streamlit as st
from utility import preprocess_text, normalize_values_minmax, SentenceHolder, fix_space_newline
import random

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dim_feedforward=2048, dropout=0.1, max_len=5000):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        pos = torch.zeros(max_len, embed_dim)
        pos_indices = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-log(10000.0) / embed_dim))
        pos[:, 0::2] = torch.sin(pos_indices * div_term)
        pos[:, 1::2] = torch.cos(pos_indices * div_term)
        self.register_buffer('positional_encoding', pos.unsqueeze(0))

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src = src + self.positional_encoding[:, :src.size(1)]
        src2, _ = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        src = self.norm1(src + self.dropout1(src2))
        src2 = self.linear2(self.dropout(torch.relu(self.linear1(src))))
        src = self.norm2(src + self.dropout2(src2))
        return src

@st.cache_resource
def load_model():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    model.eval()
    return model

sentence_embed_model = load_model()
attention_model = SelfAttention(embed_dim=384, num_heads=4).eval()

def encode_text(
    sentences: List[str],
    weights: Dict[str, float] = {"single": 0.7, "context": 0.3},
    context=False,
    context_len=1,
    preprocess=True,
    attention=True
):
    if preprocess:
        sentences = [preprocess_text(s) for s in sentences]
    with torch.no_grad():
        e = sentence_embed_model.encode(sentences, convert_to_numpy=True)
        e = torch.tensor(e, dtype=torch.float16)
        if attention:
            e = e.unsqueeze(1)
            att_out = attention_model(e)
            e = att_out.squeeze(1).half().numpy()
        else:
            e = e.half().numpy()
    f = []
    X = e
    for i, emb in enumerate(X):
        add = emb
        if context:
            start = max(0, i - context_len)
            end = min(len(X), i + context_len + 1)
            add = X[start:end].mean(axis=0)
        final = emb * weights["single"] + add * weights["context"]
        f.append(final)
    return np.array(f, dtype=np.float16)

def find_representative_sentence(X, labels, cluster_label):
    idxs = np.where(labels == cluster_label)[0]
    c_points = X[idxs]
    if len(c_points) == 0:
        raise ValueError(f"No points for cluster {cluster_label}.")
    centroid = c_points.mean(axis=0).reshape(1, -1)
    sims = cosine_similarity(c_points, centroid).flatten()
    return idxs[np.argmax(sims)]

def get_sentence_with_context(texts, idx, context_len):
    s = max(0, idx - context_len)
    e = min(len(texts), idx + context_len + 1)
    return " ".join(texts[s:e])

def observe_best_cluster(
    sentences_holder: List[SentenceHolder],
    max_clusters=10,
    weights={"single": 0.7, "context": 0.3},
    context=False,
    context_len=1,
    preprocess=True,
    attention=True,
    clustering_method: Union[Type[KMeans]] = KMeans,
    score_weights={"sil": 0.4, "db": 0.2, "ch": 0.4}
):
    sents = [s.text for s in sentences_holder]
    X = encode_text(sents, weights, context, context_len, preprocess, attention)
    max_clusters = min(max_clusters, len(X))
    best_score, best = float("-inf"), None
    for i in range(2, max_clusters):
        c = clustering_method(n_clusters=i)
        lbls = c.fit_predict(X)
        sil = silhouette_score(X, lbls)
        db = davies_bouldin_score(X, lbls)
        ch = calinski_harabasz_score(X, lbls)
        score = sil * score_weights["sil"] + (1 - db) * score_weights["db"] + ch * score_weights["ch"]
        if score > best_score:
            best_score, best = score, {"n_clusters": i, "sil": sil, "db": db, "ch": ch, "labels": lbls}
    lbls = best["labels"]
    clusters = []
    for cl in sorted(set(lbls)):
        c_sents = [sentences_holder[i] for i, lb in enumerate(lbls) if lb == cl]
        rep_idx = find_representative_sentence(X, lbls, cl)
        rep_s = sentences_holder[rep_idx]
        rep_context = SentenceHolder(
            get_sentence_with_context(sents, rep_idx, context_len),
            source=rep_s.source,
            author=rep_s.author,
            date=rep_s.date
        ) if context else rep_s
        clusters.append({
            "cluster_id": int(cl),
            "sentences": [c_sents[i] for i in np.argsort(-cosine_similarity([X[find_representative_sentence(X, lbls, cl)]], X[lbls == cl])[0])],
            "representative": rep_s,
            "representative_with_context": rep_context
        })
    return {
        "clusters": clusters,
        "metrics": {
            "silhouette": float(best["sil"]),
            "davies_bouldin": float(best["db"]),
            "calinski_harabasz": float(best["ch"]),
            "score": float(best_score)
        }
    }

def visualize_grouping(text):
    s = sent_tokenize(text)
    s_h = [SentenceHolder(t) for t in s]
    c = observe_best_cluster(
        s_h,
        max_clusters=8,
        context=True,
        context_len=1,
        weights={"single":0.7,"context":0.3},
        preprocess=True,
        attention=True,
        clustering_method=KMeans,
        score_weights={"sil":0.4,"db":0.2,"ch":0.4}
    )["clusters"]
    for cluster in c:
        print(Fore.BLUE + f"Cluster {cluster['cluster_id']}:" + Fore.RESET)
        print(Fore.LIGHTGREEN_EX + "Representative: " + Fore.RESET + fix_space_newline(cluster["representative"].text))
        print(Fore.LIGHTCYAN_EX + "Representative with context: " + Fore.RESET + fix_space_newline(cluster["representative_with_context"].text))
        print(Fore.LIGHTYELLOW_EX + "Sentences:" + Fore.RESET)
        for sent in cluster["sentences"]:
            print(fix_space_newline(sent.text))
        print()
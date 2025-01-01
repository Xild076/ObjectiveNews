import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sentence_transformers import SentenceTransformer
import nltk
nltk.download("punkt_tab")
from nltk import sent_tokenize
from typing import List, Dict, Any, Union, Type
from colorama import Fore
import streamlit as st
from utility import preprocess_text, normalize_values_minmax, SentenceHolder, fix_space_newline

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.positional_encoding = self._positional_encoding(embed_dim, 5000)

    def _positional_encoding(self, embed_dim, max_len):
        p = torch.zeros(max_len, embed_dim)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        p[:, 0::2] = torch.sin(pos * div)
        p[:, 1::2] = torch.cos(pos * div)
        return p.unsqueeze(0).transpose(0, 1)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src = src + self.positional_encoding[: src.size(0), :]
        src2, _ = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        src = self.norm1(src + self.dropout1(src2))
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = self.norm2(src + self.dropout2(src2))
        return src

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

sentence_embed_model = load_model()
attention_model = SelfAttention(embed_dim=384, num_heads=4)

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
    e = sentence_embed_model.encode(sentences)
    e = torch.tensor(e)
    if attention:
        e = e.unsqueeze(1)
        att_out = attention_model(e)
        e = att_out.squeeze(1).detach().numpy()
    else:
        e = e.numpy()
    f = []
    for i, emb in enumerate(e):
        add = emb
        if context:
            start = max(0, i - context_len)
            end = min(len(e), i + context_len + 1)
            c_emb = e[start:end].mean(axis=0)
            add = c_emb
        final = emb * weights["single"] + add * weights["context"]
        f.append(final)
    return np.array(f)

def find_representative_sentence(X, labels, cluster_label):
    from sklearn.metrics.pairwise import cosine_similarity
    idxs = np.where(labels == cluster_label)[0]
    c_points = X[idxs]
    if len(c_points) == 0:
        raise ValueError(f"No points found for cluster {cluster_label}.")
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
    clustering_method: Union[Type[AgglomerativeClustering], Type[KMeans]] = KMeans,
    score_weights={"sil": 0.4, "db": 0.2, "ch": 0.4}
):
    sents = [s.text for s in sentences_holder]
    X = encode_text(sents, weights, context, context_len, preprocess, attention)
    max_clusters = min(max_clusters, len(X))
    c_scores = []
    for i in range(2, max_clusters):
        c = clustering_method(n_clusters=i)
        lbls = c.fit_predict(X)
        sil = silhouette_score(X, lbls)
        db = davies_bouldin_score(X, lbls)
        ch = calinski_harabasz_score(X, lbls)
        c_scores.append({"n_clusters": i, "sil": sil, "db": db, "ch": ch, "labels": lbls})
    sil_vals = [c["sil"] for c in c_scores]
    db_vals = [c["db"] for c in c_scores]
    ch_vals = [c["ch"] for c in c_scores]
    norm_sil = normalize_values_minmax(sil_vals, reverse=False)
    norm_db = normalize_values_minmax(db_vals, reverse=True)
    norm_ch = normalize_values_minmax(ch_vals, reverse=False)
    b = float("-inf")
    best = None
    for i, c in enumerate(c_scores):
        sc = (
            norm_sil[i] * score_weights["sil"]
            + norm_db[i] * score_weights["db"]
            + norm_ch[i] * score_weights["ch"]
        )
        if sc > b:
            b = sc
            best = c
    lbls = best["labels"]
    sil, db, ch = best["sil"], best["db"], best["ch"]
    clusters = []
    u_lbls = sorted(set(lbls))
    from sklearn.metrics.pairwise import cosine_similarity
    for cl in u_lbls:
        c_sents = [sentences_holder[i] for i, lb in enumerate(lbls) if lb == cl]
        rep_idx = find_representative_sentence(X, lbls, cl)
        rep_s = sentences_holder[rep_idx]
        c_idxs = [i for i, lb in enumerate(lbls) if lb == cl]
        c_embs = X[c_idxs]
        rep_emb = X[rep_idx].reshape(1, -1)
        sims = cosine_similarity(c_embs, rep_emb).flatten()
        sorted_idx = np.argsort(-sims)
        sorted_sents = [c_sents[i] for i in sorted_idx]
        if context:
            rep_context = SentenceHolder(
                get_sentence_with_context(sents, rep_idx, context_len),
                source=rep_s.source,
                author=rep_s.author,
                date=rep_s.date
            )
        else:
            rep_context = rep_s
        clusters.append({
            "cluster_id": int(cl),
            "sentences": sorted_sents,
            "representative": rep_s,
            "representative_with_context": rep_context
        })
    return {
        "clusters": clusters,
        "metrics": {
            "silhouette": float(sil),
            "davies_bouldin": float(db),
            "calinski_harabasz": float(ch),
            "score": float(b)
        }
    }

def visualize_grouping(text):
    s = sent_tokenize(text)
    print("Sentence Length:", len(s))
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
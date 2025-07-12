import numpy as np
import hdbscan
import warnings
import os
import ast
import json
from sentence_transformers import SentenceTransformer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from typing import Literal, Tuple, Dict, List, Any
import umap
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, homogeneity_score, completeness_score, v_measure_score

warnings.filterwarnings("ignore", module="^sklearn")

print("Loading models...")
model = SentenceTransformer("all-MiniLM-L6-v2")
lemmatizer = WordNetLemmatizer()
print("Models loaded.")

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, embed_dim = x.shape
        
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        attended_values = torch.matmul(attention_weights, V)
        
        attended_values = attended_values.transpose(1, 2).contiguous().view(
            batch_size, seq_len, embed_dim
        )
        
        output = self.out_proj(attended_values)
        
        return output, attention_weights

def clean_text(text):
    text = text.replace("\n", " ").replace("\t", " ").replace("\r", " ")
    return text

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    text = clean_text(text)
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens
              if word.isalpha() and word not in stop_words]
    return ' '.join(tokens) if tokens else text

def split_sentences(text):
    sentences = sent_tokenize(text)
    return [clean_text(sentence) for sentence in sentences]

def encode_text(sentences,
                weights,
                context,
                context_len,
                preprocess,
                attention=False):
    if not sentences or len(sentences) == 0:
        return np.array([])
        
    if preprocess:
        sentences = [preprocess_text(sentence) for sentence in sentences]
        sentences = [s for s in sentences if s and s.strip()]
        if not sentences:
            return np.array([])
            
    embeddings = model.encode(sentences, show_progress_bar=False)
    
    if not context:
        return embeddings
    
    final_embeddings = []
    context_window = context_len
    for i, emb in enumerate(embeddings):
        start_index = max(0, i - context_window)
        end_index = min(len(embeddings), i + context_window + 1)
        context_embeddings = embeddings[start_index:end_index]
        context_mean = context_embeddings.mean(axis=0)
        
        weighted_emb = (emb * weights["single"]) + (context_mean * weights["context"])
        final_embeddings.append(weighted_emb)
    
    return np.array(final_embeddings)

def dim_reduction(embeddings, n_neighbors=15, n_components=2, metric:Literal['euclidean', 'manhattan', 'cosine', 'correlation', 'chebyshev']='cosine'):
    reducer = umap.UMAP(n_neighbors=n_neighbors,
                        n_components=n_components,
                        metric=metric)
    reduced_embeddings = reducer.fit_transform(embeddings)
    return reduced_embeddings

def cluster_embeddings(embeddings, 
                       metric:Literal['euclidean', 'manhattan', 'cosine', 'hamming', 'jaccard', 'canberra', 'braycurtis']='cosine', 
                       algorithm:Literal['best', 'generic', 'prims_kdtree', 'prims_balltree', 'boruvka_kdtree', 'boruvka_balltree']='best',
                       cluster_selection_method:Literal['leaf', 'eom']='eom'):
    
    if metric == 'cosine' and algorithm in ['prims_balltree', 'boruvka_balltree']:
        algorithm = 'best'
    
    cluster = hdbscan.HDBSCAN(min_cluster_size=2,
                              metric=metric,
                              algorithm=algorithm,
                              cluster_selection_method=cluster_selection_method,
                              )
    return cluster.fit_predict(embeddings)

def load_samples(file_path='src/grouping_data'):
    paths = os.listdir(file_path)
    data = []
    for path in paths:
        general_folder = os.path.join(file_path, path)
        text = open(os.path.join(general_folder, 'text.txt'), 'r').read()
        clusters = open(os.path.join(general_folder, 'clusters.txt'), 'r').read()
        sentences = split_sentences(text)
        clusters = ast.literal_eval(clusters)

        if len(sentences) != len(clusters):
            print("Clusters and sentences length mismatch in file, skipping:", path)
            continue

        data.append({
            'sentences': sentences,
            'clusters': clusters
        })
    return data

def cluster_texts(sentences,
                  weights=0.1,
                  context:bool = True,
                  context_len:int = 5,
                  preprocess:bool = True,
                  attention:bool = False,
                  norm:Literal['l1', 'l2', 'max', 'none'] = 'l2',
                  n_neighbors:int = 15,
                  n_components:int = 2,
                  umap_metric:Literal['euclidean', 'manhattan', 'cosine', 'correlation', 'chebyshev'] = 'cosine',
                  cluster_metric:Literal['euclidean', 'manhattan', 'cosine', 'hamming', 'jaccard', 'canberra', 'braycurtis'] = 'cosine',
                  algorithm:Literal['best', 'generic', 'prims_kdtree', 'prims_balltree', 'boruvka_kdtree', 'boruvka_balltree'] = 'best',
                  cluster_selection_method:Literal['leaf', 'eom'] = 'eom'):
    
    if not sentences or len(sentences) == 0:
        return []
    
    if len(sentences) == 1:
        return [0]
    
    weights_new = {
        "single": weights,
        "context": 1 - weights
    }
    embeddings = encode_text(sentences, weights_new, context, context_len, preprocess, attention)
    
    if embeddings is None or len(embeddings) == 0:
        return []
    
    if len(embeddings) == 1:
        return [0]
    
    if norm != 'none':
        embeddings = normalize(embeddings, norm=norm)
    
    actual_n_neighbors = min(n_neighbors, len(embeddings) - 1)
    if actual_n_neighbors < 1:
        actual_n_neighbors = 1
        
    embeddings = dim_reduction(embeddings, actual_n_neighbors, n_components, umap_metric)
    clusters = cluster_embeddings(embeddings, metric=cluster_metric, algorithm=algorithm, cluster_selection_method=cluster_selection_method)
    return clusters.tolist()

def reward_function(cluster_pred, cluster_true):
    ari = adjusted_rand_score(cluster_true, cluster_pred)
    nmi = normalized_mutual_info_score(cluster_true, cluster_pred)
    homogeneity = homogeneity_score(cluster_true, cluster_pred)
    completeness = completeness_score(cluster_true, cluster_pred)
    v_measure = v_measure_score(cluster_true, cluster_pred)

    ari_norm = (ari + 1) / 2

    composite_score = (
        0.3 * ari_norm +
        0.3 * nmi +
        0.15 * homogeneity +
        0.15 * completeness +
        0.1 * v_measure
    )
    return composite_score

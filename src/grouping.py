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
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utility import clean_text, SentenceHolder, split_sentences

warnings.filterwarnings("ignore", module="^sklearn")

print("Loading models...")
model = SentenceTransformer("all-MiniLM-L6-v2")
lemmatizer = WordNetLemmatizer()
print("Models loaded.")

class SelfAttentionModel(nn.Module):
    def __init__(self, embed_dim=384, num_heads=8, num_layers=2, dropout=0.1):
        super(SelfAttentionModel, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in range(num_layers)
        ])
        
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        
        self.final_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        
        for attention, norm in zip(self.attention_layers, self.layer_norms):
            attn_output, _ = attention(x, x, x)
            x = norm(x + self.dropout(attn_output))
        
        ffn_output = self.ffn(x)
        x = self.final_norm(x + self.dropout(ffn_output))
        
        return x.squeeze(0) if x.size(0) == 1 else x

def load_attention_model(model_path=None):
    if model_path is None:
        model_paths = [
            "models/best_attention_model.pth",
            "models/self_attention_model.pth",
            "models/final_attention_model.pth"
        ]
        model_path = next((path for path in model_paths if os.path.exists(path)), None)
    
    if model_path and os.path.exists(model_path):
        try:
            attention_model = SelfAttentionModel(embed_dim=384)
            state_dict = torch.load(model_path, map_location='cpu')
            if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                attention_model.load_state_dict(state_dict['model_state_dict'])
            else:
                attention_model.load_state_dict(state_dict)
            attention_model.eval()
            print(f"Loaded attention model from {model_path}")
            return attention_model
        except Exception as e:
            print(f"Failed to load model from {model_path}: {e}")
            return None
    return None

attention_model = load_attention_model()

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    text = clean_text(text)
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens
              if word.isalpha() and word not in stop_words]
    return ' '.join(tokens) if tokens else text
    abbreviations = {'Mr.', 'Mrs.', 'Dr.', 'Jr.', 'Sr.', 'vs.', 'etc.', 'i.e.', 'e.g.', 'U.S.'}
    opener_to_closer = {'"': '"', '“': '”'}
    closers = {v: k for k, v in opener_to_closer.items()}
    stack = []
    sentences = []
    buffer = []
    i = 0
    n = len(text)

    while i < n:
        ch = text[i]
        buffer.append(ch)

        if ch in opener_to_closer and not stack:
            stack.append(opener_to_closer[ch])
        elif stack and ch == stack[-1]:
            stack.pop()

        if ch in '.?!' and not stack:
            j = i + 1
            while j < n and text[j] in '.?!':
                buffer.append(text[j])
                j += 1

            tail = ''.join(buffer).strip().split()[-1]
            if tail not in abbreviations:
                if j == n or text[j].isspace():
                    sentences.append(''.join(buffer).strip())
                    buffer = []
                    i = j - 1

        i += 1

    rem = ''.join(buffer).strip()
    if rem:
        sentences.append(rem)

    return sentences

def encode_text_with_attention(embeddings, att_model):
    if att_model is None:
        raise ValueError("Attention model is not loaded or provided.")
    
    device = next(att_model.parameters()).device
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        refined_embeddings = att_model(embeddings_tensor)
        
    return refined_embeddings.detach().cpu().numpy()

def encode_text(sentences, weights, context, context_len, preprocess, attention, att_model):
    if not sentences or len(sentences) == 0:
        return np.array([])
    if preprocess:
        sentences = [preprocess_text(sentence) for sentence in sentences]
        sentences = [s for s in sentences if s and s.strip()]
        if not sentences:
            return np.array([])
    embeddings = model.encode(sentences, show_progress_bar=False)
    if attention and att_model is not None:
        embeddings = encode_text_with_attention(embeddings, att_model)
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

def dim_reduction(embeddings, n_neighbors=15, n_components=2, metric='cosine'):
    reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=n_components, metric=metric)
    reduced_embeddings = reducer.fit_transform(embeddings)
    return reduced_embeddings

def cluster_embeddings(embeddings, metric='cosine', algorithm='best', cluster_selection_method='eom', min_cluster_size=2, min_samples=None):
    if metric in ['cosine', 'hamming', 'jaccard', 'canberra', 'braycurtis']:
        if algorithm in ['prims_balltree', 'boruvka_balltree', 'best']:
            algorithm = 'generic'
    embeddings = embeddings.astype(np.float64)
    cluster = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric=metric, algorithm=algorithm, cluster_selection_method=cluster_selection_method)
    return cluster.fit_predict(embeddings)

def load_samples(file_path='data/grouping_data'):
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
        data.append({'sentences': sentences, 'clusters': clusters})
    return data

def find_representative_sentence(X: np.ndarray, labels: np.ndarray, cluster_label: int) -> int:
    cluster_indices = np.where(labels == cluster_label)[0]
    if len(cluster_indices) == 0:
        raise ValueError(f"No points found for cluster {cluster_label}.")
    cluster_points = X[cluster_indices]
    centroid = np.mean(cluster_points, axis=0, keepdims=True)
    similarities = cosine_similarity(cluster_points, centroid).flatten()
    rep_relative_idx = np.argmax(similarities)
    rep_idx = cluster_indices[rep_relative_idx]
    return rep_idx

def cluster_sentences(sentences, att_model=None, weights=0.1, context:bool = True, context_len:int = 5, preprocess:bool = True, attention:bool = False, norm:str = 'l2', n_neighbors:int = 15, n_components:int = 2, umap_metric:str = 'cosine', cluster_metric:str = 'cosine', algorithm:str = 'best', cluster_selection_method:str = 'eom', min_cluster_size:int = 2, min_samples:int = 1):
    if not sentences or len(sentences) == 0:
        return []
    if len(sentences) == 1:
        return [0]
    weights_new = {"single": weights, "context": 1 - weights}
    if attention and att_model is None:
        att_model = load_attention_model()
    embeddings = encode_text(sentences, weights_new, context, context_len, preprocess, attention, att_model)
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
    clusters = cluster_embeddings(embeddings, metric=cluster_metric, algorithm=algorithm, cluster_selection_method=cluster_selection_method, min_cluster_size=min_cluster_size, min_samples=min_samples)
    return clusters.tolist()

def reward_function(cluster_pred, cluster_true):
    ari = adjusted_rand_score(cluster_true, cluster_pred)
    nmi = normalized_mutual_info_score(cluster_true, cluster_pred)
    homogeneity = homogeneity_score(cluster_true, cluster_pred)
    completeness = completeness_score(cluster_true, cluster_pred)
    v_measure = v_measure_score(cluster_true, cluster_pred)
    ari_norm = (ari + 1) / 2
    composite_score = (0.3 * ari_norm + 0.3 * nmi + 0.15 * homogeneity + 0.15 * completeness + 0.1 * v_measure)
    return composite_score

OPTIMAL_CLUSTERING_PARAMS = {
    "weights": 0.4088074527809119,
    "context": True,
    "context_len": 2,
    "preprocess": False,
    "attention": True,
    "norm": 'max',
    "n_neighbors": 6,
    "n_components": 4,
    "umap_metric": 'correlation',
    "cluster_metric": 'euclidean',
    "algorithm": 'prims_kdtree',
    "cluster_selection_method": 'leaf',
    "min_cluster_size": 2,
    "min_samples": None
}

def cluster_texts(sentences: List[SentenceHolder], params=OPTIMAL_CLUSTERING_PARAMS):
    if not sentences or len(sentences) == 0:
        return []

    sentences_text = [sentence.text for sentence in sentences]
    cluster_labels = cluster_sentences(
        sentences_text,
        att_model=attention_model,
        **params
    )

    cluster_dicts = []
    unique_labels = sorted(l for l in set(cluster_labels) if l != -1)

    assigned_indices = set()

    weights_new = {"single": params.get("weights", 0.1), "context": 1 - params.get("weights", 0.1)}
    embeddings = encode_text(
        sentences_text,
        weights_new,
        params.get("context", True),
        params.get("context_len", 5),
        params.get("preprocess", True),
        params.get("attention", False),
        attention_model
    )

    for c in unique_labels:
        cluster_indices = [i for i, label in enumerate(cluster_labels) if label == c and i not in assigned_indices]
        assigned_indices.update(cluster_indices)
        cluster_sents = [sentences[i] for i in cluster_indices]
        if len(cluster_indices) == 0:
            rep_idx = None
        else:
            rep_idx = find_representative_sentence(embeddings, np.array(cluster_labels), c)
        representative_sentence = sentences[rep_idx] if rep_idx is not None else None
        cluster_dicts.append({
            "label": c,
            "sentences": cluster_sents,
            "representative": representative_sentence
        })
    return cluster_dicts
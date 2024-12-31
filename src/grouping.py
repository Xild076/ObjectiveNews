from utility import DLog

logger = DLog(name="Grouping", level="DEBUG", log_dir="logs")

logger.info("Importing modules...")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

from sentence_transformers import SentenceTransformer

logger.info("Downloading NLTK...")
import nltk
nltk.download('punkt_tab')
from nltk import sent_tokenize
logger.info("NLTK downloaded...")

from typing import List, Dict, Any, Union, Type
import random
import math
from colorama import Fore

from utility import preprocess_text, normalize_values_minmax, SentenceHolder, fix_space_newline
logger.info("Modules imported...")

logger.info("Setting seeds...")
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
logger.info("Seeds set...")

class SelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dim_feedforward=2048, dropout=0.1):
        super(SelfAttention, self).__init__()
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
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        return pe

    def forward(self, src: torch.Tensor, src_mask=None, src_key_padding_mask=None) -> torch.Tensor:
        src = src + self.positional_encoding[:src.size(0), :]
        src2, _ = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

logger.info("Loading models...")
sentence_embed_model = SentenceTransformer('all-MiniLM-L6-v2')
attention_model = SelfAttention(embed_dim=384, num_heads=4)
logger.info("Models loaded...")

def encode_text(sentences: List[str],
                weights: Dict[str, float] = {'single': 0.7, 'context': 0.3},
                context: bool = False,
                context_len: int = 1,
                preprocess=True,
                attention=True) -> np.ndarray:
    if preprocess:
        sentences = [preprocess_text(sent) for sent in sentences]
    
    embeddings = sentence_embed_model.encode(sentences)
    embeddings = torch.tensor(embeddings)

    if attention:
        embeddings = embeddings.unsqueeze(1)
        attention_output = attention_model(embeddings)
        embeddings = attention_output.squeeze(1).detach().numpy()

    final_embeddings = []
    for i, embedding in enumerate(embeddings):
        addition = embedding
        if context:
            start_index = max(0, i - context_len)
            end_index = min(len(embeddings), i + context_len + 1)
            context_embeddings = embeddings[start_index:end_index]
            embedding_context = context_embeddings.mean(axis=0)
            addition = embedding_context

        final_embedding = (embedding * weights['single']) + (addition * weights['context'])
        final_embeddings.append(final_embedding)

    return np.array(final_embeddings)

def find_representative_sentence(X: np.ndarray, labels: np.ndarray, cluster_label: int) -> int:
    from sklearn.metrics.pairwise import cosine_similarity
    cluster_indices = np.where(labels == cluster_label)[0]
    cluster_points = X[cluster_indices]
    if len(cluster_points) == 0:
        raise ValueError(f"No points found for cluster {cluster_label}.")
    centroid = cluster_points.mean(axis=0).reshape(1, -1)
    similarities = cosine_similarity(cluster_points, centroid).flatten()
    rep_relative_idx = np.argmax(similarities)
    rep_idx = cluster_indices[rep_relative_idx]
    return rep_idx

def get_sentence_with_context(texts: List[str], idx: int, context_len: int) -> str:
    start = max(0, idx - context_len)
    end = min(len(texts), idx + context_len + 1)
    context_sentences = texts[start:end]
    return " ".join(context_sentences)

def observe_best_cluster(sentences_holder: List[SentenceHolder],
                         max_clusters: int = 10,
                         weights: Dict[str, float] = {'single': 0.7, 'context': 0.3},
                         context: bool = False,
                         context_len: int = 1,
                         preprocess=True,
                         attention=True,
                         clustering_method: Union[Type[AgglomerativeClustering], Type[KMeans]] = KMeans,
                         score_weights = {'sil': 0.4, 'db': 0.2, 'ch': 0.4}) -> Dict[Any, Any]:
    sentences = [sent.text for sent in sentences_holder]
    X = encode_text(sentences, weights, context, context_len, preprocess, attention)

    max_clusters = min(max_clusters, len(X))

    cluster_scores = []
    for i in range(2, max_clusters):
        clusterer = clustering_method(n_clusters=i)
        labels = clusterer.fit_predict(X)
        sil = silhouette_score(X, labels)
        db = davies_bouldin_score(X, labels)
        ch = calinski_harabasz_score(X, labels)
        cluster_scores.append({
            'n_clusters': i, 
            'sil': sil, 
            'db': db, 
            'ch': ch, 
            'labels': labels
        })
    
    sil_values = [c['sil'] for c in cluster_scores]
    db_values = [c['db'] for c in cluster_scores]
    ch_values = [c['ch'] for c in cluster_scores]

    norm_sil = normalize_values_minmax(sil_values, reverse=False)
    norm_db = normalize_values_minmax(db_values, reverse=True)
    norm_ch = normalize_values_minmax(ch_values, reverse=False)

    best_score = float('-inf')
    best_cluster = None
    for i, c in enumerate(cluster_scores):
        score = (norm_sil[i] * score_weights['sil']) + \
                (norm_db[i] * score_weights['db']) + \
                (norm_ch[i] * score_weights['ch'])
        if score > best_score:
            best_score = score
            best_cluster = c

    lbls = best_cluster['labels']
    sil = best_cluster['sil']
    db = best_cluster['db']
    ch = best_cluster['ch']

    cluster_dicts = []
    unique_labels = sorted(set(lbls))
    for c in unique_labels:
        cluster_sentences = [sentences_holder[i] for i, label in enumerate(lbls) if label == c]
        rep_index = find_representative_sentence(X, lbls, c)
        rep_sentence = sentences_holder[rep_index]
        rep_embedding = X[rep_index].reshape(1, -1)
        cluster_indices = [i for i, label in enumerate(lbls) if label == c]
        cluster_embeddings = X[cluster_indices]
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(cluster_embeddings, rep_embedding).flatten()
        sorted_indices = np.argsort(-similarities)
        sorted_cluster_sentences = [cluster_sentences[i] for i in sorted_indices]
        if context:
            rep_with_context = SentenceHolder(get_sentence_with_context(sentences, 
                                                                        rep_index, 
                                                                        context_len),
                                            source=rep_sentence.source, 
                                            author=rep_sentence.author, 
                                            date=rep_sentence.date)
        else:
            rep_with_context = rep_sentence

        cluster_dicts.append({
            'cluster_id': int(c),
            'sentences': sorted_cluster_sentences,
            'representative': rep_sentence,
            'representative_with_context': rep_with_context
        })
    
    return {
        'clusters': cluster_dicts,
        'metrics': {
            'silhouette': float(sil), 
            'davies_bouldin': float(db), 
            'calinski_harabasz': float(ch), 
            'score': float(best_score)
        }
    }

def visualize_grouping(text):
    sentences = sent_tokenize(text)
    print("Sentence Length: " + str(len(sentences)))
    sentences = [SentenceHolder(text=sent) for sent in sentences]
    clusters = observe_best_cluster(sentences, max_clusters=8, 
                                    context=True, context_len=1,
                                    weights={'single':0.7, 'context':0.3},
                                    preprocess=True, attention=True, 
                                    clustering_method=KMeans,
                                    score_weights={'sil':0.4, 'db':0.2, 'ch':0.4})['clusters']

    for cluster in clusters:
        print(Fore.BLUE + f"Cluster {cluster['cluster_id']}:" + Fore.RESET)
        print(Fore.LIGHTGREEN_EX + "Representative: " + Fore.RESET + fix_space_newline(cluster['representative'].text))
        print(Fore.LIGHTCYAN_EX + f"Representative with context: " + Fore.RESET + fix_space_newline(cluster['representative_with_context'].text))
        print(Fore.LIGHTYELLOW_EX + f"Sentences:" + Fore.RESET)
        for sent in cluster['sentences']:
            print(fix_space_newline(sent.text))
        print("\n")
    

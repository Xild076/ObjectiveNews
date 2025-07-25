import numpy as np
import hdbscan
import warnings
import os
import ast
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_samples
from typing import Dict, List, Any
from umap import UMAP
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, homogeneity_score, completeness_score, v_measure_score
import torch
import torch.nn as nn
from utility import clean_text, SentenceHolder, split_sentences, normalize_values_minmax, DLog, load_sent_transformer, load_lemma, cache_data_decorator, cache_resource_decorator
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from typing import Union, Type

warnings.filterwarnings("ignore", module="^sklearn")
logger = DLog("GROUPING", "DEBUG")

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

@cache_resource_decorator
def load_attention_model(model_path=None):
    logger.info("Loading Attention model...")
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
            logger.info(f"Loaded attention model from {model_path}")
            return attention_model
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            return None
    return None

attention_model = load_attention_model()

@cache_data_decorator
def preprocess_text(text):
    logger.info("Preprocessing text...")
    lemmatizer = load_lemma()
    stop_words = set(stopwords.words('english'))
    text = clean_text(text)
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens
              if word.isalpha() and word not in stop_words]
    return ' '.join(tokens) if tokens else text

@cache_data_decorator
def encode_text_with_attention(embeddings, _att_model):
    logger.info("Encoding text with attention...")
    if _att_model is None:
        raise ValueError("Attention model is not loaded or provided.")
    device = next(_att_model.parameters()).device
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32).to(device)
    with torch.no_grad():
        refined_embeddings = _att_model(embeddings_tensor)
    return refined_embeddings.detach().cpu().numpy()

@cache_data_decorator
def encode_text(sentences, weights, context, context_len, preprocess, attention, _att_model):
    logger.info("Encoding text...")
    model = load_sent_transformer()
    if not sentences or len(sentences) == 0:
        return np.array([])
    if preprocess:
        sentences = [preprocess_text(sentence) for sentence in sentences]
        sentences = [s for s in sentences if s and s.strip()]
        if not sentences:
            return np.array([])
    embeddings = model.encode(sentences, show_progress_bar=False)
    if attention and _att_model is not None:
        embeddings = encode_text_with_attention(embeddings, _att_model)
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

@cache_data_decorator
def dim_reduction(embeddings, n_neighbors=15, n_components=2, metric='cosine'):
    logger.info("Dimensional Reduction...")
    n_samples = len(embeddings)
    if n_samples < 3:
        return embeddings
    safe_n_neighbors = min(n_neighbors, n_samples - 1)
    safe_n_components = min(n_components, n_samples - 1)
    if safe_n_neighbors < 1:
        safe_n_neighbors = 1
    if safe_n_components < 1:
        safe_n_components = 1
    try:
        reducer = UMAP(n_neighbors=safe_n_neighbors, n_components=safe_n_components, metric=metric)
        reduced_embeddings = reducer.fit_transform(embeddings)
        return reduced_embeddings
    except Exception as e:
        logger.error(f"Failed reduction: {str(e)[:50]}")
        return embeddings

def cluster_embeddings(embeddings, metric='cosine', algorithm='best', cluster_selection_method='eom', min_cluster_size=2, min_samples=None):
    logger.info("Clustering embeddings...")
    if metric in ['cosine', 'hamming', 'jaccard', 'canberra', 'braycurtis']:
        if algorithm in ['prims_balltree', 'boruvka_balltree', 'best']:
            algorithm = 'generic'
    embeddings = embeddings.astype(np.float64)
    cluster = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric=metric, algorithm=algorithm, cluster_selection_method=cluster_selection_method)
    return cluster.fit_predict(embeddings)

@cache_resource_decorator
def load_samples(file_path='data/grouping_data'):
    logger.info("Loading samples...")
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
    logger.info("Finding representative sentences...")
    cluster_indices = np.where(labels == cluster_label)[0]
    if len(cluster_indices) == 0:
        raise ValueError(f"No points found for cluster {cluster_label}.")
    cluster_points = X[cluster_indices]
    centroid = np.mean(cluster_points, axis=0, keepdims=True)
    similarities = cosine_similarity(cluster_points, centroid).flatten()
    rep_relative_idx = np.argmax(similarities)
    rep_idx = cluster_indices[rep_relative_idx]
    return rep_idx

def cluster_sentences(sentences, _att_model=None, weights=0.1, context:bool = True, context_len:int = 5, preprocess:bool = True, attention:bool = False, norm:str = 'l2', n_neighbors:int = 15, n_components:int = 2, umap_metric:str = 'cosine', cluster_metric:str = 'cosine', algorithm:str = 'best', cluster_selection_method:str = 'eom', min_cluster_size:int = 2, min_samples:int = 1):
    logger.info("Clustering sentences...")
    if not sentences or len(sentences) == 0:
        return [0]
    if len(sentences) == 1:
        return [0]
    weights_new = {"single": weights, "context": 1 - weights}
    if attention and _att_model is None:
        _att_model = load_attention_model()
    embeddings = encode_text(sentences, weights_new, context, context_len, preprocess, attention, _att_model)
    if embeddings is None or len(embeddings) == 0:
        return [0]
    if len(embeddings) == 1:
        return [0]
    if norm != 'none':
        embeddings = normalize(embeddings, norm=norm)
    n_samples = len(embeddings)
    actual_n_neighbors = min(n_neighbors, max(1, n_samples - 1))
    actual_n_components = min(n_components, max(1, n_samples - 1))
    embeddings = dim_reduction(embeddings, actual_n_neighbors, actual_n_components, umap_metric)
    try:
        clusters = cluster_embeddings(embeddings, metric=cluster_metric, algorithm=algorithm, cluster_selection_method=cluster_selection_method, min_cluster_size=min_cluster_size, min_samples=min_samples)
        if clusters is None or len(clusters) == 0 or all(label == -1 for label in clusters):
            return [0] * len(sentences)
        return clusters.tolist()
    except Exception as e:
        return [0] * len(sentences)

def reward_function(cluster_pred, cluster_true):
    logger.info("Calculating reward...")
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
    logger.info("Clustering texts...")
    if not sentences or len(sentences) == 0:
        return [{
            "label": 0,
            "sentences": [],
            "representative": None,
            "representative_with_context": (None, None)
        }]
    sentences_text = [sentence.text for sentence in sentences]
    cluster_labels = cluster_sentences(
        sentences_text,
        _att_model=attention_model,
        **params
    )
    if all(label == -1 for label in cluster_labels):
        cluster_labels = [0] * len(sentences)
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
            "representative": representative_sentence,
            "representative_with_context": (
                representative_sentence,
                SentenceHolder(
                    " ".join([sents.text for sents in sentences[max(0, rep_idx - params.get("context_len", 1)):min(len(sentences), rep_idx + 1 + params.get("context_len", 1))]]),
                    representative_sentence.source,
                    representative_sentence.author,
                    representative_sentence.date
                )
            )
        })
    return cluster_dicts

def observe_best_cluster(sentences_holder: List[SentenceHolder],
                         max_clusters: int = 10,
                         weights: Dict[str, float] = {'single': 0.7, 'context': 0.3},
                         context: bool = False,
                         context_len: int = 1,
                         preprocess=True,
                         attention=True,
                         clustering_method: Union[Type[AgglomerativeClustering], Type[KMeans]] = KMeans,
                         score_weights = {'sil': 0.4, 'db': 0.2, 'ch': 0.4}) -> Dict[Any, Any]:
    logger.info("Observing best clsuters...")
    sentences = [sent.text for sent in sentences_holder]
    X = encode_text(sentences, weights, context, context_len, preprocess, attention, attention_model)

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
        similarities = cosine_similarity(cluster_embeddings, rep_embedding).flatten()
        sorted_indices = np.argsort(-similarities)
        sorted_cluster_sentences = [cluster_sentences[i] for i in sorted_indices]
        if context:
            rep_with_context = SentenceHolder(" ".join([sents.text for sents in sentences_holder[max(0, rep_index - context_len):min(len(sentences_holder), rep_index + 1 + context_len)]]),
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

def group_representative_sentences(rep_sentences: List,
                                   min_cluster_size: int = 3,
                                   min_avg_sim: float = 0.25,
                                   min_silhouette: float = 0.05,
                                   max_sim_std: float = 0.13):
    model = load_sent_transformer()
    reps_with_context, reps_no_context = [], []
    for item in rep_sentences:
        if isinstance(item, tuple):
            rep, ctx = item
            c = ctx if hasattr(ctx, 'text') else SentenceHolder(ctx, rep.source, rep.author, rep.date)
            reps_with_context.append(c)
            reps_no_context.append(rep)
        else:
            reps_with_context.append(item)
            reps_no_context.append(item)

    if len(reps_no_context) < min_cluster_size:
        if not reps_no_context:
            return []
        return [{
            'label': 0,
            'sentences': reps_no_context,
            'representative': reps_no_context[0],
            'representative_with_context': (reps_no_context[0], reps_with_context[0])
        }]

    texts = [s.text for s in reps_with_context]
    embeddings = model.encode(texts, show_progress_bar=False)
    normed = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    n_samples = len(normed)

    if n_samples <= 1:
        return []

    umap_neighbors = min(10, n_samples - 1)
    umap_components = min(5, n_samples - 1)
    
    if umap_components == 0:
        umap_components = 1

    reducer = UMAP(n_neighbors=umap_neighbors,
                   n_components=umap_components,
                   metric='cosine',
                   random_state=42)
    reduced = reducer.fit_transform(normed)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                min_samples=max(1, min_cluster_size // 2),
                                metric='euclidean',
                                allow_single_cluster=True)
    labels = clusterer.fit_predict(reduced)

    unique_labels = np.unique(labels)
    if len(unique_labels) > 1 and not (len(unique_labels) == 2 and -1 in unique_labels):
        sil_scores = silhouette_samples(normed, labels, metric='cosine')
    else:
        sil_scores = np.ones(n_samples)

    sim_matrix = cosine_similarity(embeddings)
    clusters = []
    
    noise_indices = list(np.where(labels == -1)[0])
    
    for lbl in set(labels) - {-1}:
        idxs = list(np.where(labels == lbl)[0])
        sub = sim_matrix[np.ix_(idxs, idxs)]
        pairs = np.triu_indices(len(sub), k=1)
        
        is_coherent = True
        if pairs[0].size > 0:
            pairwise_sims = sub[pairs]
            avg_sim = pairwise_sims.mean()
            sim_std = pairwise_sims.std()
            
            if (avg_sim < min_avg_sim or 
                sim_std > max_sim_std or 
                sil_scores[idxs].mean() < min_silhouette):
                is_coherent = False
        
        if is_coherent:
            rep_i = idxs[np.argmax(sub.sum(axis=1))]
            clusters.append({
                'label': int(lbl),
                'sentences': [reps_no_context[i] for i in idxs],
                'representative': reps_no_context[rep_i],
                'representative_with_context': (
                    reps_no_context[rep_i],
                    reps_with_context[rep_i]
                )
            })
        else:
            noise_indices.extend(idxs)

    if noise_indices and not clusters:
        noise_sentences_no_ctx = [reps_no_context[i] for i in noise_indices]
        noise_sentences_with_ctx = [reps_with_context[i] for i in noise_indices]
        return [{
            'label': 0,
            'sentences': noise_sentences_no_ctx,
            'representative': noise_sentences_no_ctx[0],
            'representative_with_context': (noise_sentences_no_ctx[0], noise_sentences_with_ctx[0])
        }]
    
    if not clusters and not noise_indices:
        return []

    return clusters

def group_individual_articles(article):
    logger.info("Grouping individual articles...")
    rep_sentences = []
    text = article.get('text', '')
    sentences = split_sentences(text)
    sentences = [SentenceHolder(text=text, source=article['source'], author=article['author'], date=article['date']) for text in sentences]

    clusters = cluster_texts(sentences)

    for cluster in clusters:
        rep_sentences.append(cluster['representative_with_context'])
    return rep_sentences
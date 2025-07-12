import numpy as np
import hdbscan
from sentence_transformers import SentenceTransformer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import warnings
from sklearn.preprocessing import normalize
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, homogeneity_score, completeness_score, v_measure_score, silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from typing import Literal, Tuple, Dict, List, Any
import os
import json
import ast
from datetime import datetime
import gc
import threading
import time
from tqdm import tqdm
import re
from collections import Counter
import math
warnings.filterwarnings("ignore")

model = SentenceTransformer('all-mpnet-base-v2')
alternative_models = {
    'news': SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'),
    'general': SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')
}
lemmatizer = WordNetLemmatizer()
_stop_words_cache = {}
_embedding_cache = {}
_cache_size_limit = 500

def cleanup_caches():
    global _embedding_cache
    if len(_embedding_cache) > _cache_size_limit:
        keys_to_remove = list(_embedding_cache.keys())[:-_cache_size_limit//2]
        for key in keys_to_remove:
            del _embedding_cache[key]
    gc.collect()

def memory_cleanup():
    while True:
        time.sleep(30)
        cleanup_caches()

cleanup_thread = threading.Thread(target=memory_cleanup, daemon=True)
cleanup_thread.start()

def get_stop_words(language='english'):
    if language not in _stop_words_cache:
        try:
            _stop_words_cache[language] = set(stopwords.words(language))
        except LookupError:
            _stop_words_cache[language] = set()
    return _stop_words_cache[language]

def split_sentences(text: str) -> list:
    sentences = sent_tokenize(text)
    return [s.strip() for s in sentences if s.strip()]

def preprocess_text(text: str, language='english') -> str:
    stop_words = get_stop_words(language)
    text = text.lower().replace("\n", " ").replace("\t", " ").replace("\r", " ")
    tokens = [lemmatizer.lemmatize(word) for word in word_tokenize(text) 
              if word.isalpha() and word not in stop_words]
    return ' '.join(tokens)

def encode_text(sentences, weights, context, context_len, preprocess, model_type='default'):
    cache_key = (tuple(sentences[:2]) if len(sentences) > 2 else tuple(sentences), 
                 preprocess, context, context_len, model_type)
    
    if cache_key in _embedding_cache:
        embeddings = _embedding_cache[cache_key].copy()
    else:
        if preprocess:
            processed_sentences = [preprocess_text(s) for s in sentences]
            sentences = processed_sentences
        
        selected_model = model
        if model_type in alternative_models:
            selected_model = alternative_models[model_type]
        
        embeddings = selected_model.encode(sentences, show_progress_bar=False, batch_size=8, 
                                         convert_to_numpy=True, normalize_embeddings=False)
        embeddings = embeddings.astype(np.float32, copy=False)
        
        if len(_embedding_cache) < _cache_size_limit:
            _embedding_cache[cache_key] = embeddings.copy()
    
    if not context:
        return embeddings
    
    final_embeddings = np.empty_like(embeddings)
    for i in range(len(embeddings)):
        final_embeddings[i] = adaptive_context_weighting(embeddings, i, context_len, weights)
    
    return final_embeddings

def find_representative_sentence(X: np.ndarray, labels: np.ndarray, cluster_label: int) -> int:
    cluster_indices = np.where(labels == cluster_label)[0]
    if len(cluster_indices) == 0:
        raise ValueError(f"No points found for cluster {cluster_label}.")
    
    if len(cluster_indices) == 1:
        return cluster_indices[0]
    
    cluster_points = X[cluster_indices]
    centroid = cluster_points.mean(axis=0)
    
    distances = np.sum((cluster_points - centroid) ** 2, axis=1)
    rep_relative_idx = np.argmin(distances)
    return cluster_indices[rep_relative_idx]

def cluster_sentences(sentences, weights, 
                      context=False, 
                      context_len=0, 
                      preprocess=True, 
                      metric:Literal['braycurtis', 'canberra', 'chebyshev', 'cityblock', 'dice', 
                                     'euclidean', 'hamming', 'haversine', 'infinity', 'jaccard', 
                                     'kulsinski', 'l1', 'l2', 'mahalanobis', 'manhattan', 'matching', 
                                     'minkowski', 'p', 'pyfunc', 'rogerstanimoto', 'russellrao', 
                                     'seuclidean', 'sokalmichener', 'sokalsneath', 'wminkowski']='euclidean',
                      cluster_selection_method:Literal['leaf', 'eom'] = 'eom',
                      normalization:Literal['l1', 'l2', 'max', 'none'] = 'l2',
                      algorithm:Literal['hdbscan', 'kmeans', 'agglomerative', 'dbscan'] = 'hdbscan',
                      model_type:str = 'default'
                      ):
    if len(sentences) == 0:
        return np.array([]), []

    text_type = detect_text_type(sentences)
    if model_type == 'default':
        model_type = text_type
    
    embeddings = encode_text(sentences, weights, context, context_len, preprocess, model_type)
    
    if normalization != 'none':
        normalize(embeddings, norm=normalization, copy=False)

    n_sentences = len(sentences)
    
    if algorithm == 'hdbscan':
        min_cluster_size = max(2, min(8, n_sentences // 15))
        min_samples = max(1, min(3, min_cluster_size // 2))
        
        try:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric=metric,
                cluster_selection_method=cluster_selection_method,
                memory=None,
                core_dist_n_jobs=1,
                cluster_selection_epsilon=0.05,
                alpha=1.0
            )
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                labels = clusterer.fit_predict(embeddings)
            
            del clusterer
            
        except Exception:
            labels = np.array([i % max(1, n_sentences // 4) for i in range(n_sentences)])
    
    elif algorithm == 'kmeans':
        n_clusters = max(2, min(8, n_sentences // 10))
        try:
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = clusterer.fit_predict(embeddings)
            del clusterer
        except Exception:
            labels = np.array([i % n_clusters for i in range(n_sentences)])
    
    elif algorithm == 'agglomerative':
        n_clusters = max(2, min(8, n_sentences // 12))
        try:
            clusterer = AgglomerativeClustering(n_clusters=n_clusters, metric='cosine', linkage='average')
            labels = clusterer.fit_predict(embeddings)
            del clusterer
        except Exception:
            labels = np.array([i % n_clusters for i in range(n_sentences)])
    
    elif algorithm == 'dbscan':
        try:
            clusterer = DBSCAN(eps=0.3, min_samples=max(2, n_sentences // 20), metric=metric)
            labels = clusterer.fit_predict(embeddings)
            del clusterer
        except Exception:
            labels = np.array([i % max(1, n_sentences // 5) for i in range(n_sentences)])

    unique_labels = np.unique(labels)
    noise_mask = unique_labels != -1
    unique_labels = unique_labels[noise_mask]
    
    n_clusters = len(unique_labels)
    cluster_ratio = n_clusters / n_sentences if n_sentences > 0 else 1
    
    if cluster_ratio > 0.8 or n_clusters == 0:
        optimal_clusters = max(2, min(6, n_sentences // 8))
        labels = np.array([i % optimal_clusters for i in range(n_sentences)])
        unique_labels = np.unique(labels)
    
    representative_indices = []
    for label in unique_labels:
        try:
            rep_idx = find_representative_sentence(embeddings, labels, label)
            representative_indices.append(int(rep_idx))
        except (ValueError, IndexError):
            continue

    return labels.tolist(), representative_indices

def locate_optimal_clustering_metrics(test_data_folder='src/grouping_data'):    
    weight_possibilities = np.array([[1-w, w] for w in np.arange(0.0, 1.1, 0.33)], dtype=np.float32)
    preprocess_options = [True, False]
    context_options = [False, True]
    context_lengths = np.arange(2.0, 8.1, 2.0)
    metrics = ['euclidean', 'manhattan', 'cosine', 'cityblock', 'braycurtis']
    cluster_selection_methods = ['eom', 'leaf']
    normalization_methods = ['l2', 'l1', 'none']
    algorithms = ['hdbscan', 'kmeans', 'agglomerative']
    model_types = ['default', 'news', 'general']

    test_data = []
    for folder in os.listdir(test_data_folder):
        folder_path = os.path.join(test_data_folder, folder)
        if not os.path.isdir(folder_path):
            continue
            
        try:
            with open(os.path.join(folder_path, 'text.txt'), 'r', encoding='utf-8') as f:
                text = f.read()
            with open(os.path.join(folder_path, 'clusters.txt'), 'r', encoding='utf-8') as f:
                clusters = np.array(ast.literal_eval(f.read()), dtype=np.int32)
            
            sentences = split_sentences(text)
            linguistic_features = extract_linguistic_features(sentences)

            if len(sentences) != len(clusters) or len(sentences) < 5:
                continue
                
            test_data.append((folder, sentences, clusters, linguistic_features))
        except (FileNotFoundError, ValueError, SyntaxError, MemoryError):
            continue
    
    if not test_data:
        return None
    
    total_configs = (len(weight_possibilities) * len(preprocess_options) * 
                    len(context_options) * len(context_lengths) * 
                    len(metrics) * len(cluster_selection_methods) * 
                    len(normalization_methods) * len(algorithms) * len(model_types))
    
    run_id = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    best_score = -1
    best_config = None
    config_count = 0
    batch_count = 0
    batch_size = 15
    
    os.makedirs("src/grouping_data/saved_batches", exist_ok=True)
    
    pbar = tqdm(total=total_configs, desc="Processing configs", unit="config", 
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] Best: {postfix}',
                miniters=10)
    pbar.set_postfix_str("N/A")
    
    batch_results = []
    cleanup_counter = 0
    
    for w_idx, weights_arr in enumerate(weight_possibilities):
        weights = {'single': float(weights_arr[0]), 'context': float(weights_arr[1])}
        
        for preprocess_flag in preprocess_options:
            for context in context_options:
                for context_len in context_lengths:
                    for metric in metrics:
                        for cluster_selection_method in cluster_selection_methods:
                            for normalization in normalization_methods:
                                for algorithm in algorithms:
                                    for model_type in model_types:
                                        config_count += 1
                                        cleanup_counter += 1
                                        
                                        all_metrics = []
                                        config = {
                                            'weights': weights,
                                            'preprocess': preprocess_flag,
                                            'context': context,
                                            'context_len': float(context_len),
                                            'metric': metric,
                                            'cluster_selection_method': cluster_selection_method,
                                            'normalization': normalization,
                                            'algorithm': algorithm,
                                            'model_type': model_type
                                        }
                                        
                                        for folder, sentences, clusters, features in test_data:
                                            try:
                                                context_len_param = max(1, min(10
                                                , int(len(sentences) / context_len)))
                                                
                                                with warnings.catch_warnings():
                                                    warnings.simplefilter("ignore")
                                                    labels, _ = cluster_sentences(
                                                        sentences,
                                                        weights=weights,
                                                        context=context,
                                                        context_len=context_len_param,
                                                        preprocess=preprocess_flag,
                                                        metric=metric,
                                                        cluster_selection_method=cluster_selection_method,
                                                        normalization=normalization,
                                                        algorithm=algorithm,
                                                        model_type=model_type
                                                    )
                                                
                                                if len(labels) == len(clusters):
                                                    embeddings = encode_text(sentences, weights, context, context_len_param, preprocess_flag, model_type)
                                                    metrics_dict = calculate_multiple_metrics(clusters, labels, embeddings)
                                                    all_metrics.append(metrics_dict)
                                                
                                            except (Exception, MemoryError):
                                                continue
                                        
                                        if all_metrics:
                                            avg_composite = np.mean([m['composite'] for m in all_metrics])
                                            avg_ari = np.mean([m['ari'] for m in all_metrics])
                                            avg_nmi = np.mean([m['nmi'] for m in all_metrics])
                                            avg_silhouette = np.mean([m['silhouette'] for m in all_metrics])
                                            
                                            result_entry = {
                                                'config_id': config_count,
                                                'config': config,
                                                'composite_score': float(avg_composite),
                                                'ari_score': float(avg_ari),
                                                'nmi_score': float(avg_nmi),
                                                'silhouette_score': float(avg_silhouette),
                                                'valid_tests': len(all_metrics)
                                            }
                                            
                                            if avg_composite > best_score:
                                                best_score = avg_composite
                                                best_config = result_entry
                                                pbar.set_postfix_str(f"{best_score:.4f}")
                                            
                                            batch_results.append(result_entry)
                                
                                        pbar.update(1)
                                        
                                        if len(batch_results) >= batch_size:
                                            batch_count += 1
                                            batch_filename = f"src/grouping_data/saved_batches/clustering_batch_{run_id}_{batch_count:04d}.json"
                                            
                                            try:
                                                with open(batch_filename, 'w', encoding='utf-8') as f:
                                                    json.dump({'results': batch_results}, f, separators=(',', ':'))
                                            except Exception:
                                                pass
                                            
                                            batch_results = []
                                        
                                        if cleanup_counter % 30 == 0:
                                            gc.collect()
                                            cleanup_caches()
    
    pbar.close()
    
    if batch_results:
        batch_count += 1
        batch_filename = f"src/grouping_data/saved_batches/clustering_batch_{run_id}_{batch_count:04d}.json"
        try:
            with open(batch_filename, 'w', encoding='utf-8') as f:
                json.dump({'results': batch_results}, f, separators=(',', ':'))
        except Exception:
            pass
    
    summary_data = {
        'metadata': {
            'run_id': run_id,
            'timestamp': datetime.utcnow().isoformat(),
            'total_configs_tested': config_count,
            'total_batches': batch_count,
            'datasets': [d[0] for d in test_data]
        },
        'best_config': best_config
    }
    
    summary_filename = f"clustering_summary_{run_id}.json"
    try:
        with open(summary_filename, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, separators=(',', ':'))
    except Exception:
        pass
    
    cleanup_caches()
    gc.collect()
    
    return summary_data

def calculate_multiple_metrics(true_labels: np.ndarray, pred_labels: np.ndarray, embeddings: np.ndarray = None) -> Dict[str, float]:
    metrics = {}
    
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            metrics['ari'] = adjusted_rand_score(true_labels, pred_labels)
            metrics['nmi'] = normalized_mutual_info_score(true_labels, pred_labels)
            metrics['homogeneity'] = homogeneity_score(true_labels, pred_labels)
            metrics['completeness'] = completeness_score(true_labels, pred_labels)
            metrics['v_measure'] = v_measure_score(true_labels, pred_labels)
    except Exception:
        metrics['ari'] = metrics['nmi'] = metrics['homogeneity'] = metrics['completeness'] = metrics['v_measure'] = 0.0
    
    if embeddings is not None:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                unique_labels = np.unique(pred_labels)
                if len(unique_labels) > 1 and len(unique_labels) < len(pred_labels):
                    metrics['silhouette'] = silhouette_score(embeddings, pred_labels)
                else:
                    metrics['silhouette'] = 0.0
        except Exception:
            metrics['silhouette'] = 0.0
    else:
        metrics['silhouette'] = 0.0
    
    noise_ratio = np.sum(pred_labels == -1) / len(pred_labels) if len(pred_labels) > 0 else 1.0
    metrics['noise_ratio'] = noise_ratio
    
    unique_clusters = len(np.unique(pred_labels[pred_labels != -1]))
    total_points = len(pred_labels)
    metrics['cluster_ratio'] = unique_clusters / total_points if total_points > 0 else 0.0
    
    metrics['composite'] = (
        metrics['ari'] * 0.25 + 
        metrics['nmi'] * 0.25 + 
        metrics['v_measure'] * 0.20 + 
        metrics['silhouette'] * 0.15 + 
        (1 - metrics['noise_ratio']) * 0.10 + 
        (1 - min(metrics['cluster_ratio'], 0.8)) * 0.05
    )
    
    return metrics

def detect_text_type(sentences: List[str]) -> str:
    combined_text = ' '.join(sentences[:5]).lower()
    
    news_indicators = ['said', 'reported', 'according', 'officials', 'government', 'police', 'court', 'president']
    technical_indicators = ['algorithm', 'function', 'method', 'system', 'process', 'data', 'analysis']
    
    news_score = sum(1 for indicator in news_indicators if indicator in combined_text)
    tech_score = sum(1 for indicator in technical_indicators if indicator in combined_text)
    
    if news_score > tech_score:
        return 'news'
    elif tech_score > 0:
        return 'technical'
    return 'general'

def adaptive_context_weighting(embeddings: np.ndarray, i: int, context_len: int, weights: Dict[str, float]) -> np.ndarray:
    n_embeddings = len(embeddings)
    start_idx = max(0, i - context_len)
    end_idx = min(n_embeddings, i + context_len + 1)
    
    context_embeddings = embeddings[start_idx:end_idx]
    distances = np.linalg.norm(context_embeddings - embeddings[i], axis=1)
    
    max_dist = np.max(distances) if len(distances) > 1 else 1.0
    if max_dist > 0:
        similarity_weights = 1 - (distances / max_dist)
    else:
        similarity_weights = np.ones(len(distances))
    
    weighted_context = np.average(context_embeddings, axis=0, weights=similarity_weights)
    
    return embeddings[i] * weights["single"] + weighted_context * weights["context"]

def extract_linguistic_features(sentences: List[str]) -> Dict[str, float]:
    features = {}
    
    total_chars = sum(len(s) for s in sentences)
    total_words = sum(len(s.split()) for s in sentences)
    features['avg_sentence_length'] = total_chars / len(sentences) if sentences else 0
    features['avg_words_per_sentence'] = total_words / len(sentences) if sentences else 0
    
    punctuation_density = sum(s.count('.') + s.count('!') + s.count('?') for s in sentences) / total_chars if total_chars > 0 else 0
    features['punctuation_density'] = punctuation_density
    
    all_words = ' '.join(sentences).lower().split()
    word_freq = Counter(all_words)
    unique_words = len(word_freq)
    features['lexical_diversity'] = unique_words / total_words if total_words > 0 else 0
    
    return features

if __name__ == "__main__":
    result = locate_optimal_clustering_metrics()
    if result and result.get('best_config'):
        print(f"Best Composite Score: {result['best_config']['composite_score']:.4f}")
        print(f"ARI Score: {result['best_config']['ari_score']:.4f}")
        print(f"NMI Score: {result['best_config']['nmi_score']:.4f}")
        print(f"Silhouette Score: {result['best_config']['silhouette_score']:.4f}")
        print(f"Best Config: {result['best_config']['config']}")
    else:
        print("No valid configurations found")
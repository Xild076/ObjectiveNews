import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sentence_transformers import SentenceTransformer
from typing import Union, Type, List, Dict, Any
from nltk.stem import WordNetLemmatizer
from concurrent.futures import ThreadPoolExecutor, as_completed


lemmatizer = WordNetLemmatizer()
model = SentenceTransformer('all-mpnet-base-v2')


def find_representative_sentence(X: np.ndarray, 
                                 labels: np.ndarray, 
                                 cluster_id: int) -> int:
    cluster_indices = [i for i, label in enumerate(labels) if label == cluster_id]
    cluster_embeddings = X[cluster_indices]
    centroid = np.mean(cluster_embeddings, axis=0)
    distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
    rep_index = cluster_indices[np.argmin(distances)]
    return rep_index

def get_sentence_with_context(sentences: List[str], 
                              rep_index: int, 
                              context_len: int = 1) -> str:
    start = max(rep_index - context_len, 0)
    end = min(rep_index + context_len + 1, len(sentences))
    context_sents = [s.strip() for s in sentences[start:end]]
    return " ".join(context_sents)

def encode_text(sentences: List[str], 
               weights: Dict[str,float] = {'single':0.7,'context':0.3}, 
               context: bool = False, 
               context_len: int = 1, 
               lemmatize: bool = False) -> np.ndarray:
    processed_sentences = []
    if lemmatize:
        for sent in sentences:
            tokens = sent.split()
            lem_tokens = [lemmatizer.lemmatize(t) for t in tokens]
            processed_sentences.append(" ".join(lem_tokens))
    else:
        processed_sentences = sentences
    embeddings = []
    with ThreadPoolExecutor() as executor:
        future_to_index = {executor.submit(model.encode, s): i for i, s in enumerate(processed_sentences)}
        results = {}
        for future in as_completed(future_to_index):
            i = future_to_index[future]
            e_s = future.result()
            if context:
                start = max(i - context_len, 0)
                end = min(i + context_len + 1, len(processed_sentences))
                c_sents = " ".join(processed_sentences[start:end])
                e_c = model.encode(c_sents)
            else:
                e_c = e_s
            embeddings.append((e_s * weights['single']) + (e_c * weights['context']))
    return np.array(embeddings)

def cluster_text(
    sentences: Union[List[str], List[Dict[str, Any]], np.ndarray], 
    clustering_method: Union[Type[AgglomerativeClustering], Type[KMeans]] = AgglomerativeClustering, 
    context_weights: Dict[str, float] = {'single': 0.7, 'context': 0.3}, 
    score_weights: Dict[str, float] = {'sil': 0.4, 'db': 0.55, 'ch': 0.05},
    context: bool = False, 
    lemmatize: bool = False, 
    context_len: int = 1, 
    representative_context_len: int = 1,
    max_clusters=10
) -> Dict[str, Any]:
    if isinstance(sentences, dict):
        texts = sentences.get('text', [])
        sources = sentences.get('source', [])
    elif isinstance(sentences, list) and len(sentences) > 0 and isinstance(sentences[0], dict):
        texts = [s.get('text', '') for s in sentences]
        sources = [s.get('source', '') for s in sentences]
    else:
        texts = sentences
        sources = None
    if isinstance(sentences, list):
        X = encode_text(texts, context_weights, context, context_len, lemmatize)

    results = []
    with ThreadPoolExecutor() as executor:
        future_to_n = {}
        for n in range(2, min(len(texts), max_clusters)):
            if clustering_method == AgglomerativeClustering:
                algo = AgglomerativeClustering(n_clusters=n)
            else:
                algo = KMeans(n_clusters=n, n_init='auto')
            future = executor.submit(algo.fit_predict, X)
            future_to_n[future] = n
        for future in as_completed(future_to_n):
            n = future_to_n[future]
            lbls = future.result()
            if len(set(lbls)) == 1:
                continue
            sil = silhouette_score(X, lbls)
            db = davies_bouldin_score(X, lbls)
            ch = calinski_harabasz_score(X, lbls)
            results.append((n, lbls, sil, db, ch))
    if not results:
        return {}
    sils = [r[2] for r in results]
    dbs = [r[3] for r in results]
    chs = [r[4] for r in results]
    sil_min, sil_max = min(sils), max(sils)
    db_min, db_max = min(dbs), max(dbs)
    ch_min, ch_max = min(chs), max(chs)
    best_score = -999
    best = None
    total = sum(score_weights.values())
    normalized_score_weights = {key: value / total for key, value in score_weights.items()}
    for n, lbls, sil, db, ch in results:
        sil_norm = (sil - sil_min)/(sil_max - sil_min) if sil_max != sil_min else 1
        db_norm = 1-(db - db_min)/(db_max - db_min) if db_max != db_min else 1
        ch_norm = (ch - ch_min)/(ch_max - ch_min) if ch_max != ch_min else 1
        score = (sil_norm * normalized_score_weights.get('sil', 0) +
                 db_norm * normalized_score_weights.get('db', 0) +
                 ch_norm * normalized_score_weights.get('ch', 0))
        if score > best_score:
            best_score = score
            best = (n, lbls, sil, db, ch)
    lbls = best[1]
    sil, db, ch = best[2], best[3], best[4]
    
    cluster_dicts = []
    for c in sorted(set(lbls)):
        cluster_sentences = [texts[i] for i, label in enumerate(lbls) if label == c]
        if sources:
            cluster_sources = set(sources[i] for i, label in enumerate(lbls) if label == c)
        else:
            cluster_sources = set()
        rep_index = find_representative_sentence(X, lbls, c)
        rep_sentence = texts[rep_index]
        rep_with_context = get_sentence_with_context(texts, rep_index, representative_context_len)
        cluster_dicts.append({
            'cluster_id': int(c), 
            'sentences': cluster_sentences, 
            'sources': cluster_sources, 
            'representative': rep_sentence.strip(),
            'representative_with_context': rep_with_context.strip()
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
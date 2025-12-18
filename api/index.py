import os
import sys
from typing import List, Optional, Dict, Any, Union, Tuple

from fastapi import FastAPI

# Mangum adapts ASGI apps (FastAPI) to serverless runtimes such as Vercel/AWS Lambda.
# We import lazily so local runs without Mangum still work.
try:
    from mangum import Mangum
except ImportError:  # pragma: no cover - Mangum only needed in serverless
    Mangum = None
from pydantic import BaseModel

# Ensure we can import from src/
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from grouping.grouping import (
    cluster_sentences,
    cluster_texts,
    group_individual_articles,
    OPTIMAL_CLUSTERING_PARAMS,
)
from utility import SentenceHolder
from objectify.objectify import objectify_text, calculate_objectivity
from objectify.synonym import get_synonyms
from reliability import get_source_label, normalize_domain, _load_source_df
from article_analysis import article_analysis

app = FastAPI(title="ObjectiveNews API", version="1.0")

# Allow local Next.js dev server to call the API when developing locally.
from fastapi.middleware.cors import CORSMiddleware

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:3001",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins + ["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SentenceIn(BaseModel):
    text: str
    source: Optional[str] = None
    author: Optional[str] = None
    date: Optional[str] = None


class ClusterParams(BaseModel):
    weights: float = 0.4
    context: bool = True
    context_len: int = 1
    preprocess: bool = False
    attention: bool = False
    norm: str = "l2"
    reduce: bool = False
    n_neighbors: int = 5
    n_components: int = 2
    umap_metric: str = "cosine"
    cluster_metric: str = "euclidean"
    algorithm: str = "generic"
    cluster_selection_method: str = "eom"
    min_cluster_size: int = 3
    min_samples: int = 1


class ClusterRequest(BaseModel):
    sentences: List[str]
    # Direct parameters for cluster_sentences (optional)
    attention: bool = False
    preprocess: bool = True
    context: bool = True
    context_len: int = 5
    weights: float = 0.1
    norm: str = "l2"
    reduce: bool = False
    n_neighbors: int = 15
    n_components: int = 2
    umap_metric: str = "cosine"
    cluster_metric: str = "cosine"
    algorithm: str = "best"
    cluster_selection_method: str = "eom"
    min_cluster_size: int = 2
    min_samples: int = 1


class ClusterTextsRequest(BaseModel):
    sentences: List[SentenceIn]
    params: Optional[ClusterParams] = None


class ArticleIn(BaseModel):
    text: str
    source: Optional[str] = None
    author: Optional[str] = None
    date: Optional[str] = None


class ObjectifyRequest(BaseModel):
    text: str


class AnalysisRequest(BaseModel):
    text: str
    link_n: int = 10
    diverse_links: bool = True
    summarize_level: str = "medium"


class SynonymRequest(BaseModel):
    word: str
    topn: int = 15


class DomainReliabilityRequest(BaseModel):
    domain: str


# --- Helpers ---

def _serialize_sentence(s: SentenceHolder) -> Dict[str, Any]:
    return {
        "text": s.text,
        "source": getattr(s, "source", None),
        "author": getattr(s, "author", None),
        "date": getattr(s, "date", None),
    }


def _serialize_cluster_dict(cluster: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "label": cluster.get("label"),
        "sentences": [_serialize_sentence(s) for s in cluster.get("sentences", [])],
    }
    rep = cluster.get("representative")
    if rep is not None:
        out["representative"] = _serialize_sentence(rep)
    rep_ctx = cluster.get("representative_with_context")
    if isinstance(rep_ctx, tuple) and rep_ctx:
        r0, r1 = rep_ctx
        out["representative_with_context"] = (
            _serialize_sentence(r0) if isinstance(r0, SentenceHolder) else r0,
            _serialize_sentence(r1) if isinstance(r1, SentenceHolder) else r1,
        )
    elif rep_ctx is not None:
        out["representative_with_context"] = (
            _serialize_sentence(rep_ctx),
            _serialize_sentence(rep_ctx),
        )
    return out


# --- Routes ---

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/cluster")
async def api_cluster(req: ClusterRequest):
    labels = cluster_sentences(
        sentences=req.sentences,
        _att_model=None,
        weights=req.weights,
        context=req.context,
        context_len=req.context_len,
        preprocess=req.preprocess,
        attention=req.attention,
        norm=req.norm,
        reduce=req.reduce,
        n_neighbors=req.n_neighbors,
        n_components=req.n_components,
        umap_metric=req.umap_metric,
        cluster_metric=req.cluster_metric,
        algorithm=req.algorithm,
        cluster_selection_method=req.cluster_selection_method,
        min_cluster_size=req.min_cluster_size,
        min_samples=req.min_samples,
    )
    return {"labels": labels}


@app.post("/cluster-texts")
async def api_cluster_texts(req: ClusterTextsRequest):
    sents = [SentenceHolder(text=s.text, source=s.source, author=s.author, date=s.date) for s in req.sentences]
    params = (req.params.dict() if req.params is not None else OPTIMAL_CLUSTERING_PARAMS)
    clusters = cluster_texts(sents, params=params)
    return {"clusters": [_serialize_cluster_dict(c) for c in clusters]}


@app.post("/group-article")
async def api_group_article(article: ArticleIn):
    reps = group_individual_articles({
        "text": article.text,
        "source": article.source,
        "author": article.author,
        "date": article.date,
    })
    serialized: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for pair in reps:
        if isinstance(pair, tuple) and len(pair) == 2:
            a, b = pair
            a = _serialize_sentence(a) if isinstance(a, SentenceHolder) else a
            b = _serialize_sentence(b) if isinstance(b, SentenceHolder) else b
            serialized.append((a, b))
    return {"representatives": serialized}


@app.post("/objectify")
async def api_objectify(req: ObjectifyRequest):
    original_score = calculate_objectivity(req.text)
    objectified = objectify_text(req.text)
    new_score = calculate_objectivity(objectified)
    return {
        "original_text": req.text,
        "objectified_text": objectified,
        "original_score": original_score,
        "new_score": new_score,
        "improvement": new_score - original_score
    }


@app.post("/analyze")
async def api_analyze(req: AnalysisRequest):
    try:
        level = {"best": "slow"}.get(req.summarize_level, req.summarize_level)
        result = article_analysis(
            text=req.text,
            link_n=req.link_n,
            diverse_links=req.diverse_links,
            summarize_level=level,
            progress_callback=None
        )
        serialized = []
        for cluster in result:
            c_out = {
                "summary": cluster.get("summary", ""),
                "reliability": cluster.get("reliability", 0.0),
                "sentences": []
            }
            rep = cluster.get("representative")
            if rep and hasattr(rep, "text"):
                c_out["representative"] = _serialize_sentence(rep)
            
            sents = cluster.get("sentences", []) or []
            for s in sents:
                if isinstance(s, tuple):
                    s = s[0]
                if hasattr(s, "text"):
                    c_out["sentences"].append(_serialize_sentence(s))
            serialized.append(c_out)
        return {"clusters": serialized}
    except Exception as e:
        return {"error": str(e), "clusters": []}


@app.post("/synonyms")
async def api_synonyms(req: SynonymRequest):
    synonyms_raw = get_synonyms(req.word, deep=True, include_external=True, topn=req.topn)
    out = []
    for syn in synonyms_raw or []:
        word = syn.get("word")
        definition = syn.get("definition")
        obj = calculate_objectivity(word) if word else 0.0
        out.append({"word": word, "definition": definition, "objectivity": obj})
    return {"synonyms": out}


@app.post("/domain-reliability")
async def api_domain_reliability(req: DomainReliabilityRequest):
    normalized = normalize_domain(req.domain)
    score, _ = get_source_label(normalized, _load_source_df())
    percent = (score + 1) / 2 * 100 if score is not None else 0.0
    return {"domain": normalized, "score": score, "percent": percent}


# Adapter for Vercel/AWS Lambda. Exposed as `handler` so the runtime can invoke FastAPI.
if Mangum is not None:
    handler = Mangum(app)
else:
    handler = None

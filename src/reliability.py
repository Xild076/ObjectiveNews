import os
import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Tuple
from urllib.parse import urlparse
from utility import cache_resource_decorator
from objectify.objectify import calculate_objectivity

_SOURCE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "golden_truth_dataset.csv"))

@cache_resource_decorator
def _load_source_df():
    dir_path = os.path.dirname(_SOURCE_PATH)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    if not os.path.exists(_SOURCE_PATH):
        pd.DataFrame(columns=["domain", "reliability_label"]).to_csv(_SOURCE_PATH, index=False)
    
    df = pd.read_csv(_SOURCE_PATH)
    if "domain" in df.columns:
        df["domain"] = df["domain"].astype(str).str.strip()
    return df

def _save_source_df(df: pd.DataFrame):
    df.to_csv(_SOURCE_PATH, index=False)

def normalize_domain(src: str) -> str:
    src = src.strip()
    if src.startswith(("http://", "https://")):
        src = urlparse(src).netloc
    return src.lower().removeprefix("www.")

def _parse_date(v):
    if isinstance(v, datetime):
        return v
    if isinstance(v, str):
        for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%m/%d/%Y"):
            try:
                return datetime.strptime(v, fmt)
            except (ValueError, TypeError):
                continue
    return None

def normalize_minmax(raw: Dict[int, float], reverse: bool = False) -> Dict[int, float]:
    if not raw:
        return {}
    vals = np.array(list(raw.values()), dtype=float)
    lo, hi = vals.min(), vals.max()
    if hi == lo:
        return {k: 0.5 for k in raw}
    if reverse:
        return {k: 1 - ((v - lo) / (hi - lo)) for k, v in raw.items()}
    return {k: (v - lo) / (hi - lo) for k, v in raw.items()}

def get_source_label(domain: str, df: pd.DataFrame=None) -> Tuple[float, pd.DataFrame]:
    if df is None:
        df = _load_source_df()
    m = df["domain"] == domain
    if m.any():
        return float(df.loc[m, "reliability_label"].iloc[0]), df
    new_row = pd.DataFrame([{"domain": domain, "reliability_label": 0.5}])
    df = pd.concat([df, new_row], ignore_index=True)
    return 0.5, df

def get_source_reputation01(domain: str, df: pd.DataFrame=None) -> Tuple[float, pd.DataFrame]:
    if df is None:
        df = _load_source_df()
    m = df["domain"] == domain
    if not m.any():
        # persist a neutral placeholder for unknown domains
        new_row = pd.DataFrame([{"domain": domain, "reliability_label": 0.5}])
        df = pd.concat([df, new_row], ignore_index=True)
        return 0.5, df
    sub = df.loc[m]
    rep_candidates: List[float] = []
    if "newsguard_score" in sub.columns and sub["newsguard_score"].notna().any():
        ng = pd.to_numeric(sub["newsguard_score"], errors="coerce").dropna()
        if not ng.empty:
            rep_candidates.append(float(ng.mean()) / 100.0)
    if "reliability_label" in sub.columns:
        rl = pd.to_numeric(sub["reliability_label"], errors="coerce").dropna()
        mapped = []
        for v in rl:
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if np.isnan(fv):
                continue
            if fv < 0:
                fv = 0.0
            elif fv > 1.5:
                fv = fv / 100.0
            fv = min(1.0, max(0.0, fv))
            mapped.append(fv)
        if mapped:
            rep_candidates.append(float(np.mean(mapped)))
    rep = float(np.mean(rep_candidates)) if rep_candidates else 0.5
    return rep, df

def calculate_reliability(clusters: List[Dict]) -> List[Dict]:
    source_df = _load_source_df()
    dates = defaultdict(list)
    sources = defaultdict(list)
    obj_scores = defaultdict(list)
    src_rep_scores = defaultdict(list)

    for i, c in enumerate(clusters):
        for s in c["sentences"]:
            dom = normalize_domain(getattr(s, "source", "") or "")
            if dom:
                rep01, source_df = get_source_reputation01(dom, source_df)
                sources[i].append(dom)
                src_rep_scores[i].append(rep01)
            dt = _parse_date(getattr(s, "date", None))
            if dt:
                dates[i].append(dt)
            obj_scores[i].append(calculate_objectivity(s.text))

    unique_cluster_sources = {i: set(s) for i, s in sources.items()}
    total_unique_sources = len(set.union(*unique_cluster_sources.values()) if unique_cluster_sources else set())
    source_diversity_norm = {
        i: (len(v) - 1) / (total_unique_sources - 1) if total_unique_sources > 1 else 0.0
        for i, v in unique_cluster_sources.items()
    }

    temporal_spread_raw = {i: (max(d) - min(d)).days for i, d in dates.items() if len(d) > 1}
    temporal_cohesion_norm = {
        i: max(0, 1 - (v / 90)) for i, v in temporal_spread_raw.items()
    }

    now = datetime.now()
    recency_raw = {i: (now - max(d)).days for i, d in dates.items() if d}
    recency_norm = normalize_minmax(recency_raw, reverse=True)

    obj_norm = {i: np.mean(v) if v else 0.5 for i, v in obj_scores.items()}

    src_rep_abs = {i: (float(np.mean(v)) if v else 0.5) for i, v in src_rep_scores.items()}

    for i, c in enumerate(clusters):
        s_rep = src_rep_abs.get(i, 0.5)
        s_div = source_diversity_norm.get(i, 0.0)
        o = obj_norm.get(i, 0.5)
        rc = recency_norm.get(i, 0.5)
        t_coh = temporal_cohesion_norm.get(i, 1.0)

        score = (0.5 * s_rep) + (0.2 * s_div) + (0.25 * o) + (0.05 * rc)
        score *= t_coh

        c["reliability"] = max(0, min(score * 100, 100))
        c["reliability_details"] = {
            "source_reputation": s_rep * 100,
            "source_diversity": s_div * 100,
            "recency": rc * 100,
            "objectivity": o * 100,
            "temporal_cohesion": t_coh * 100
        }
        c["sources"] = list(unique_cluster_sources.get(i, []))
    
    _save_source_df(source_df)
    return clusters
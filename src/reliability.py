import os
import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Tuple
from urllib.parse import urlparse
from utility import DLog, cache_resource_decorator
from objectify.objectify import calculate_objectivity

_SOURCE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "golden_truth_dataset.csv"))
logger = DLog(name="RELIABILITY", level="DEBUG")

def _load_source_df():
    logger.info(f"Loading source reliability data from {_SOURCE_PATH}")
    dir_path = os.path.dirname(_SOURCE_PATH)
    if dir_path and not os.path.exists(dir_path):
        logger.info(f"Creating directory {dir_path} for source reliability data.")
        os.makedirs(dir_path, exist_ok=True)
    if not os.path.exists(_SOURCE_PATH):
        logger.info(f"Source reliability data file not found at {_SOURCE_PATH}. Creating a new empty DataFrame.")
        pd.DataFrame(columns=["domain", "reliability_label"]).to_csv(_SOURCE_PATH, index=False)
    
    df = pd.read_csv(_SOURCE_PATH)
    if "domain" in df.columns:
        df["domain"] = df["domain"].astype(str).str.strip()
    return df

def _save_source_df(df: pd.DataFrame):
    logger.info(f"Saving updated source reliability data to {_SOURCE_PATH}")
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
            except:
                continue
    return None

def normalize_minmax(raw: Dict[int, float]) -> Dict[int, float]:
    if not raw:
        return {}
    vals = np.array(list(raw.values()), dtype=float)
    lo, hi = vals.min(), vals.max()
    if hi == lo:
        return {k: 0.5 for k in raw}
    return {k: (v - lo) / (hi - lo) for k, v in raw.items()}

def get_source_label(domain: str, df: pd.DataFrame=None) -> Tuple[float, pd.DataFrame]:
    if df is None:
        df = _load_source_df()
    m = df["domain"] == domain
    if m.any():
        return float(df.loc[m, "reliability_label"].iloc[0]), df
    logger.info(f"Domain '{domain}' not found, adding to in-memory DataFrame with default score 0.0")
    new_row = pd.DataFrame([{"domain": domain, "reliability_label": 0.0}])
    df = pd.concat([df, new_row], ignore_index=True)
    return 0.0, df

def calculate_reliability(clusters: List[Dict]) -> List[Dict]:
    source_df = _load_source_df()
    dates = defaultdict(list)
    internal_scores = defaultdict(list)
    obj_scores = defaultdict(list)
    for i, c in enumerate(clusters):
        for s in c["sentences"]:
            dom = normalize_domain(getattr(s, "source", "") or "")
            if dom:
                label, source_df = get_source_label(dom, source_df)
                internal_scores[i].append(label)
            dt = _parse_date(getattr(s, "date", None))
            if dt:
                dates[i].append(dt)
            obj_scores[i].append(calculate_objectivity(s.text))
    cov_raw = {i: (max(d) - min(d)).days for i, d in dates.items() if len(d) > 1}
    rec_base = min((min(d) for d in dates.values() if d), default=None)
    rec_raw = {i: (max(d) - rec_base).days for i, d in dates.items() if d and rec_base is not None}
    cov_norm = normalize_minmax(cov_raw)
    rec_norm = normalize_minmax(rec_raw)
    obj_norm = {i: np.mean(v) if v else 0.8 for i, v in obj_scores.items()}
    src_norm = normalize_minmax({i: np.mean(v) for i, v in internal_scores.items() if v})
    for i, c in enumerate(clusters):
        s = (src_norm.get(i, 0.0) + 1) / 2
        o = obj_norm.get(i, 0.8)
        cv = cov_norm.get(i, 0.5)
        rc = rec_norm.get(i, 0.5)
        score = 0.5 * s + 0.1667 * cv + 0.1667 * rc + 0.1667 * o
        c["reliability"] = max(0, min(score * 100, 100))
        c["reliability_details"] = {
            "source_reputation": s,
            "coverage_diversity": cv,
            "recency": rc,
            "objectivity": o,
        }
    _save_source_df(source_df)
    return clusters
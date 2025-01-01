import numpy as np
from dateutil import parser

def calculate_general_reliability(scores, alpha=1.0, sigma=None):
    scores = np.array(scores)
    r_min = np.min(scores)
    n = len(scores)
    
    sigma = np.std(scores) if sigma is None else max(sigma, 1e-6)
    
    weights = np.exp(-((scores - r_min) ** 2) / (2 * sigma ** 2))
    r_general = np.sum(weights * scores) / np.sum(weights)
    
    penalty_count = np.sum(scores > 2 * r_min)
    penalty_ratio = penalty_count / n

    r_final = r_general * (1 + alpha * penalty_ratio)
    return r_final

def calculate_date_relevancy(dates_dict, coverage_weight=1.0, last_date_weight=1.0, bounded_range=False):
    parsed_data = {}
    all_min_dates = []
    all_max_dates = []

    for identifier, str_dates in dates_dict.items():
        if not str_dates:
            parsed_data[identifier] = {
                "dates": [],
                "min_date": None,
                "max_date": None
            }
            continue
        dt_list = [parser.parse(d) if isinstance(d, str) else d for d in str_dates]
        parsed_data[identifier] = {
            "dates": dt_list,
            "min_date": min(dt_list),
            "max_date": max(dt_list)
        }
        all_min_dates.append(min(dt_list))
        all_max_dates.append(max(dt_list))

    if not all_min_dates or not all_max_dates:
        return {}

    coverage_values = np.array([(info["max_date"] - info["min_date"]).days for info in parsed_data.values()])

    coverage_min = min(coverage_values)
    coverage_max = max(coverage_values)
    coverage_range = max(coverage_max - coverage_min, 1e-6)

    all_max_dates_sorted = sorted(all_max_dates)
    median_idx = len(all_max_dates_sorted) // 2
    if len(all_max_dates_sorted) % 2 == 1:
        median_dt = all_max_dates_sorted[median_idx]
    else:
        median_dt1 = all_max_dates_sorted[median_idx - 1]
        median_dt2 = all_max_dates_sorted[median_idx]
        median_dt = median_dt1 + (median_dt2 - median_dt1) / 2

    min_last_dt = min(all_max_dates)
    max_last_dt = max(all_max_dates)
    total_span_days = (max_last_dt - min_last_dt).days or 1

    if bounded_range:
        total_weight = coverage_weight + last_date_weight
        coverage_weight /= total_weight
        last_date_weight /= total_weight

    for identifier, info in parsed_data.items():
        cov_days = info["coverage_days"]
        coverage_score = (cov_days - coverage_min) / coverage_range
        coverage_score = coverage_score ** 0.5
        last_date = info["max_date"]
        dist_from_median = abs((last_date - median_dt).days)
        half_span = total_span_days / 2
        distance_score = 1.0 - (dist_from_median / half_span)
        last_date_score = max(0.0, min(1.0, distance_score))
        relevancy = 1.0 + (coverage_weight * (1.0 - coverage_score)) + (last_date_weight * (1.0 - last_date_score))
        parsed_data[identifier]["coverage_score"] = coverage_score
        parsed_data[identifier]["last_date"] = last_date
        parsed_data[identifier]["last_date_score"] = last_date_score
        parsed_data[identifier]["relevancy"] = relevancy

    results = {}
    for identifier, info in parsed_data.items():
        if not info["dates"]:
            results[identifier] = {
                "dates": [],
                "min_date": None,
                "max_date": None,
                "coverage_days": 0,
                "coverage_score": 0.0,
                "last_date": None,
                "last_date_score": 0.0,
                "relevancy": 1.0
            }
            continue
        results[identifier] = {
            "dates": [dt.isoformat() for dt in info["dates"]],
            "min_date": info["min_date"].isoformat(),
            "max_date": info["max_date"].isoformat(),
            "coverage_days": info["coverage_days"],
            "coverage_score": round(info["coverage_score"], 3),
            "last_date": info["last_date"].isoformat(),
            "last_date_score": round(info["last_date_score"], 3),
            "relevancy": round(info["relevancy"], 3),
        }

    return results

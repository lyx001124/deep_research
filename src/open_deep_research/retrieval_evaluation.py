"""Deterministic metrics for evaluating local document retrieval."""

import math
from typing import Any


def _record_id(record: dict[str, Any]) -> str:
    citation = record.get("citation")
    if isinstance(citation, str) and citation:
        return citation.casefold()
    return (
        f"{record.get('relative_path', '')}#page={record.get('page', '')}"
    ).casefold()


def evaluate_ranked_records(
    retrieved: list[dict[str, Any]],
    relevant: dict[str, float],
    k: int,
) -> dict[str, float]:
    """Calculate standard IR metrics at K from citation-level relevance labels."""
    if k < 1:
        raise ValueError("k must be at least 1")
    normalized_relevant = {
        citation.casefold(): max(0.0, float(grade))
        for citation, grade in relevant.items()
        if float(grade) > 0
    }
    ranked_ids = []
    for record in retrieved:
        record_id = _record_id(record)
        if record_id not in ranked_ids:
            ranked_ids.append(record_id)
        if len(ranked_ids) >= k:
            break
    hits = [citation for citation in ranked_ids if citation in normalized_relevant]
    unique_hits = set(hits)

    hit_at_k = 1.0 if unique_hits else 0.0
    recall_at_k = len(unique_hits) / max(1, len(normalized_relevant))
    precision_at_k = len(hits) / k
    citation_accuracy = len(hits) / max(1, len(ranked_ids))

    reciprocal_rank = 0.0
    for rank, citation in enumerate(ranked_ids, start=1):
        if citation in normalized_relevant:
            reciprocal_rank = 1.0 / rank
            break

    discounted_gain = sum(
        (2 ** normalized_relevant.get(citation, 0.0) - 1) / math.log2(rank + 1)
        for rank, citation in enumerate(ranked_ids, start=1)
    )
    ideal_grades = sorted(normalized_relevant.values(), reverse=True)[:k]
    ideal_gain = sum(
        (2**grade - 1) / math.log2(rank + 1)
        for rank, grade in enumerate(ideal_grades, start=1)
    )
    ndcg_at_k = discounted_gain / ideal_gain if ideal_gain else 0.0

    return {
        "hit_at_k": round(hit_at_k, 6),
        "recall_at_k": round(recall_at_k, 6),
        "precision_at_k": round(precision_at_k, 6),
        "citation_accuracy": round(citation_accuracy, 6),
        "reciprocal_rank": round(reciprocal_rank, 6),
        "ndcg_at_k": round(ndcg_at_k, 6),
    }


def percentile(values: list[float], percentile_value: float) -> float:
    """Return a linearly interpolated percentile without external dependencies."""
    if not values:
        return 0.0
    if not 0 <= percentile_value <= 100:
        raise ValueError("percentile must be between 0 and 100")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * percentile_value / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def aggregate_evaluation_results(results: list[dict[str, Any]]) -> dict[str, float]:
    """Average case metrics and summarize latency and cache behavior."""
    if not results:
        return {
            "case_count": 0,
            "hit_at_k": 0.0,
            "recall_at_k": 0.0,
            "precision_at_k": 0.0,
            "citation_accuracy": 0.0,
            "mrr": 0.0,
            "ndcg_at_k": 0.0,
            "avg_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "cache_hit_rate": 0.0,
        }

    def average(field: str) -> float:
        return sum(float(result.get(field, 0.0)) for result in results) / len(results)

    latencies = [float(result.get("latency_ms", 0.0)) for result in results]
    return {
        "case_count": len(results),
        "hit_at_k": round(average("hit_at_k"), 6),
        "recall_at_k": round(average("recall_at_k"), 6),
        "precision_at_k": round(average("precision_at_k"), 6),
        "citation_accuracy": round(average("citation_accuracy"), 6),
        "mrr": round(average("reciprocal_rank"), 6),
        "ndcg_at_k": round(average("ndcg_at_k"), 6),
        "avg_latency_ms": round(sum(latencies) / len(latencies), 2),
        "p95_latency_ms": round(percentile(latencies, 95), 2),
        "cache_hit_rate": round(average("cache_hit"), 6),
    }

"""Tests for deterministic local retrieval metrics."""

import pytest

from open_deep_research.retrieval_evaluation import (
    aggregate_evaluation_results,
    evaluate_ranked_records,
    percentile,
)


def test_ranked_metrics_match_known_example():
    retrieved = [
        {"citation": "local-pdf://a.pdf#page=2"},
        {"citation": "local-pdf://noise.pdf#page=1"},
        {"citation": "local-pdf://a.pdf#page=3"},
    ]
    relevant = {
        "local-pdf://a.pdf#page=2": 2,
        "local-pdf://a.pdf#page=3": 1,
    }
    metrics = evaluate_ranked_records(retrieved, relevant, k=3)
    assert metrics["hit_at_k"] == 1.0
    assert metrics["recall_at_k"] == 1.0
    assert metrics["precision_at_k"] == pytest.approx(2 / 3, abs=1e-6)
    assert metrics["citation_accuracy"] == pytest.approx(2 / 3, abs=1e-6)
    assert metrics["reciprocal_rank"] == 1.0
    assert 0.9 < metrics["ndcg_at_k"] < 1.0


def test_metrics_handle_no_relevant_retrieval():
    metrics = evaluate_ranked_records(
        [{"citation": "local-pdf://noise.pdf#page=1"}],
        {"local-pdf://a.pdf#page=2": 1},
        k=2,
    )
    assert metrics == {
        "hit_at_k": 0.0,
        "recall_at_k": 0.0,
        "precision_at_k": 0.0,
        "citation_accuracy": 0.0,
        "reciprocal_rank": 0.0,
        "ndcg_at_k": 0.0,
    }


def test_aggregate_metrics_and_latency_percentile():
    results = [
        {
            "hit_at_k": 1,
            "recall_at_k": 1,
            "precision_at_k": 0.5,
            "citation_accuracy": 0.5,
            "reciprocal_rank": 1,
            "ndcg_at_k": 1,
            "latency_ms": 10,
            "cache_hit": 0,
        },
        {
            "hit_at_k": 0,
            "recall_at_k": 0,
            "precision_at_k": 0,
            "citation_accuracy": 0,
            "reciprocal_rank": 0,
            "ndcg_at_k": 0,
            "latency_ms": 30,
            "cache_hit": 1,
        },
    ]
    summary = aggregate_evaluation_results(results)
    assert summary["case_count"] == 2
    assert summary["mrr"] == 0.5
    assert summary["avg_latency_ms"] == 20
    assert summary["p95_latency_ms"] == 29
    assert summary["cache_hit_rate"] == 0.5
    assert percentile([10, 30], 50) == 20


def test_invalid_k_is_rejected():
    with pytest.raises(ValueError):
        evaluate_ranked_records([], {}, k=0)


def test_citation_accuracy_uses_actual_result_count_and_deduplicates():
    metrics = evaluate_ranked_records(
        [
            {"citation": "local-pdf://a.pdf#page=2"},
            {"citation": "local-pdf://a.pdf#page=2"},
        ],
        {"local-pdf://a.pdf#page=2": 1},
        k=5,
    )
    assert metrics["precision_at_k"] == 0.2
    assert metrics["citation_accuracy"] == 1.0

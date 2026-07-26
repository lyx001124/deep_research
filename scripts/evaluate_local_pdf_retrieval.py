"""Evaluate local PDF retrieval against a citation-level JSON case set."""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from open_deep_research.local_pdf_tools import (  # noqa: E402
    clear_pdf_index_cache,
    create_embedding_client,
    load_cached_pdf_chunks,
    load_cached_embeddings,
    rank_pdf_chunks,
    rank_pdf_chunks_hybrid,
)
from open_deep_research.configuration import Configuration  # noqa: E402
from open_deep_research.retrieval_evaluation import (  # noqa: E402
    aggregate_evaluation_results,
    evaluate_ranked_records,
)


def _load_cases(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        cases = json.load(file)
    if not isinstance(cases, list) or not cases:
        raise ValueError("evaluation file must contain a non-empty JSON list")
    return cases


def _citation(relative_path: str, page: int) -> str:
    return f"local-pdf://{quote(relative_path, safe='/')}#page={page}"


def _relevance_labels(case: dict[str, Any]) -> dict[str, float]:
    labels = {}
    for item in case.get("relevant", []):
        labels[_citation(str(item["relative_path"]), int(item["page"]))] = float(
            item.get("grade", 1)
        )
    if not labels:
        raise ValueError(f"case {case.get('id', '<unknown>')} has no relevance labels")
    return labels


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    cases = _load_cases(args.cases)
    clear_pdf_index_cache()
    results = []
    embeddings = None
    embedding_namespace = ""
    if args.retrieval_mode == "hybrid":
        embeddings = create_embedding_client(
            Configuration(
                local_pdf_embedding_model=args.embedding_model,
                local_pdf_embedding_base_url=args.embedding_base_url,
            )
        )
        embedding_namespace = f"{args.embedding_model}|{args.embedding_base_url or ''}"
    for case in cases:
        started = time.perf_counter()
        chunks, cache_hit = load_cached_pdf_chunks(
            str(args.library),
            chunk_size=args.chunk_size,
            chunk_overlap=min(args.chunk_overlap, args.chunk_size - 1),
            max_files=args.max_files,
            cache_enabled=not args.no_cache,
        )
        if args.retrieval_mode == "hybrid":
            document_vectors, _ = load_cached_embeddings(
                chunks,
                embeddings,
                namespace=embedding_namespace,
                cache_enabled=not args.no_cache,
            )
            records = rank_pdf_chunks_hybrid(
                str(case["query"]),
                chunks,
                args.k,
                embeddings,
                candidate_limit=args.hybrid_candidate_limit,
                lexical_weight=args.lexical_weight,
                semantic_weight=1.0 - args.lexical_weight,
                document_vectors=document_vectors,
            )
        else:
            records = rank_pdf_chunks(str(case["query"]), chunks, args.k)
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        result = {
            "id": str(case.get("id", "<unknown>")),
            "query": str(case["query"]),
            "latency_ms": latency_ms,
            "cache_hit": int(cache_hit),
            "retrieved": [record["citation"] for record in records],
            **evaluate_ranked_records(records, _relevance_labels(case), args.k),
        }
        results.append(result)

    output = {
        "k": args.k,
        "retrieval_mode": args.retrieval_mode,
        "summary": aggregate_evaluation_results(results),
        "cases": results,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True, help="PDF library directory")
    parser.add_argument("--cases", type=Path, required=True, help="JSON evaluation cases")
    parser.add_argument("--output", type=Path, help="optional JSON result path")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--chunk-size", type=int, default=1200)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    parser.add_argument("--max-files", type=int, default=100)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument(
        "--retrieval-mode", choices=("bm25", "hybrid"), default="bm25"
    )
    parser.add_argument(
        "--embedding-model", default="openai:text-embedding-3-small"
    )
    parser.add_argument("--embedding-base-url")
    parser.add_argument("--hybrid-candidate-limit", type=int, default=30)
    parser.add_argument("--lexical-weight", type=float, default=0.5)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if (
        args.k < 1
        or args.chunk_size < 2
        or args.hybrid_candidate_limit < 1
        or not 0 <= args.lexical_weight <= 1
    ):
        raise ValueError("invalid retrieval evaluation arguments")
    summary = evaluate(args)["summary"]
    print(f"Retrieval Mode: {args.retrieval_mode}")
    print(f"cases: {summary['case_count']}")
    print(f"Hit@{args.k}: {summary['hit_at_k']:.3f}")
    print(f"Recall@{args.k}: {summary['recall_at_k']:.3f}")
    print(f"Precision@{args.k}: {summary['precision_at_k']:.3f}")
    print(f"Citation Accuracy: {summary['citation_accuracy']:.3f}")
    print(f"MRR: {summary['mrr']:.3f}")
    print(f"nDCG@{args.k}: {summary['ndcg_at_k']:.3f}")
    print(f"Avg Latency: {summary['avg_latency_ms']:.2f} ms")
    print(f"P95 Latency: {summary['p95_latency_ms']:.2f} ms")
    print(f"Cache Hit Rate: {summary['cache_hit_rate']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

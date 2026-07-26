"""Local PDF ingestion and hybrid retrieval tools."""

import asyncio
import hashlib
import json
import math
import os
import re
import threading
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Optional, Protocol
from urllib.parse import quote

import fitz
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg, tool

from open_deep_research.configuration import Configuration


LOCAL_DOCUMENTS_START = "<local_documents>"
LOCAL_DOCUMENTS_END = "</local_documents>"
MAX_PDF_BYTES = 50 * 1024 * 1024
MAX_PAGES_PER_PDF = 500
WORD_PATTERN = re.compile(r"[a-zA-Z][a-zA-Z0-9_+-]*|[\u4e00-\u9fff]+")
LOCAL_PDF_CITATION_PATTERN = re.compile(r"local-pdf://[^\s)\]>]+#page=\d+", re.IGNORECASE)
_PDF_INDEX_CACHE: dict[str, tuple["LocalPDFChunk", ...]] = {}
_PDF_INDEX_CACHE_ORDER: list[str] = []
_PDF_INDEX_CACHE_LOCK = threading.Lock()
_PDF_INDEX_CACHE_HITS = 0
_PDF_INDEX_CACHE_MISSES = 0
_EMBEDDING_CACHE: dict[str, tuple[tuple[float, ...], ...]] = {}
_EMBEDDING_CACHE_ORDER: list[str] = []
_EMBEDDING_CACHE_LOCK = threading.Lock()


@dataclass(frozen=True)
class LocalPDFChunk:
    """One searchable page-aware chunk extracted from a local PDF."""

    relative_path: str
    page: int
    chunk_index: int
    text: str


class EmbeddingClient(Protocol):
    """Minimal synchronous embedding interface used by hybrid retrieval."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed document texts."""

    def embed_query(self, text: str) -> list[float]:
        """Embed one search query."""


def _discover_pdf_paths(library: Path, max_files: int) -> list[Path]:
    """Return deterministic, in-library PDF paths eligible for indexing."""
    paths = []
    for path in sorted(
        (candidate for candidate in library.rglob("*.pdf") if candidate.is_file()),
        key=lambda candidate: candidate.as_posix().casefold(),
    ):
        try:
            if not path.resolve().is_relative_to(library):
                continue
            if path.stat().st_size > MAX_PDF_BYTES:
                continue
        except OSError:
            continue
        paths.append(path)
        if len(paths) >= max_files:
            break
    return paths


def _pdf_index_cache_key(
    library: Path,
    pdf_paths: list[Path],
    chunk_size: int,
    chunk_overlap: int,
    max_files: int,
) -> str:
    """Fingerprint index inputs so file changes invalidate cached chunks."""
    manifest = []
    for path in pdf_paths:
        stat = path.stat()
        manifest.append(
            (path.relative_to(library).as_posix(), stat.st_size, stat.st_mtime_ns)
        )
    payload = json.dumps(
        {
            "library": str(library),
            "files": manifest,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "max_files": max_files,
        },
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def clear_pdf_index_cache() -> None:
    """Clear cached PDF indexes and metrics, primarily for tests and maintenance."""
    global _PDF_INDEX_CACHE_HITS, _PDF_INDEX_CACHE_MISSES
    with _PDF_INDEX_CACHE_LOCK:
        _PDF_INDEX_CACHE.clear()
        _PDF_INDEX_CACHE_ORDER.clear()
        _PDF_INDEX_CACHE_HITS = 0
        _PDF_INDEX_CACHE_MISSES = 0
    with _EMBEDDING_CACHE_LOCK:
        _EMBEDDING_CACHE.clear()
        _EMBEDDING_CACHE_ORDER.clear()


def get_pdf_index_cache_stats() -> dict[str, int]:
    """Return process-local cache counters without exposing document content."""
    with _PDF_INDEX_CACHE_LOCK:
        total = _PDF_INDEX_CACHE_HITS + _PDF_INDEX_CACHE_MISSES
        return {
            "hits": _PDF_INDEX_CACHE_HITS,
            "misses": _PDF_INDEX_CACHE_MISSES,
            "entries": len(_PDF_INDEX_CACHE),
            "hit_rate_percent": round(100 * _PDF_INDEX_CACHE_HITS / total) if total else 0,
        }


def _tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for match in WORD_PATTERN.findall(text.casefold()):
        if re.fullmatch(r"[\u4e00-\u9fff]+", match):
            tokens.extend(match)
            tokens.extend(match[index : index + 2] for index in range(len(match) - 1))
        else:
            tokens.append(match)
    return tokens


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    cleaned = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    if not cleaned:
        return []
    chunks = []
    start = 0
    step = max(1, chunk_size - overlap)
    while start < len(cleaned):
        end = min(len(cleaned), start + chunk_size)
        if end < len(cleaned):
            boundary = max(cleaned.rfind("\n", start, end), cleaned.rfind("。", start, end))
            if boundary > start + chunk_size // 2:
                end = boundary + 1
        chunks.append(cleaned[start:end])
        if end == len(cleaned):
            break
        start = min(end, start + step)
    return chunks


def load_pdf_chunks(
    library_path: str,
    *,
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
    max_files: int = 20,
) -> list[LocalPDFChunk]:
    """Load PDF chunks from one explicitly configured library directory."""
    library = Path(library_path).expanduser().resolve()
    if not library.is_dir():
        return []

    chunks: list[LocalPDFChunk] = []
    pdf_paths = _discover_pdf_paths(library, max_files)
    for pdf_path in pdf_paths:
        try:
            relative_path = pdf_path.relative_to(library).as_posix()
            with fitz.open(pdf_path) as document:
                if document.needs_pass:
                    continue
                for page_index in range(min(document.page_count, MAX_PAGES_PER_PDF)):
                    page_text = document.load_page(page_index).get_text("text")
                    for chunk_index, chunk_text in enumerate(
                        _split_text(page_text, chunk_size, chunk_overlap)
                    ):
                        chunks.append(
                            LocalPDFChunk(
                                relative_path=relative_path,
                                page=page_index + 1,
                                chunk_index=chunk_index,
                                text=chunk_text,
                            )
                        )
        except (OSError, RuntimeError, ValueError):
            continue
    return chunks


def load_cached_pdf_chunks(
    library_path: str,
    *,
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
    max_files: int = 20,
    cache_enabled: bool = True,
    max_cache_entries: int = 8,
) -> tuple[list[LocalPDFChunk], bool]:
    """Load an automatically invalidated, bounded in-memory PDF index."""
    global _PDF_INDEX_CACHE_HITS, _PDF_INDEX_CACHE_MISSES
    library = Path(library_path).expanduser().resolve()
    if not library.is_dir() or not cache_enabled:
        return (
            load_pdf_chunks(
                library_path,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                max_files=max_files,
            ),
            False,
        )

    pdf_paths = _discover_pdf_paths(library, max_files)
    try:
        cache_key = _pdf_index_cache_key(
            library,
            pdf_paths,
            chunk_size,
            chunk_overlap,
            max_files,
        )
    except OSError:
        return [], False

    with _PDF_INDEX_CACHE_LOCK:
        cached = _PDF_INDEX_CACHE.get(cache_key)
        if cached is not None:
            _PDF_INDEX_CACHE_HITS += 1
            _PDF_INDEX_CACHE_ORDER.remove(cache_key)
            _PDF_INDEX_CACHE_ORDER.append(cache_key)
            return list(cached), True
        _PDF_INDEX_CACHE_MISSES += 1

    chunks = load_pdf_chunks(
        library_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        max_files=max_files,
    )
    with _PDF_INDEX_CACHE_LOCK:
        cached = _PDF_INDEX_CACHE.get(cache_key)
        if cached is not None:
            _PDF_INDEX_CACHE_HITS += 1
            if cache_key in _PDF_INDEX_CACHE_ORDER:
                _PDF_INDEX_CACHE_ORDER.remove(cache_key)
            _PDF_INDEX_CACHE_ORDER.append(cache_key)
            return list(cached), True
        _PDF_INDEX_CACHE[cache_key] = tuple(chunks)
        _PDF_INDEX_CACHE_ORDER.append(cache_key)
        while len(_PDF_INDEX_CACHE_ORDER) > max(1, max_cache_entries):
            oldest = _PDF_INDEX_CACHE_ORDER.pop(0)
            _PDF_INDEX_CACHE.pop(oldest, None)
    return chunks, False


def rank_pdf_chunks(
    query: str,
    chunks: list[LocalPDFChunk],
    limit: int,
) -> list[dict]:
    """Rank chunks with BM25 while retaining page-level source metadata."""
    query_terms = Counter(_tokenize(query))
    if not query_terms or not chunks:
        return []

    tokenized = [_tokenize(chunk.text) for chunk in chunks]
    average_length = sum(map(len, tokenized)) / max(1, len(tokenized))
    document_frequency = Counter()
    for terms in tokenized:
        document_frequency.update(set(terms) & query_terms.keys())

    ranked = []
    document_count = len(chunks)
    for chunk, terms in zip(chunks, tokenized):
        frequencies = Counter(terms)
        score = 0.0
        for term, query_frequency in query_terms.items():
            frequency = frequencies[term]
            if not frequency:
                continue
            document_hits = document_frequency[term]
            inverse_frequency = math.log(
                1 + (document_count - document_hits + 0.5) / (document_hits + 0.5)
            )
            length_normalization = 1.2 * (
                1 - 0.75 + 0.75 * len(terms) / max(1.0, average_length)
            )
            score += (
                query_frequency
                * inverse_frequency
                * frequency
                * 2.2
                / (frequency + length_normalization)
            )
        if score <= 0:
            continue
        encoded_path = quote(chunk.relative_path, safe="/")
        ranked.append(
            {
                "file_name": Path(chunk.relative_path).name,
                "relative_path": chunk.relative_path,
                "page": chunk.page,
                "chunk_index": chunk.chunk_index,
                "score": round(score, 4),
                "text": chunk.text,
                "citation": f"local-pdf://{encoded_path}#page={chunk.page}",
            }
        )
    ranked.sort(
        key=lambda item: (item["score"], -item["page"], item["relative_path"]),
        reverse=True,
    )
    unique_pages = []
    seen_citations = set()
    for item in ranked:
        if item["citation"] in seen_citations:
            continue
        seen_citations.add(item["citation"])
        unique_pages.append(item)
        if len(unique_pages) >= limit:
            break
    return unique_pages


def _record_for_chunk(chunk: LocalPDFChunk, score: float) -> dict:
    """Build one serializable retrieval record."""
    encoded_path = quote(chunk.relative_path, safe="/")
    return {
        "file_name": Path(chunk.relative_path).name,
        "relative_path": chunk.relative_path,
        "page": chunk.page,
        "chunk_index": chunk.chunk_index,
        "score": round(score, 4),
        "text": chunk.text,
        "citation": f"local-pdf://{encoded_path}#page={chunk.page}",
    }


def _embedding_cache_key(chunks: list[LocalPDFChunk], namespace: str) -> str:
    """Fingerprint chunk content and embedding provider without retaining text."""
    digest = hashlib.sha256(namespace.encode("utf-8"))
    for chunk in chunks:
        digest.update(chunk.relative_path.encode("utf-8"))
        digest.update(str(chunk.page).encode("ascii"))
        digest.update(str(chunk.chunk_index).encode("ascii"))
        digest.update(hashlib.sha256(chunk.text.encode("utf-8")).digest())
    return digest.hexdigest()


def load_cached_embeddings(
    chunks: list[LocalPDFChunk],
    embeddings: EmbeddingClient,
    *,
    namespace: str,
    cache_enabled: bool = True,
    max_cache_entries: int = 8,
) -> tuple[list[list[float]], bool]:
    """Load document vectors from a bounded content-addressed process cache."""
    if not chunks:
        return [], False
    cache_key = _embedding_cache_key(chunks, namespace)
    if cache_enabled:
        with _EMBEDDING_CACHE_LOCK:
            cached = _EMBEDDING_CACHE.get(cache_key)
            if cached is not None:
                if cache_key in _EMBEDDING_CACHE_ORDER:
                    _EMBEDDING_CACHE_ORDER.remove(cache_key)
                _EMBEDDING_CACHE_ORDER.append(cache_key)
                return [list(vector) for vector in cached], True
    vectors = embeddings.embed_documents([chunk.text for chunk in chunks])
    if len(vectors) != len(chunks):
        raise ValueError("embedding provider returned an unexpected vector count")
    if cache_enabled:
        immutable_vectors = tuple(tuple(float(value) for value in vector) for vector in vectors)
        with _EMBEDDING_CACHE_LOCK:
            existing = _EMBEDDING_CACHE.get(cache_key)
            if existing is not None:
                return [list(vector) for vector in existing], True
            _EMBEDDING_CACHE[cache_key] = immutable_vectors
            _EMBEDDING_CACHE_ORDER.append(cache_key)
            while len(_EMBEDDING_CACHE_ORDER) > max(1, max_cache_entries):
                oldest = _EMBEDDING_CACHE_ORDER.pop(0)
                _EMBEDDING_CACHE.pop(oldest, None)
    return vectors, False


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    """Return cosine similarity while tolerating zero and malformed vectors."""
    if not left or len(left) != len(right):
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def rank_pdf_chunks_semantic(
    query: str,
    chunks: list[LocalPDFChunk],
    limit: int,
    embeddings: EmbeddingClient,
    document_vectors: Optional[list[list[float]]] = None,
) -> list[dict]:
    """Rank PDF chunks by embedding cosine similarity with page deduplication."""
    if not query.strip() or not chunks or limit < 1:
        return []
    document_vectors = document_vectors or embeddings.embed_documents(
        [chunk.text for chunk in chunks]
    )
    if len(document_vectors) != len(chunks):
        raise ValueError("embedding provider returned an unexpected vector count")
    query_vector = embeddings.embed_query(query)
    ranked = []
    for chunk, vector in zip(chunks, document_vectors):
        score = _cosine_similarity(query_vector, vector)
        if score > 0:
            ranked.append(_record_for_chunk(chunk, score))
    ranked.sort(
        key=lambda item: (item["score"], -item["page"], item["relative_path"]),
        reverse=True,
    )
    unique_pages = []
    seen_citations = set()
    for item in ranked:
        if item["citation"] in seen_citations:
            continue
        seen_citations.add(item["citation"])
        unique_pages.append(item)
        if len(unique_pages) >= limit:
            break
    return unique_pages


def reciprocal_rank_fusion(
    rankings: list[tuple[list[dict], float]],
    limit: int,
    rank_constant: int = 60,
) -> list[dict]:
    """Fuse independently scaled rankings using weighted reciprocal ranks."""
    fused: dict[str, dict] = {}
    for records, weight in rankings:
        if weight <= 0:
            continue
        for rank, record in enumerate(records, start=1):
            citation = record["citation"]
            entry = fused.setdefault(
                citation,
                {"record": record, "score": 0.0, "best_rank": rank},
            )
            entry["score"] += weight / (rank_constant + rank)
            entry["best_rank"] = min(entry["best_rank"], rank)
    ordered = sorted(
        fused.values(),
        key=lambda item: (
            item["score"],
            -item["best_rank"],
            item["record"]["citation"],
        ),
        reverse=True,
    )
    results = []
    for item in ordered[:limit]:
        record = dict(item["record"])
        record["score"] = round(item["score"], 6)
        record["retrieval_mode"] = "hybrid_rrf"
        results.append(record)
    return results


def rank_pdf_chunks_hybrid(
    query: str,
    chunks: list[LocalPDFChunk],
    limit: int,
    embeddings: EmbeddingClient,
    *,
    candidate_limit: int = 30,
    lexical_weight: float = 0.5,
    semantic_weight: float = 0.5,
    document_vectors: Optional[list[list[float]]] = None,
) -> list[dict]:
    """Combine BM25 and embedding rankings with weighted RRF."""
    candidates = max(limit, candidate_limit)
    lexical = rank_pdf_chunks(query, chunks, candidates)
    semantic = rank_pdf_chunks_semantic(
        query,
        chunks,
        candidates,
        embeddings,
        document_vectors=document_vectors,
    )
    return reciprocal_rank_fusion(
        [(lexical, lexical_weight), (semantic, semantic_weight)],
        limit,
    )


def create_embedding_client(configurable: Configuration) -> EmbeddingClient:
    """Initialize the configured provider without storing its API key in graph state."""
    from langchain.embeddings import init_embeddings

    kwargs = {}
    api_key = os.getenv("LOCAL_PDF_EMBEDDING_API_KEY")
    if api_key:
        kwargs["api_key"] = api_key
    if configurable.local_pdf_embedding_base_url:
        kwargs["base_url"] = configurable.local_pdf_embedding_base_url
    return init_embeddings(configurable.local_pdf_embedding_model, **kwargs)


def serialize_local_documents(records: list[dict]) -> str:
    """Serialize retrieved chunks for preservation by downstream graph nodes."""
    return (
        f"{LOCAL_DOCUMENTS_START}\n"
        f"{json.dumps(records, ensure_ascii=False)}\n"
        f"{LOCAL_DOCUMENTS_END}"
    )


def extract_local_pdf_citations(texts: list[str]) -> set[str]:
    """Extract citations only from marked retrieval records, ignoring free text."""
    citations = set()
    for text in texts:
        pattern = re.compile(
            re.escape(LOCAL_DOCUMENTS_START)
            + r"\s*(.*?)\s*"
            + re.escape(LOCAL_DOCUMENTS_END),
            re.DOTALL,
        )
        for payload in pattern.findall(text):
            try:
                records = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if not isinstance(records, list):
                continue
            for record in records:
                if not isinstance(record, dict):
                    continue
                citation = record.get("citation")
                if (
                    isinstance(citation, str)
                    and LOCAL_PDF_CITATION_PATTERN.fullmatch(citation)
                ):
                    citations.add(citation)
    return citations


def sanitize_local_pdf_citations(
    report: str,
    allowed_citations: set[str],
) -> tuple[str, list[str]]:
    """Replace local PDF citations that were not returned by the retrieval tool."""
    allowed = {citation.casefold() for citation in allowed_citations}
    rejected = []

    def replace(match: re.Match[str]) -> str:
        citation = match.group(0).rstrip(".,;")
        if citation.casefold() in allowed:
            return match.group(0)
        rejected.append(citation)
        return "[未通过本地文档引用验证]"

    return LOCAL_PDF_CITATION_PATTERN.sub(replace, report), rejected


@tool(
    description=(
        "在管理员配置的本地 PDF 文献库中检索与问题相关的原文片段，返回文件名、页码、"
        "相关性分数、原文和 local-pdf 引用标识。适合结合用户提供的论文、标准或技术报告开展研究。"
        "只能依据返回的原文作答，并在引用时保留文件名和页码。"
    )
)
async def search_local_pdfs(
    query: str,
    max_results: Annotated[Optional[int], InjectedToolArg] = None,
    config: RunnableConfig = None,
) -> str:
    """Search the configured local PDF library without accepting arbitrary paths."""
    configurable = Configuration.from_runnable_config(config)
    if not configurable.local_pdf_search_enabled:
        return (
            f"{serialize_local_documents([])}\n"
            '<local_document_warning reason="search_disabled">'
            "本地 PDF 检索未启用。"
            "</local_document_warning>"
        )
    if not configurable.pdf_library_path:
        return (
            f"{serialize_local_documents([])}\n"
            '<local_document_warning reason="library_not_configured">'
            "尚未配置本地 PDF 文献库目录。"
            "</local_document_warning>"
        )
    index_started = time.perf_counter()
    chunks, cache_hit = await asyncio.to_thread(
        load_cached_pdf_chunks,
        configurable.pdf_library_path,
        chunk_size=configurable.local_pdf_chunk_size,
        chunk_overlap=min(
            configurable.local_pdf_chunk_overlap,
            configurable.local_pdf_chunk_size - 1,
        ),
        max_files=configurable.max_local_pdf_files,
        cache_enabled=configurable.local_pdf_cache_enabled,
        max_cache_entries=configurable.local_pdf_cache_max_entries,
    )
    index_time_ms = round((time.perf_counter() - index_started) * 1000, 2)
    limit = min(
        max_results or configurable.max_local_pdf_results,
        configurable.max_local_pdf_results,
    )
    search_started = time.perf_counter()
    requested_mode = configurable.local_pdf_retrieval_mode
    effective_mode = "bm25"
    fallback_reason = ""
    embedding_cache_hit = False
    if requested_mode == "hybrid":
        try:
            embeddings = create_embedding_client(configurable)
            embedding_namespace = (
                f"{configurable.local_pdf_embedding_model}|"
                f"{configurable.local_pdf_embedding_base_url or ''}"
            )
            document_vectors, embedding_cache_hit = await asyncio.to_thread(
                load_cached_embeddings,
                chunks,
                embeddings,
                namespace=embedding_namespace,
                cache_enabled=configurable.local_pdf_cache_enabled,
                max_cache_entries=configurable.local_pdf_cache_max_entries,
            )
            records = await asyncio.to_thread(
                rank_pdf_chunks_hybrid,
                query,
                chunks,
                limit,
                embeddings,
                candidate_limit=configurable.local_pdf_hybrid_candidate_limit,
                lexical_weight=configurable.local_pdf_lexical_weight,
                semantic_weight=1.0 - configurable.local_pdf_lexical_weight,
                document_vectors=document_vectors,
            )
            effective_mode = "hybrid"
        except Exception as error:
            records = rank_pdf_chunks(query, chunks, limit)
            fallback_reason = type(error).__name__
    else:
        records = rank_pdf_chunks(query, chunks, limit)
    search_time_ms = round((time.perf_counter() - search_started) * 1000, 2)
    warning = ""
    if not chunks:
        warning = (
            '\n<local_document_warning reason="no_readable_pdfs">'
            "配置目录中没有可读取的 PDF。"
            "</local_document_warning>"
        )
    elif not records:
        warning = (
            '\n<local_document_warning reason="no_relevant_chunks">'
            "没有检索到与查询匹配的本地 PDF 片段。"
            "</local_document_warning>"
        )
    metrics = (
        f'\n<local_document_metrics cache_hit="{str(cache_hit).lower()}" '
        f'chunk_count="{len(chunks)}" index_time_ms="{index_time_ms}" '
        f'search_time_ms="{search_time_ms}" requested_mode="{requested_mode}" '
        f'effective_mode="{effective_mode}" '
        f'embedding_cache_hit="{str(embedding_cache_hit).lower()}" '
        f'fallback_reason="{fallback_reason}" />'
    )
    return f"{serialize_local_documents(records)}{metrics}{warning}"

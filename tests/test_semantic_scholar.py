"""Unit tests for Semantic Scholar enrichment and influence-aware ranking."""

import asyncio

import httpx

from open_deep_research import academic_tools
from open_deep_research.academic_tools import (
    _semantic_scholar_request,
    enrich_papers_with_semantic_scholar,
    normalize_rank_and_verify_papers,
    sanitize_report_citations,
    search_semantic_scholar,
    semantic_scholar_item_to_paper,
)
from open_deep_research.state import Paper


SEMANTIC_SCHOLAR_ID = "0123456789abcdef0123456789abcdef01234567"


def semantic_item(**overrides):
    item = {
        "paperId": SEMANTIC_SCHOLAR_ID,
        "title": "Deep Learning for OFDM Channel Estimation",
        "abstract": "A relevant method for OFDM channel estimation.",
        "year": 2025,
        "authors": [{"name": "Alice Chen"}],
        "url": f"https://www.semanticscholar.org/paper/{SEMANTIC_SCHOLAR_ID}",
        "externalIds": {"DOI": "10.1109/test.2025.1", "ArXiv": "2501.12345"},
        "venue": "IEEE Test Conference",
        "citationCount": 120,
        "influentialCitationCount": 12,
        "isOpenAccess": True,
        "openAccessPdf": {"url": "https://example.org/paper.pdf"},
    }
    item.update(overrides)
    return item


def base_paper(**overrides):
    data = {
        "title": "Deep Learning for OFDM Channel Estimation",
        "authors": ["Alice Chen"],
        "abstract": "A relevant method for OFDM channel estimation.",
        "published_year": 2025,
        "doi": "10.1109/test.2025.1",
        "arxiv_id": "2501.12345",
        "url": "https://doi.org/10.1109/test.2025.1",
        "source": "arxiv+crossref",
        "research_direction": "OFDM channel estimation deep learning",
        "verification_status": "source_verified",
    }
    data.update(overrides)
    return Paper(**data)


def test_semantic_scholar_record_conversion():
    paper = semantic_scholar_item_to_paper(semantic_item(), "OFDM channel estimation")
    assert paper is not None
    assert paper.doi == "10.1109/test.2025.1"
    assert paper.arxiv_id == "2501.12345"
    assert paper.semantic_scholar_id == SEMANTIC_SCHOLAR_ID
    assert paper.citation_count == 120
    assert paper.influential_citation_count == 12
    assert paper.is_open_access is True
    assert paper.open_access_url == "https://example.org/paper.pdf"


def test_semantic_scholar_search_returns_marked_records(monkeypatch):
    async def fake_request(*args, **kwargs):
        return {"data": [semantic_item()]}

    monkeypatch.setattr(academic_tools, "_semantic_scholar_request", fake_request)
    result = asyncio.run(
        search_semantic_scholar.ainvoke(
            {"query": "OFDM channel estimation"},
            config={"configurable": {"max_papers_per_query": 5}},
        )
    )
    records = academic_tools.extract_papers_from_text(result)
    assert len(records) == 1
    assert records[0]["semantic_scholar_id"] == SEMANTIC_SCHOLAR_ID
    assert records[0]["citation_count"] == 120


def test_semantic_scholar_search_degrades_to_empty_records(monkeypatch):
    async def failing_request(*args, **kwargs):
        request = httpx.Request("GET", "https://api.semanticscholar.org")
        response = httpx.Response(429, request=request)
        raise httpx.HTTPStatusError("rate limited", request=request, response=response)

    monkeypatch.setattr(academic_tools, "_semantic_scholar_request", failing_request)
    result = asyncio.run(
        search_semantic_scholar.ainvoke(
            {"query": "OFDM channel estimation"},
            config={"configurable": {"max_papers_per_query": 5}},
        )
    )
    assert academic_tools.extract_papers_from_text(result) == []
    assert 'reason="rate_limited"' in result
    assert "继续使用 arXiv" in result


def test_semantic_scholar_batch_enrichment_merges_metrics(monkeypatch):
    async def fake_request(*args, **kwargs):
        assert kwargs["json_body"] == {"ids": ["DOI:10.1109/test.2025.1"]}
        return [semantic_item()]

    monkeypatch.setattr(academic_tools, "_semantic_scholar_request", fake_request)
    enriched = asyncio.run(enrich_papers_with_semantic_scholar([base_paper()]))
    assert enriched[0].semantic_scholar_id == SEMANTIC_SCHOLAR_ID
    assert enriched[0].citation_count == 120
    assert enriched[0].influential_citation_count == 12
    assert enriched[0].source == "arxiv+crossref+semantic_scholar"


def test_semantic_scholar_enrichment_fails_open(monkeypatch):
    async def failing_request(*args, **kwargs):
        request = httpx.Request("POST", "https://api.semanticscholar.org")
        response = httpx.Response(429, request=request)
        raise httpx.HTTPStatusError("rate limited", request=request, response=response)

    monkeypatch.setattr(academic_tools, "_semantic_scholar_request", failing_request)
    original = base_paper()
    enriched = asyncio.run(enrich_papers_with_semantic_scholar([original]))
    assert enriched[0].model_dump() == original.model_dump()


def test_request_retries_rate_limit(monkeypatch):
    attempts = 0

    async def handler(request):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            return httpx.Response(429, request=request)
        return httpx.Response(200, request=request, json={"data": []})

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport)
    try:
        result = asyncio.run(
            _semantic_scholar_request(
                "GET",
                "/paper/search",
                client=client,
                params={"query": "test"},
            )
        )
    finally:
        asyncio.run(client.aclose())
    assert attempts == 3
    assert result == {"data": []}


def test_relevance_still_dominates_raw_citation_count():
    relevant = base_paper(citation_count=5, influential_citation_count=1)
    popular_but_unrelated = base_paper(
        title="Highly Cited Medical Imaging Survey",
        abstract="Medical imaging diagnosis and clinical treatment.",
        doi="10.1109/medical.2020.1",
        arxiv_id=None,
        url="https://doi.org/10.1109/medical.2020.1",
        citation_count=5000,
        influential_citation_count=500,
        research_direction=None,
    )
    accepted, _ = normalize_rank_and_verify_papers(
        [popular_but_unrelated, relevant],
        "OFDM channel estimation deep learning",
        limit=2,
        impact_weight=0.2,
    )
    assert accepted[0]["title"] == relevant.title
    assert accepted[0]["overall_score"] > accepted[1]["overall_score"]


def test_invalid_semantic_scholar_id_cannot_enter_whitelist():
    invalid = base_paper(
        doi=None,
        arxiv_id=None,
        semantic_scholar_id="invented-paper-id",
        url="https://www.semanticscholar.org/paper/invented-paper-id",
    )
    accepted, rejected = normalize_rank_and_verify_papers(
        [invalid],
        "OFDM channel estimation",
        limit=1,
    )
    assert accepted == []
    assert rejected[0]["rejection_reason"] == "missing_stable_identifier"


def test_semantic_scholar_citation_url_must_be_whitelisted():
    known_url = f"https://www.semanticscholar.org/paper/{SEMANTIC_SCHOLAR_ID}"
    paper = base_paper(
        doi=None,
        arxiv_id=None,
        semantic_scholar_id=SEMANTIC_SCHOLAR_ID,
        url=known_url,
    )
    accepted, _ = normalize_rank_and_verify_papers(
        [paper],
        "OFDM channel estimation",
        limit=1,
    )
    unknown_url = "https://www.semanticscholar.org/paper/ffffffffffffffffffffffffffffffffffffffff"
    report, rejected = sanitize_report_citations(
        f"已验证：{known_url}\n未验证：{unknown_url}",
        accepted,
    )
    assert known_url in report
    assert unknown_url not in report
    assert rejected == [unknown_url]

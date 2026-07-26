"""Unit tests for the academic research v1 data pipeline."""

import asyncio
import importlib
import json

from open_deep_research import academic_tools
from open_deep_research.academic_tools import (
    PAPER_RECORDS_END,
    PAPER_RECORDS_START,
    build_citation_context,
    deduplicate_papers,
    enrich_papers_with_crossref,
    extract_papers_from_text,
    normalize_arxiv_id,
    normalize_doi,
    normalize_rank_and_verify_papers,
    rank_crossref_matches,
    sanitize_report_citations,
    serialize_papers,
)
from open_deep_research.deep_researcher import normalize_academic_sources
from langchain_core.messages import AIMessage, HumanMessage
from open_deep_research.state import Paper


def make_paper(**overrides) -> Paper:
    data = {
        "title": "Deep Learning for OFDM Channel Estimation",
        "authors": ["Alice Chen", "Bob Li"],
        "abstract": "Deep learning improves OFDM channel estimation.",
        "published_year": 2025,
        "doi": None,
        "arxiv_id": "2501.12345",
        "url": "https://arxiv.org/abs/2501.12345",
        "source": "arxiv",
        "verification_status": "source_verified",
    }
    data.update(overrides)
    return Paper(**data)


def test_identifier_normalization():
    assert normalize_doi("https://doi.org/10.1000/XYZ.") == "10.1000/xyz"
    assert normalize_doi("not-a-doi") is None
    assert normalize_arxiv_id("https://arxiv.org/abs/2501.12345v2") == "2501.12345"


def test_serialization_round_trip():
    payload = serialize_papers([make_paper()])
    assert PAPER_RECORDS_START in payload
    assert PAPER_RECORDS_END in payload
    records = extract_papers_from_text(payload)
    assert records[0]["arxiv_id"] == "2501.12345"
    assert records[0]["title"] == "Deep Learning for OFDM Channel Estimation"


def test_deduplicate_merges_arxiv_and_crossref_metadata():
    arxiv_paper = make_paper()
    crossref_paper = make_paper(
        authors=["Alice Chen", "Bob Li", "Carol Wu"],
        doi="10.1109/test.2025.1",
        arxiv_id=None,
        url="https://doi.org/10.1109/test.2025.1",
        source="crossref",
        venue="IEEE Test Conference",
    )
    merged = deduplicate_papers([arxiv_paper, crossref_paper])
    assert len(merged) == 1
    assert merged[0].doi == "10.1109/test.2025.1"
    assert merged[0].arxiv_id == "2501.12345"
    assert merged[0].venue == "IEEE Test Conference"
    assert merged[0].source == "arxiv+crossref"


def test_crossref_enrichment_uses_only_high_confidence_title(monkeypatch):
    async def fake_query(title, rows, client=None):
        return [
            {
                "title": ["Deep Learning for OFDM Channel Estimation"],
                "DOI": "10.1109/test.2025.1",
                "URL": "https://doi.org/10.1109/test.2025.1",
                "author": [{"given": "Alice", "family": "Chen"}],
                "published-online": {"date-parts": [[2025, 1, 1]]},
                "container-title": ["IEEE Test Conference"],
            }
        ]

    monkeypatch.setattr(academic_tools, "_query_crossref_api", fake_query)
    enriched = asyncio.run(enrich_papers_with_crossref([make_paper()]))
    assert enriched[0].doi == "10.1109/test.2025.1"
    assert enriched[0].arxiv_id == "2501.12345"


def test_normalize_rank_verify_rejects_unstable_records_and_years():
    valid = make_paper()
    old = make_paper(
        title="Old Channel Estimation",
        published_year=2019,
        arxiv_id="1901.12345",
        url="https://arxiv.org/abs/1901.12345",
    )
    unstable = make_paper(
        title="Unverified Blog Summary",
        doi=None,
        arxiv_id=None,
        url="https://example.com/post",
        source="web",
    )
    accepted, rejected = normalize_rank_and_verify_papers(
        [valid, old, unstable],
        "OFDM channel estimation deep learning",
        limit=10,
        year_start=2023,
    )
    assert [paper["title"] for paper in accepted] == [valid.title]
    assert {paper["rejection_reason"] for paper in rejected} == {
        "before_year_start",
        "missing_stable_identifier",
    }


def test_citation_whitelist_and_report_sanitization():
    paper = make_paper()
    records = [paper.model_dump(mode="json")]
    context = build_citation_context(records)
    assert "共获得 1 篇" in context
    assert paper.title in context
    assert paper.url in context

    report = (
        f"有效来源：{paper.url}\n"
        "虚构来源：https://doi.org/10.9999/fake\n"
        "普通网页：https://example.com/background"
    )
    sanitized, rejected = sanitize_report_citations(report, records)
    assert paper.url in sanitized
    assert "https://example.com/background" in sanitized
    assert "https://doi.org/10.9999/fake" not in sanitized
    assert rejected == ["https://doi.org/10.9999/fake"]


def test_invalid_marked_json_is_ignored():
    malformed = f"{PAPER_RECORDS_START}\n{json.dumps({'title': 'not a list'})}\n{PAPER_RECORDS_END}"
    assert extract_papers_from_text(malformed) == []


def test_crossref_title_matching_rejects_unrelated_results():
    items = [
        {
            "title": ["An Unrelated Attention Survey"],
            "DOI": "10.1000/unrelated",
            "URL": "https://doi.org/10.1000/unrelated",
        },
        {
            "title": ["Attention Is All You Need"],
            "DOI": "10.1000/transformer",
            "URL": "https://doi.org/10.1000/transformer",
        },
    ]
    matches = rank_crossref_matches("Attention Is All You Need", items)
    assert [paper.doi for paper in matches] == ["10.1000/transformer"]


def test_normalize_academic_sources_node_updates_verified_state():
    state = {
        "research_brief": "OFDM channel estimation deep learning",
        "raw_notes": [serialize_papers([make_paper()])],
        "notes": [],
    }
    config = {
        "configurable": {
            "academic_search_enabled": True,
            "crossref_enrichment_enabled": False,
            "max_academic_papers": 10,
        }
    }
    update = asyncio.run(normalize_academic_sources(state, config))
    verified = update["verified_citations"]["value"]
    assert len(verified) == 1
    assert verified[0]["arxiv_id"] == "2501.12345"
    assert update["rejected_citations"]["value"] == []


def test_normalize_academic_sources_can_be_disabled():
    update = asyncio.run(
        normalize_academic_sources(
            {"raw_notes": [serialize_papers([make_paper()])]},
            {"configurable": {"academic_search_enabled": False}},
        )
    )
    assert update["papers"]["value"] == []
    assert update["verified_citations"]["value"] == []


def test_final_report_node_rejects_unknown_academic_url(monkeypatch):
    module = importlib.import_module("open_deep_research.deep_researcher")
    paper = make_paper().model_dump(mode="json")

    class FakeModel:
        def with_config(self, config):
            return self

        async def ainvoke(self, messages):
            return AIMessage(
                content=(
                    f"有效论文：{paper['url']}\n"
                    "虚假论文：https://doi.org/10.9999/fake"
                )
            )

    monkeypatch.setattr(module, "configurable_model", FakeModel())
    update = asyncio.run(
        module.final_report_generation(
            {
                "messages": [HumanMessage(content="研究 OFDM 信道估计")],
                "research_brief": "研究 OFDM 信道估计",
                "notes": ["研究资料"],
                "verified_citations": [paper],
                "rejected_citations": [],
            },
            {"configurable": {"citation_verification_enabled": True}},
        )
    )
    assert paper["url"] in update["final_report"]
    assert "https://doi.org/10.9999/fake" not in update["final_report"]
    rejected = update["rejected_citations"]["value"]
    assert rejected == [
        {
            "url": "https://doi.org/10.9999/fake",
            "rejection_reason": "not_in_academic_whitelist",
        }
    ]

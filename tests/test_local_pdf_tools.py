"""Unit tests for secure local PDF retrieval."""

import asyncio
import json
import re

import fitz
from langchain_core.messages import AIMessage, HumanMessage

from open_deep_research.local_pdf_tools import (
    LOCAL_DOCUMENTS_END,
    LOCAL_DOCUMENTS_START,
    clear_pdf_index_cache,
    extract_local_pdf_citations,
    get_pdf_index_cache_stats,
    load_cached_pdf_chunks,
    load_cached_embeddings,
    load_pdf_chunks,
    rank_pdf_chunks_hybrid,
    rank_pdf_chunks_semantic,
    rank_pdf_chunks,
    reciprocal_rank_fusion,
    sanitize_local_pdf_citations,
    search_local_pdfs,
)
from open_deep_research.utils import get_all_tools


def create_pdf(path, pages):
    document = fitz.open()
    for content in pages:
        page = document.new_page()
        page.insert_text((72, 72), content)
    document.save(path)
    document.close()


def extract_records(result):
    payload = re.search(
        re.escape(LOCAL_DOCUMENTS_START) + r"\s*(.*?)\s*" + re.escape(LOCAL_DOCUMENTS_END),
        result,
        re.DOTALL,
    ).group(1)
    return json.loads(payload)


def test_load_and_rank_pdf_chunks_with_page_citation(tmp_path):
    create_pdf(
        tmp_path / "ofdm.pdf",
        [
            "General introduction to wireless communication.",
            "OFDM channel estimation uses pilot symbols and neural networks.",
        ],
    )
    chunks = load_pdf_chunks(str(tmp_path), chunk_size=400, chunk_overlap=50)
    records = rank_pdf_chunks("OFDM channel estimation", chunks, limit=3)
    assert records[0]["file_name"] == "ofdm.pdf"
    assert records[0]["page"] == 2
    assert records[0]["citation"] == "local-pdf://ofdm.pdf#page=2"
    assert "pilot symbols" in records[0]["text"]


def test_ranking_returns_each_page_only_once():
    from open_deep_research.local_pdf_tools import LocalPDFChunk

    chunks = [
        LocalPDFChunk("paper.pdf", 1, 0, "OFDM pilot estimation"),
        LocalPDFChunk("paper.pdf", 1, 1, "OFDM channel estimation"),
        LocalPDFChunk("paper.pdf", 2, 0, "OFDM receiver"),
    ]
    records = rank_pdf_chunks("OFDM estimation", chunks, limit=3)
    assert [record["citation"] for record in records] == [
        "local-pdf://paper.pdf#page=1",
        "local-pdf://paper.pdf#page=2",
    ]


class FakeEmbeddings:
    def embed_documents(self, texts):
        return [
            [1.0, 0.0] if "semantic target" in text else [0.0, 1.0]
            for text in texts
        ]

    def embed_query(self, text):
        return [1.0, 0.0]


def test_semantic_ranking_can_find_a_non_lexical_match():
    from open_deep_research.local_pdf_tools import LocalPDFChunk

    chunks = [
        LocalPDFChunk("background.pdf", 1, 0, "unrelated terms"),
        LocalPDFChunk("target.pdf", 2, 0, "semantic target"),
    ]
    records = rank_pdf_chunks_semantic(
        "different vocabulary", chunks, 2, FakeEmbeddings()
    )
    assert records[0]["citation"] == "local-pdf://target.pdf#page=2"


def test_hybrid_ranking_uses_rrf_and_preserves_citations():
    from open_deep_research.local_pdf_tools import LocalPDFChunk

    chunks = [
        LocalPDFChunk("lexical.pdf", 1, 0, "OFDM pilot estimation"),
        LocalPDFChunk("semantic.pdf", 2, 0, "semantic target"),
    ]
    records = rank_pdf_chunks_hybrid(
        "OFDM estimation",
        chunks,
        2,
        FakeEmbeddings(),
        lexical_weight=0.5,
        semantic_weight=0.5,
    )
    assert {record["citation"] for record in records} == {
        "local-pdf://lexical.pdf#page=1",
        "local-pdf://semantic.pdf#page=2",
    }
    assert all(record["retrieval_mode"] == "hybrid_rrf" for record in records)


def test_rrf_is_independent_of_raw_score_scale():
    high_scale = [{"citation": "a", "score": 999, "text": "a"}]
    low_scale = [{"citation": "b", "score": 0.01, "text": "b"}]
    records = reciprocal_rank_fusion([(high_scale, 0.5), (low_scale, 0.5)], 2)
    assert {record["citation"] for record in records} == {"a", "b"}


def test_document_embeddings_are_cached_by_chunk_content():
    from open_deep_research.local_pdf_tools import LocalPDFChunk

    clear_pdf_index_cache()
    chunks = [LocalPDFChunk("paper.pdf", 1, 0, "semantic target")]
    embeddings = FakeEmbeddings()
    first, first_hit = load_cached_embeddings(
        chunks, embeddings, namespace="fake-model"
    )
    second, second_hit = load_cached_embeddings(
        chunks, embeddings, namespace="fake-model"
    )
    assert first == second
    assert first_hit is False
    assert second_hit is True


def test_hybrid_search_falls_back_to_bm25(monkeypatch, tmp_path):
    from open_deep_research import local_pdf_tools as module

    create_pdf(tmp_path / "fallback.pdf", ["OFDM pilot channel estimation."])

    def fail_to_create_embeddings(configurable):
        raise RuntimeError("embedding service unavailable")

    monkeypatch.setattr(module, "create_embedding_client", fail_to_create_embeddings)
    result = asyncio.run(
        search_local_pdfs.ainvoke(
            {"query": "OFDM pilot"},
            config={
                "configurable": {
                    "local_pdf_search_enabled": True,
                    "pdf_library_path": str(tmp_path),
                    "local_pdf_chunk_size": 400,
                    "local_pdf_retrieval_mode": "hybrid",
                }
            },
        )
    )
    assert extract_records(result)[0]["citation"] == "local-pdf://fallback.pdf#page=1"
    assert 'requested_mode="hybrid"' in result
    assert 'effective_mode="bm25"' in result
    assert 'fallback_reason="RuntimeError"' in result


def test_search_local_pdfs_returns_structured_records(tmp_path):
    create_pdf(tmp_path / "standard.pdf", ["Massive MIMO beamforming performance analysis."])
    result = asyncio.run(
        search_local_pdfs.ainvoke(
            {"query": "MIMO beamforming"},
            config={
                "configurable": {
                    "local_pdf_search_enabled": True,
                    "pdf_library_path": str(tmp_path),
                    "local_pdf_chunk_size": 400,
                }
            },
        )
    )
    records = extract_records(result)
    assert len(records) == 1
    assert records[0]["relative_path"] == "standard.pdf"


def test_pdf_index_cache_hits_and_invalidates_on_file_change(tmp_path):
    clear_pdf_index_cache()
    pdf_path = tmp_path / "cache.pdf"
    create_pdf(pdf_path, ["Initial OFDM channel estimation content."])

    first, first_hit = load_cached_pdf_chunks(
        str(tmp_path), chunk_size=400, chunk_overlap=50
    )
    second, second_hit = load_cached_pdf_chunks(
        str(tmp_path), chunk_size=400, chunk_overlap=50
    )
    assert first_hit is False
    assert second_hit is True
    assert first == second

    create_pdf(tmp_path / "new.pdf", ["New massive MIMO content."])
    third, third_hit = load_cached_pdf_chunks(
        str(tmp_path), chunk_size=400, chunk_overlap=50
    )
    assert third_hit is False
    assert len(third) > len(second)
    assert get_pdf_index_cache_stats() == {
        "hits": 1,
        "misses": 2,
        "entries": 2,
        "hit_rate_percent": 33,
    }


def test_search_local_pdfs_reports_cache_metrics(tmp_path):
    clear_pdf_index_cache()
    create_pdf(tmp_path / "metrics.pdf", ["OFDM pilot channel estimation."])
    config = {
        "configurable": {
            "local_pdf_search_enabled": True,
            "pdf_library_path": str(tmp_path),
            "local_pdf_chunk_size": 400,
        }
    }
    first = asyncio.run(
        search_local_pdfs.ainvoke({"query": "OFDM pilot"}, config=config)
    )
    second = asyncio.run(
        search_local_pdfs.ainvoke({"query": "OFDM pilot"}, config=config)
    )
    assert 'cache_hit="false"' in first
    assert 'cache_hit="true"' in second
    assert 'search_time_ms="' in second


def test_search_local_pdfs_does_not_accept_a_path_argument():
    schema = search_local_pdfs.get_input_schema().model_json_schema()
    assert "path" not in schema["properties"]
    assert "library_path" not in schema["properties"]


def test_pdf_tool_registration_requires_enablement_and_library(tmp_path):
    base_config = {
        "configurable": {
            "search_api": "none",
            "academic_search_enabled": False,
            "pdf_library_path": str(tmp_path),
        }
    }
    disabled = asyncio.run(get_all_tools(base_config))
    enabled_config = {
        "configurable": {
            **base_config["configurable"],
            "local_pdf_search_enabled": True,
        }
    }
    enabled = asyncio.run(get_all_tools(enabled_config))
    assert "search_local_pdfs" not in [item.name for item in disabled]
    assert "search_local_pdfs" in [item.name for item in enabled]


def test_missing_pdf_library_degrades_gracefully(tmp_path):
    result = asyncio.run(
        search_local_pdfs.ainvoke(
            {"query": "channel estimation"},
            config={
                "configurable": {
                    "local_pdf_search_enabled": True,
                    "pdf_library_path": str(tmp_path / "missing"),
                }
            },
        )
    )
    assert extract_records(result) == []
    assert 'reason="no_readable_pdfs"' in result


def test_disabled_pdf_search_degrades_gracefully(tmp_path):
    result = asyncio.run(
        search_local_pdfs.ainvoke(
            {"query": "channel estimation"},
            config={"configurable": {"pdf_library_path": str(tmp_path)}},
        )
    )
    assert extract_records(result) == []
    assert 'reason="search_disabled"' in result


def test_local_pdf_citation_must_be_whitelisted():
    known = "local-pdf://ofdm.pdf#page=2"
    unknown = "local-pdf://invented.pdf#page=99"
    report, rejected = sanitize_local_pdf_citations(
        f"有效：{known}\n无效：{unknown}",
        {known},
    )
    assert known in report
    assert unknown not in report
    assert rejected == [unknown]


def test_citation_extraction_ignores_unstructured_model_text():
    known = "local-pdf://ofdm.pdf#page=2"
    structured = json.dumps([{"citation": known}], ensure_ascii=False)
    citations = extract_local_pdf_citations(
        [
            "模型自行生成 local-pdf://invented.pdf#page=99",
            f"{LOCAL_DOCUMENTS_START}\n{structured}\n{LOCAL_DOCUMENTS_END}",
        ]
    )
    assert citations == {known}


def test_final_report_rejects_unknown_local_pdf_citation(monkeypatch):
    from open_deep_research import deep_researcher as module

    known = "local-pdf://ofdm.pdf#page=2"
    unknown = "local-pdf://invented.pdf#page=99"

    class FakeModel:
        def with_config(self, config):
            return self

        async def ainvoke(self, messages):
            return AIMessage(content=f"有效：{known}\n无效：{unknown}")

    monkeypatch.setattr(module, "configurable_model", FakeModel())
    update = asyncio.run(
        module.final_report_generation(
            {
                "messages": [HumanMessage(content="分析本地论文")],
                "research_brief": "分析 OFDM 信道估计",
                "raw_notes": ["模型上下文不作为本地引用白名单"],
                "notes": ["本地 PDF 研究资料"],
                "trusted_local_citations": [known],
                "verified_citations": [],
                "rejected_citations": [],
            },
            {"configurable": {"citation_verification_enabled": True}},
        )
    )
    assert known in update["final_report"]
    assert unknown not in update["final_report"]
    assert update["rejected_citations"]["value"] == [
        {
            "url": unknown,
            "rejection_reason": "not_in_local_document_whitelist",
        }
    ]

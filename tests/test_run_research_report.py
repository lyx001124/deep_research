"""Tests for the report-generation command-line entry point."""

import argparse
import importlib.util
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run_research_report.py"


def load_script_module():
    spec = importlib.util.spec_from_file_location("run_research_report", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def make_args(tmp_path):
    return argparse.Namespace(
        question="测试本地论文",
        output=tmp_path / "reports" / "report.md",
        library=tmp_path / "papers",
        retrieval_mode="hybrid",
        search_api="none",
        local_only=True,
        concurrency=2,
        iterations=3,
        tool_calls=4,
        recursion_limit=50,
    )


@pytest.mark.asyncio
async def test_run_writes_final_report_and_passes_configuration(tmp_path):
    module = load_script_module()
    args = make_args(tmp_path)
    captured = {}

    class FakeGraph:
        async def ainvoke(self, state, config):
            captured["state"] = state
            captured["config"] = config
            return {"final_report": "# 测试报告\n\n内容"}

    module.deep_researcher = FakeGraph()
    report = await module.run(args)

    assert report == "# 测试报告\n\n内容"
    assert args.output.read_text(encoding="utf-8") == report + "\n"
    configurable = captured["config"]["configurable"]
    assert configurable["local_pdf_retrieval_mode"] == "hybrid"
    assert configurable["academic_search_enabled"] is False
    assert configurable["pdf_library_path"] == str(args.library.resolve())
    assert captured["config"]["recursion_limit"] == 50


@pytest.mark.asyncio
async def test_run_rejects_empty_final_report(tmp_path):
    module = load_script_module()
    args = make_args(tmp_path)

    class EmptyGraph:
        async def ainvoke(self, state, config):
            return {"final_report": "  "}

    module.deep_researcher = EmptyGraph()
    with pytest.raises(RuntimeError, match="without a final_report"):
        await module.run(args)
    assert not args.output.exists()


def test_parser_defaults_to_hybrid_output():
    module = load_script_module()
    args = module.build_parser().parse_args(["研究问题"])
    assert args.retrieval_mode == "hybrid"
    assert args.output == Path("eval/results/final_report.md")

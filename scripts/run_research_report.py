"""Run the complete research graph and save its final report as Markdown."""

import argparse
import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
load_dotenv(PROJECT_ROOT / ".env")
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from open_deep_research.deep_researcher import deep_researcher  # noqa: E402


async def run(args: argparse.Namespace) -> str:
    """Invoke the graph once and persist the final report."""
    config = {
        "configurable": {
            "allow_clarification": False,
            "max_concurrent_research_units": args.concurrency,
            "max_researcher_iterations": args.iterations,
            "max_react_tool_calls": args.tool_calls,
            "search_api": args.search_api,
            "academic_search_enabled": not args.local_only,
            "local_pdf_search_enabled": True,
            "pdf_library_path": str(args.library.resolve()),
            "local_pdf_retrieval_mode": args.retrieval_mode,
        },
        "recursion_limit": args.recursion_limit,
    }
    result = await deep_researcher.ainvoke(
        {"messages": [HumanMessage(content=args.question)]},
        config=config,
    )
    report = str(result.get("final_report", "")).strip()
    if not report:
        raise RuntimeError("research graph completed without a final_report")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report + "\n", encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", help="research question")
    parser.add_argument(
        "--output", type=Path, default=Path("eval/results/final_report.md")
    )
    parser.add_argument("--library", type=Path, default=Path("data/papers"))
    parser.add_argument("--retrieval-mode", choices=("bm25", "hybrid"), default="hybrid")
    parser.add_argument("--search-api", default="none")
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--tool-calls", type=int, default=4)
    parser.add_argument("--recursion-limit", type=int, default=50)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = asyncio.run(run(args))
    print(f"Report saved to: {args.output.resolve()}")
    print(f"Report characters: {len(report)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""End-to-end test for the local PDF retrieval evaluation CLI."""

import json
import subprocess
import sys
from pathlib import Path

import fitz


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def create_pdf(path, pages):
    document = fitz.open()
    for content in pages:
        page = document.new_page()
        page.insert_text((72, 72), content)
    document.save(path)
    document.close()


def test_evaluation_cli_writes_machine_readable_report(tmp_path):
    library = tmp_path / "papers"
    library.mkdir()
    create_pdf(
        library / "ofdm.pdf",
        [
            "General wireless communication background.",
            "OFDM channel estimation uses pilot symbols and neural networks.",
        ],
    )
    cases = tmp_path / "cases.json"
    cases.write_text(
        json.dumps(
            [
                {
                    "id": "ofdm",
                    "query": "OFDM pilot channel estimation",
                    "relevant": [
                        {"relative_path": "ofdm.pdf", "page": 2, "grade": 2}
                    ],
                },
                {
                    "id": "ofdm-cache",
                    "query": "neural network channel estimation",
                    "relevant": [
                        {"relative_path": "ofdm.pdf", "page": 2, "grade": 2}
                    ],
                },
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    output = tmp_path / "result.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "evaluate_local_pdf_retrieval.py"),
            "--library",
            str(library),
            "--cases",
            str(cases),
            "--output",
            str(output),
            "--k",
            "2",
            "--chunk-size",
            "400",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["summary"]["case_count"] == 2
    assert report["summary"]["hit_at_k"] == 1.0
    assert report["summary"]["mrr"] == 1.0
    assert report["summary"]["cache_hit_rate"] == 0.5
    assert "Hit@2: 1.000" in completed.stdout
    assert "Precision@2: 0.500" in completed.stdout
    assert "Citation Accuracy: 1.000" in completed.stdout
    assert "Cache Hit Rate: 0.500" in completed.stdout

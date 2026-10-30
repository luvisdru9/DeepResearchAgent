"""Create the immutable bench_002 v2 snapshot from the complete ALCE PDF.

The original snapshot captured a 12,000-character HTML excerpt that ended before
ALCE's citation-recall definition.  This one-shot builder downloads the official
arXiv PDF, verifies the required definition before writing, and reuses the
already frozen RAGTruth document without mutating the v1 snapshot.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from deepresearch.retrieval.loaders import PDFDocumentLoader, default_fetcher
from deepresearch.retrieval.snapshot import (
    FrozenQueryRecord,
    load_frozen_snapshot,
    sha256_bytes,
    write_frozen_snapshot,
)
from deepresearch.schema import SourceDoc


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_ROOT = ROOT / "data/benchmarks/researchbench_v2/snapshots"
SOURCE_SNAPSHOT = SNAPSHOT_ROOT / "bench_002_20261012_v1/manifest.json"
DESTINATION = SNAPSHOT_ROOT / "bench_002_20261018_v2"
ALCE_PDF_URL = "https://arxiv.org/pdf/2305.14627"
CAPTURED_AT = datetime(2026, 10, 18, tzinfo=timezone.utc)


def _complete_alce_document(previous: SourceDoc) -> SourceDoc:
    """Download and validate the complete primary-source text before freezing it."""

    fetched = default_fetcher(ALCE_PDF_URL, 90.0, 30_000_000)
    # Reuse the exact downloaded bytes for parsing so the recorded raw hash and
    # extracted text always describe the same network response.
    loader = PDFDocumentLoader(
        timeout_seconds=90.0,
        max_bytes=30_000_000,
        max_chars=200_000,
        fetcher=lambda _url, _timeout, _max_bytes: fetched,
    )
    loaded = loader.load(ALCE_PDF_URL)
    # PDF 两栏布局会在句中插入换行；完整性探针只归一化空白，
    # 实际冻结的正文仍保留 pdftotext 原始输出。
    lowered = " ".join(loaded.content.casefold().split())
    required_phrases = (
        "citation recall",
        "recall = 1 if the concatenation of all cited passages fully supports the segment",
        "citation precision",
        # Two-column PDF extraction may interleave the left column between these
        # halves, so validate both semantic halves independently.
        "nli model to verify whether a statement is",
        "supported by its citations",
    )
    missing = [phrase for phrase in required_phrases if phrase not in lowered]
    if missing:
        raise ValueError(f"Complete ALCE capture is missing required metric text: {missing}")
    if len(loaded.content) < 100_000 or loaded.metadata.get("content_truncated"):
        raise ValueError("ALCE PDF extraction is unexpectedly short or truncated.")

    return previous.model_copy(
        update={
            "title": "Enabling Large Language Models to Generate Text with Citations (ALCE)",
            "url": ALCE_PDF_URL,
            "content": loaded.content,
            "retrieved_at": CAPTURED_AT,
            "source_type": "research_paper",
            "metadata": {
                "capture_scope": "complete_primary_source_pdf",
                "capture_source": "official_arxiv_pdf",
                "raw_pdf_sha256": sha256_bytes(fetched.body),
                **loaded.metadata,
            },
        }
    )


def main() -> None:
    """Write a new snapshot directory; immutable writers reject accidental reruns."""

    previous = load_frozen_snapshot(SOURCE_SNAPSHOT)
    by_id = {document.id: document for document in previous.documents}
    alce = _complete_alce_document(by_id["b002_alce"])
    ragtruth = by_id["b002_ragtruth"]
    queries = [
        FrozenQueryRecord.model_validate(item.model_dump(mode="json"))
        for item in previous.manifest.queries
    ]
    manifest = write_frozen_snapshot(
        DESTINATION,
        snapshot_id="bench_002_20261018_v2",
        sample_id="bench_002",
        # This is a corrected capture of the same 0.1.0 authored sample; later
        # metadata revisions continue to share the baseline snapshot contract.
        benchmark_version="0.1.0",
        captured_at=CAPTURED_AT,
        provider="curated_b3_source_repair",
        queries=queries,
        documents=[alce, ragtruth],
    )
    print(manifest.model_dump_json(indent=2))


if __name__ == "__main__":
    main()

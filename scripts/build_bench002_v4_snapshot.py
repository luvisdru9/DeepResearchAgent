"""Freeze complete reading-order ALCE and RAGTruth primary-source texts.

The v3 snapshot repaired ALCE's two-column extraction, while preserving the
historical 12,000-character RAGTruth capture.  That capture ends halfway through
the four-class hallucination taxonomy.  This builder keeps v3 ALCE unchanged and
replaces only RAGTruth with the complete official OpenReview PDF extraction.
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
SOURCE_SNAPSHOT = SNAPSHOT_ROOT / "bench_002_20261018_v3/manifest.json"
DESTINATION = SNAPSHOT_ROOT / "bench_002_20261018_v4"
RAGTRUTH_PDF_URL = "https://arxiv.org/pdf/2401.00396"
CAPTURED_AT = datetime(2026, 10, 18, 9, 15, tzinfo=timezone.utc)


def _complete_ragtruth_document(previous: SourceDoc) -> SourceDoc:
    """Download once and validate all four taxonomy definitions before freezing."""

    fetched = default_fetcher(RAGTRUTH_PDF_URL, 90.0, 30_000_000)
    loader = PDFDocumentLoader(
        timeout_seconds=90.0,
        max_bytes=30_000_000,
        max_chars=200_000,
        preserve_layout=False,
        fetcher=lambda _url, _timeout, _max_bytes: fetched,
    )
    loaded = loader.load(RAGTRUTH_PDF_URL)
    normalized = " ".join(loaded.content.casefold().split())
    required_phrases = (
        "evident conflict",
        "subtle conflict",
        "evident introduction",
        "subtle introduction",
        "unsupported or contradictory claims",
    )
    missing = [phrase for phrase in required_phrases if phrase not in normalized]
    if missing:
        raise ValueError(f"Complete RAGTruth capture is missing taxonomy text: {missing}")
    if len(loaded.content) < 50_000 or loaded.metadata.get("content_truncated"):
        raise ValueError(
            "Reading-order RAGTruth extraction is unexpectedly short or truncated "
            f"(chars={len(loaded.content)}, metadata={loaded.metadata})."
        )

    return previous.model_copy(
        update={
            "title": "RAGTruth: A Hallucination Corpus for Developing Trustworthy "
            "Retrieval-Augmented Language Models",
            "url": RAGTRUTH_PDF_URL,
            "content": loaded.content,
            "retrieved_at": CAPTURED_AT,
            "source_type": "research_paper",
            "metadata": {
                "capture_scope": "complete_primary_source_pdf_reading_order",
                "capture_source": "official_arxiv_pdf",
                "raw_pdf_sha256": sha256_bytes(fetched.body),
                **loaded.metadata,
            },
        }
    )


def main() -> None:
    """Write a new immutable snapshot while retaining every prior capture."""

    previous = load_frozen_snapshot(SOURCE_SNAPSHOT)
    by_id = {document.id: document for document in previous.documents}
    alce = by_id["b002_alce"]
    ragtruth = _complete_ragtruth_document(by_id["b002_ragtruth"])
    queries = [
        FrozenQueryRecord.model_validate(item.model_dump(mode="json"))
        for item in previous.manifest.queries
    ]
    manifest = write_frozen_snapshot(
        DESTINATION,
        snapshot_id="bench_002_20261018_v4",
        sample_id="bench_002",
        benchmark_version="0.1.0",
        captured_at=CAPTURED_AT,
        provider="curated_b3_complete_primary_sources",
        queries=queries,
        documents=[alce, ragtruth],
    )
    print(manifest.model_dump_json(indent=2))


if __name__ == "__main__":
    main()

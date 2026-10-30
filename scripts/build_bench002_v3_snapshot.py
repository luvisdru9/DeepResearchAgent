"""Create a reading-order ALCE snapshot without mutating prior B3 captures.

The v2 snapshot proved that the complete PDF contains the required methodology,
but ``pdftotext -layout`` interleaved ALCE's two columns.  That representation is
useful visually but prevents Reader quotes from being exact continuous spans.
This builder keeps the same official PDF and freezes its linear reading order.
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
SOURCE_SNAPSHOT = SNAPSHOT_ROOT / "bench_002_20261018_v2/manifest.json"
DESTINATION = SNAPSHOT_ROOT / "bench_002_20261018_v3"
ALCE_PDF_URL = "https://arxiv.org/pdf/2305.14627"
CAPTURED_AT = datetime(2026, 10, 18, 8, 45, tzinfo=timezone.utc)


def _reading_order_alce_document(previous: SourceDoc) -> SourceDoc:
    """Download once and extract the complete primary source in reading order."""

    fetched = default_fetcher(ALCE_PDF_URL, 90.0, 30_000_000)
    loader = PDFDocumentLoader(
        timeout_seconds=90.0,
        max_bytes=30_000_000,
        max_chars=200_000,
        preserve_layout=False,
        fetcher=lambda _url, _timeout, _max_bytes: fetched,
    )
    loaded = loader.load(ALCE_PDF_URL)
    normalized = " ".join(loaded.content.casefold().split())
    required_phrases = (
        "citation recall",
        "recall = 1 if the concatenation of all cited passages fully supports the segment",
        "citation precision",
        "nli model to verify whether a statement is supported by its citations",
    )
    missing = [phrase for phrase in required_phrases if phrase not in normalized]
    if missing:
        raise ValueError(f"Reading-order ALCE capture is missing metric text: {missing}")
    # Reading-order output intentionally omits the wide whitespace padding used
    # by ``-layout``, so it is substantially shorter than the v2 physical layout.
    if len(loaded.content) < 60_000 or loaded.metadata.get("content_truncated"):
        raise ValueError(
            "Reading-order ALCE extraction is unexpectedly short or truncated "
            f"(chars={len(loaded.content)}, metadata={loaded.metadata})."
        )

    return previous.model_copy(
        update={
            "title": "Enabling Large Language Models to Generate Text with Citations (ALCE)",
            "url": ALCE_PDF_URL,
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
    """Write a new immutable snapshot; prior v1/v2 captures remain reproducible."""

    previous = load_frozen_snapshot(SOURCE_SNAPSHOT)
    by_id = {document.id: document for document in previous.documents}
    alce = _reading_order_alce_document(by_id["b002_alce"])
    ragtruth = by_id["b002_ragtruth"]
    queries = [
        FrozenQueryRecord.model_validate(item.model_dump(mode="json"))
        for item in previous.manifest.queries
    ]
    manifest = write_frozen_snapshot(
        DESTINATION,
        snapshot_id="bench_002_20261018_v3",
        sample_id="bench_002",
        benchmark_version="0.1.0",
        captured_at=CAPTURED_AT,
        provider="curated_b3_reading_order_repair",
        queries=queries,
        documents=[alce, ragtruth],
    )
    print(manifest.model_dump_json(indent=2))


if __name__ == "__main__":
    main()

"""Validate the complete two-paper bench_002 snapshot and metadata revision."""

from __future__ import annotations

from deepresearch.evaluation import load_benchmark_bundle
from deepresearch.retrieval.chunking import TextChunker
from deepresearch.retrieval.snapshot import load_frozen_snapshot


MANIFEST = (
    "data/benchmarks/researchbench_v2/snapshots/"
    "bench_002_20261018_v4/manifest.json"
)


def test_v016_uses_complete_bench002_sources_and_source_faithful_b001_route():
    previous = load_benchmark_bundle(
        "data/benchmarks/researchbench_v2/dataset_v0.1.5.json"
    ).dataset_v2
    current = load_benchmark_bundle(
        "data/benchmarks/researchbench_v2/dataset_v0.1.6.json"
    ).dataset_v2

    assert previous is not None and current is not None
    assert current.metadata.version == "0.1.6"
    previous_samples = {item.id: item for item in previous.samples}
    current_samples = {item.id: item for item in current.samples}
    assert current_samples["bench_003"] == previous_samples["bench_003"]
    assert current_samples["bench_002"].snapshot.snapshot_id == "bench_002_20261018_v4"
    claim = next(
        item for item in current_samples["bench_001"].claims if item.id == "b001.c2"
    )
    assert "减少重复检索" in claim.text
    assert "后续 Agent" in claim.text
    assert "跨 Agent 复用" not in claim.text


def test_complete_ragtruth_snapshot_contains_all_four_taxonomy_definitions():
    snapshot = load_frozen_snapshot(MANIFEST)
    ragtruth = next(item for item in snapshot.documents if item.id == "b002_ragtruth")
    normalized = " ".join(ragtruth.content.casefold().split())

    assert len(ragtruth.content) > 50_000
    assert ragtruth.metadata["content_truncated"] is False
    assert ragtruth.metadata["extraction_layout"] == "reading_order"
    for category in (
        "evident conflict",
        "subtle conflict",
        "evident introduction",
        "subtle introduction",
    ):
        assert category in normalized


def test_ragtruth_taxonomy_chunks_are_retained_for_methodology_query():
    snapshot = load_frozen_snapshot(MANIFEST)
    ragtruth = next(item for item in snapshot.documents if item.id == "b002_ragtruth")
    chunks = TextChunker(
        max_chars=2500,
        overlap_chars=200,
        max_chunks_per_doc=4,
    ).chunk_documents_for_query(
        [ragtruth],
        "如何评测 DeepResearchAgent 的引用质量和幻觉率？幻觉分类是什么？",
    )
    selected = "\n".join(item.text for item in chunks).casefold()

    for category in (
        "evident conflict",
        "subtle conflict",
        "evident introduction",
        "subtle introduction",
    ):
        assert category in selected

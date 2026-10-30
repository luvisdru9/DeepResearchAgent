"""Regression coverage for the repaired immutable ALCE snapshot."""

from __future__ import annotations

import asyncio
import json

from deepresearch.agents.reader import LLMReaderAgent
from deepresearch.evaluation import load_benchmark_bundle
from deepresearch.llm import MockLLMClient
from deepresearch.quality.evidence_facets import covered_evidence_facets
from deepresearch.quality.methodology import assess_methodology_obligations
from deepresearch.retrieval.chunking import TextChunker
from deepresearch.retrieval.snapshot import load_frozen_snapshot
from deepresearch.schema import SubTask


MANIFEST = (
    "data/benchmarks/researchbench_v2/snapshots/"
    "bench_002_20261018_v2/manifest.json"
)
QUERY = "如何评测 DeepResearchAgent 的引用质量和幻觉率？"


def test_v014_changes_only_bench002_snapshot_reference():
    previous = load_benchmark_bundle(
        "data/benchmarks/researchbench_v2/dataset_v0.1.3.json"
    ).dataset_v2
    current = load_benchmark_bundle(
        "data/benchmarks/researchbench_v2/dataset_v0.1.4.json"
    ).dataset_v2

    assert previous is not None and current is not None
    assert current.metadata.version == "0.1.4"
    previous_refs = {item.id: item.snapshot for item in previous.samples}
    current_refs = {item.id: item.snapshot for item in current.samples}
    assert current_refs["bench_001"] == previous_refs["bench_001"]
    assert current_refs["bench_003"] == previous_refs["bench_003"]
    assert current_refs["bench_002"] != previous_refs["bench_002"]
    assert current_refs["bench_002"].snapshot_id == "bench_002_20261018_v2"


def test_complete_alce_snapshot_contains_metric_definition_and_preserves_v1():
    current = load_frozen_snapshot(MANIFEST)
    previous = load_frozen_snapshot(
        "data/benchmarks/researchbench_v2/snapshots/"
        "bench_002_20261012_v1/manifest.json"
    )
    current_alce = next(item for item in current.documents if item.id == "b002_alce")
    previous_alce = next(item for item in previous.documents if item.id == "b002_alce")
    normalized = " ".join(current_alce.content.casefold().split())

    assert len(previous_alce.content) == 12_000
    assert len(current_alce.content) > 100_000
    assert current_alce.metadata["content_truncated"] is False
    assert "citation recall" in normalized
    assert (
        "recall = 1 if the concatenation of all cited passages fully supports the segment"
        in normalized
    )
    assert "for this question, citation recall = 2 / 3 = 66%" in normalized


def test_alce_definition_survives_chunk_selection_reader_grounding_and_method_gate():
    snapshot = load_frozen_snapshot(MANIFEST)
    alce = next(item for item in snapshot.documents if item.id == "b002_alce")
    chunker = TextChunker(max_chars=2500, overlap_chars=200, max_chunks_per_doc=4)
    chunks = chunker.chunk_documents_for_query(
        [alce],
        f"{QUERY} 引用覆盖率 citation completeness",
    )
    definition_chunk = next(
        item for item in chunks if "fully supports the segment" in item.text.casefold()
    )
    reader_doc = chunker.to_reader_docs([alce], [definition_chunk])[0]
    quote_start = reader_doc.content.casefold().index("citation recall")
    quote_end = reader_doc.content.casefold().index(
        "for this question, citation recall", quote_start
    )
    formula_end = reader_doc.content.index("\n", quote_end)
    quote = reader_doc.content[quote_start:formula_end]
    response = json.dumps(
        {
            "evidence": [
                {
                    "source_id": "b002_alce",
                    "quote": quote,
                    "claim": (
                        "ALCE citation recall checks how many generated statements are fully "
                        "supported by their cited passages."
                    ),
                    "confidence": 0.95,
                }
            ]
        },
        ensure_ascii=False,
    )
    reader = LLMReaderAgent(MockLLMClient(responses=[response]), fallback_to_rule=False)

    evidence = asyncio.run(
        reader.extract_async(
            SubTask(id="task_coverage", question="定义 citation completeness 和分母"),
            [reader_doc],
        )
    )
    covered = covered_evidence_facets(
        ["citation_presence_scope"],
        [f"{evidence[0].claim}\n{evidence[0].quote}"],
    )
    diagnostics = assess_methodology_obligations(
        QUERY,
        [
            (
                "引用覆盖率以需要证据的原子主张为分析单位，分子是具有有效引用的主张数，"
                "分母是需要证据的原子主张总数。",
                [evidence[0].id],
            )
        ],
        evidence,
    )

    assert evidence[0].quote == quote
    assert evidence[0].chunk_id == definition_chunk.id
    assert covered == ("citation_presence_scope",)
    assert diagnostics[0].obligation_type == "citation_coverage_definition"
    assert diagnostics[0].status == "satisfied"

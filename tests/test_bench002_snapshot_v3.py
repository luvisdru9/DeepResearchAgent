"""Regression coverage for ALCE's linear reading-order snapshot."""

from __future__ import annotations

import asyncio
import json

from deepresearch.agents.reader import LLMReaderAgent, _ground_quote
from deepresearch.evaluation import load_benchmark_bundle
from deepresearch.llm import MockLLMClient
from deepresearch.quality.evidence_facets import covered_evidence_facets
from deepresearch.retrieval.chunking import TextChunker
from deepresearch.retrieval.snapshot import load_frozen_snapshot
from deepresearch.schema import SubTask


MANIFEST = (
    "data/benchmarks/researchbench_v2/snapshots/"
    "bench_002_20261018_v3/manifest.json"
)
QUERY = "如何评测 DeepResearchAgent 的引用质量和幻觉率？"


def test_v015_changes_only_bench002_snapshot_reference():
    previous = load_benchmark_bundle(
        "data/benchmarks/researchbench_v2/dataset_v0.1.4.json"
    ).dataset_v2
    current = load_benchmark_bundle(
        "data/benchmarks/researchbench_v2/dataset_v0.1.5.json"
    ).dataset_v2

    assert previous is not None and current is not None
    assert current.metadata.version == "0.1.5"
    previous_refs = {item.id: item.snapshot for item in previous.samples}
    current_refs = {item.id: item.snapshot for item in current.samples}
    assert current_refs["bench_001"] == previous_refs["bench_001"]
    assert current_refs["bench_003"] == previous_refs["bench_003"]
    assert current_refs["bench_002"] != previous_refs["bench_002"]
    assert current_refs["bench_002"].snapshot_id == "bench_002_20261018_v3"


def test_reading_order_snapshot_keeps_definition_as_continuous_groundable_text():
    snapshot = load_frozen_snapshot(MANIFEST)
    alce = next(item for item in snapshot.documents if item.id == "b002_alce")
    normalized = " ".join(alce.content.casefold().split())
    quote = (
        "Recall = 1 if the concatenation of all cited passages fully supports the segment."
    )

    assert len(alce.content) > 60_000
    assert alce.metadata["content_truncated"] is False
    assert alce.metadata["extraction_layout"] == "reading_order"
    assert quote.casefold() in normalized
    assert _ground_quote(alce.content, quote) is not None
    assert (
        "nli model to verify whether a statement is supported by its citations"
        in normalized
    )


def test_reading_order_definition_survives_chunking_and_llm_grounding():
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
    candidate = (
        "Recall = 1 if the concatenation of all cited passages fully supports the segment."
    )
    response = json.dumps(
        {
            "evidence": [
                {
                    "source_id": "b002_alce",
                    "quote": candidate,
                    "claim": (
                        "ALCE citation recall checks whether all cited passages jointly "
                        "support the evaluated segment."
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

    assert evidence[0].quote == candidate
    assert evidence[0].chunk_id == definition_chunk.id
    assert covered == ("citation_presence_scope",)

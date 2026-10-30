"""Build the three immutable B1 smoke snapshots from reviewed local/replay sources.

This script is intentionally one-shot: ``write_frozen_snapshot`` refuses to overwrite
an existing snapshot. A content change must use a new snapshot ID and benchmark version.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from deepresearch.retrieval.snapshot import FrozenQueryRecord, write_frozen_snapshot
from deepresearch.schema import SourceDoc


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "data/benchmarks/researchbench_v2/snapshots"
CAPTURED_AT = datetime(2026, 10, 12, tzinfo=timezone.utc)
BENCHMARK_VERSION = "0.1.0"


def _local_document(
    source_id: str,
    relative_path: str,
    *,
    source_type: str,
) -> SourceDoc:
    """Freeze the current bytes of a project-owned technical source."""

    path = ROOT / relative_path
    raw = path.read_text(encoding="utf-8")
    title = path.stem
    for line in raw.splitlines():
        if line.startswith("# "):
            title = line[2:].strip()
            break
    return SourceDoc(
        id=source_id,
        title=title,
        url=f"repo://{relative_path}",
        content=raw,
        source_type=source_type,
        retrieved_at=CAPTURED_AT,
        metadata={"capture_scope": "complete_file", "source_path": relative_path},
    )


def _replay_document(
    replay_path: str,
    *,
    source_id: str,
    url_contains: str,
) -> SourceDoc:
    """Promote a previously captured primary source into the immutable pilot snapshot."""

    payload = json.loads((ROOT / replay_path).read_text(encoding="utf-8"))
    for docs in payload.get("docs_by_task", {}).values():
        for raw_document in docs:
            if url_contains in raw_document.get("url", ""):
                document = SourceDoc.model_validate(raw_document)
                return document.model_copy(
                    update={
                        "id": source_id,
                        "source_type": "research_paper",
                        "metadata": {
                            **document.metadata,
                            "capture_scope": "retrieved_primary_source_text",
                            "capture_replay": replay_path,
                        },
                    }
                )
    raise ValueError(f"Could not find URL containing '{url_contains}' in {replay_path}.")


def main() -> None:
    """Create one isolated source universe for each B1 benchmark sample."""

    specifications = [
        {
            "sample_id": "bench_001",
            "snapshot_id": "bench_001_20261012_v1",
            "documents": [
                _local_document("b001_plan", "PLAN.md", source_type="official_document"),
                _local_document(
                    "b001_multi_agent",
                    "data/local_corpus/multi_agent_research.md",
                    source_type="technical_document",
                ),
            ],
            "queries": [
                ("b001.q1", "多智能体深度研究 DAG 编排 共享记忆", ["b001_plan", "b001_multi_agent"]),
                ("b001.q2", "DAG 子任务依赖 并发 失败重试", ["b001_multi_agent", "b001_plan"]),
                ("b001.q3", "共享记忆 状态 证据 来源血缘", ["b001_multi_agent", "b001_plan"]),
            ],
        },
        {
            "sample_id": "bench_002",
            "snapshot_id": "bench_002_20261012_v1",
            "documents": [
                _replay_document(
                    "runs/20261011T172338Z_a27e2f1dcb14_replay.json",
                    source_id="b002_alce",
                    url_contains="ar5iv.labs.arxiv.org/html/2305.14627",
                ),
                _replay_document(
                    "runs/20261012T051226Z_aa53edf12e44_replay.json",
                    source_id="b002_ragtruth",
                    url_contains="openreview.net/pdf/893823e19b1fde8b5c59d7e42a95767dae425ab1.pdf",
                ),
            ],
            "queries": [
                ("b002.q1", "DeepResearchAgent 引用质量 幻觉率 评测", ["b002_alce", "b002_ragtruth"]),
                ("b002.q2", "ALCE citation correctness completeness metrics", ["b002_alce", "b002_ragtruth"]),
                ("b002.q3", "RAGTruth word-level hallucination annotation", ["b002_ragtruth", "b002_alce"]),
            ],
        },
        {
            "sample_id": "bench_003",
            "snapshot_id": "bench_003_20261012_v1",
            "documents": [
                _local_document("b003_plan", "PLAN.md", source_type="official_document"),
                _local_document(
                    "b003_red_blue",
                    "data/local_corpus/red_blue_verification.md",
                    source_type="technical_document",
                ),
            ],
            "queries": [
                ("b003.q1", "Red-Blue 对抗核查 深度研究报告", ["b003_red_blue", "b003_plan"]),
                ("b003.q2", "Red Agent 事实 引用 逻辑 遗漏", ["b003_red_blue", "b003_plan"]),
                ("b003.q3", "Blue 证据修订 Verifier 最终核验", ["b003_red_blue", "b003_plan"]),
            ],
        },
    ]

    hashes: dict[str, str] = {}
    for item in specifications:
        queries = [
            FrozenQueryRecord(id=query_id, query=query, result_source_ids=result_ids)
            for query_id, query, result_ids in item["queries"]
        ]
        manifest = write_frozen_snapshot(
            OUTPUT_ROOT / item["snapshot_id"],
            snapshot_id=item["snapshot_id"],
            sample_id=item["sample_id"],
            benchmark_version=BENCHMARK_VERSION,
            captured_at=CAPTURED_AT,
            provider="curated_b1_pilot",
            queries=queries,
            documents=item["documents"],
        )
        hashes[item["sample_id"]] = manifest.content_hash
    print(json.dumps(hashes, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

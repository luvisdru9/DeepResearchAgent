import json
import shutil
import socket
from pathlib import Path

import pytest

from deepresearch.agents.reader import ReaderAgent
from deepresearch.config import RetrievalConfig
from deepresearch.evaluation.benchmark import (
    load_benchmark_bundle,
    resolve_snapshot_manifest_path,
)
from deepresearch.evaluation.runner import build_generation_task
from deepresearch.retrieval.factory import create_search_provider
from deepresearch.retrieval.frozen import FrozenSearchProvider
from deepresearch.retrieval.snapshot import load_frozen_snapshot
from deepresearch.schema import SubTask


BENCHMARK_PATH = Path("data/benchmarks/researchbench_v2/dataset.json")


def test_frozen_track_repeats_retrieval_and_evidence_source_sets(monkeypatch):
    """The same snapshot must produce the same source universe without network access."""

    def fail_network(*args, **kwargs):
        del args, kwargs
        raise AssertionError("Frozen retrieval attempted to access the network.")

    monkeypatch.setattr(socket, "create_connection", fail_network)
    bundle = load_benchmark_bundle(BENCHMARK_PATH)
    assert bundle.dataset_v2 is not None
    reader = ReaderAgent()

    for sample in bundle.dataset_v2.samples:
        assert sample.snapshot is not None
        manifest_path = resolve_snapshot_manifest_path(BENCHMARK_PATH, sample.snapshot)
        provider = FrozenSearchProvider(manifest_path)
        subtask = SubTask(id=f"{sample.id}_research", question=sample.question)

        first_docs, first_diagnostics = provider.search_with_diagnostics(sample.question, top_k=5)
        second_docs, second_diagnostics = provider.search_with_diagnostics(sample.question, top_k=5)
        first_evidence = reader.extract(subtask, first_docs)
        second_evidence = reader.extract(subtask, second_docs)

        assert [doc.id for doc in first_docs] == [doc.id for doc in second_docs]
        assert {item.source_id for item in first_evidence} == {
            item.source_id for item in second_evidence
        }
        assert first_diagnostics["offline"] is True
        assert first_diagnostics["snapshot_content_hash"] == second_diagnostics[
            "snapshot_content_hash"
        ]


def test_frozen_snapshot_rejects_modified_document(tmp_path):
    bundle = load_benchmark_bundle(BENCHMARK_PATH)
    assert bundle.dataset_v2 is not None
    snapshot = bundle.dataset_v2.samples[0].snapshot
    assert snapshot is not None
    source_manifest = resolve_snapshot_manifest_path(BENCHMARK_PATH, snapshot)
    copied_root = tmp_path / "snapshot"
    shutil.copytree(source_manifest.parent, copied_root)
    manifest = json.loads((copied_root / "manifest.json").read_text(encoding="utf-8"))
    document_path = copied_root / manifest["documents"][0]["path"]
    document_path.write_text(document_path.read_text(encoding="utf-8") + "tampered", encoding="utf-8")

    with pytest.raises(ValueError, match="document hash mismatch"):
        load_frozen_snapshot(copied_root / "manifest.json")


def test_frozen_provider_factory_requires_and_loads_manifest():
    with pytest.raises(ValueError, match="frozen_manifest_path is required"):
        create_search_provider(RetrievalConfig(provider="frozen"))

    bundle = load_benchmark_bundle(BENCHMARK_PATH)
    assert bundle.dataset_v2 is not None
    snapshot = bundle.dataset_v2.samples[0].snapshot
    assert snapshot is not None
    manifest_path = resolve_snapshot_manifest_path(BENCHMARK_PATH, snapshot)
    provider = create_search_provider(
        RetrievalConfig(provider="frozen", frozen_manifest_path=manifest_path)
    )
    assert isinstance(provider, FrozenSearchProvider)


def test_generation_task_cannot_contain_gold_claims_or_rubric():
    """Benchmark answers are normalized only after generation, never added to ResearchTask."""

    bundle = load_benchmark_bundle(BENCHMARK_PATH)
    assert bundle.dataset_v2 is not None
    authored = bundle.dataset_v2.samples[0]
    normalized = bundle.samples[0]
    task_payload = json.dumps(
        build_generation_task(normalized).model_dump(mode="json"),
        ensure_ascii=False,
    )

    assert all(claim.text not in task_payload for claim in authored.claims)
    assert all(criterion.description not in task_payload for criterion in authored.rubric)

import json
from pathlib import Path

import pytest

from deepresearch.evaluation import (
    AtomicClaim,
    BenchmarkDatasetV2,
    benchmark_v2_json_schema,
    export_benchmark_v2_json_schema,
    load_benchmark,
    load_benchmark_bundle,
)


def test_full_researchbench_has_required_coverage_and_fields():
    samples = load_benchmark("data/researchbench_full.json")

    assert len(samples) == 30
    assert {sample.task_type for sample in samples} == {
        "fact_qa",
        "comparative_research",
        "trend_analysis",
        "controversy_multihop",
    }
    assert {sample.language for sample in samples} == {"zh", "en"}
    assert all(sample.expected_claims for sample in samples)
    assert all(sample.required_sources for sample in samples)
    assert all(sample.forbidden_hallucinations for sample in samples)
    assert all(sample.judge_rubric for sample in samples)


def test_benchmark_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text(
        json.dumps(
            [
                {"id": "same", "question": "first"},
                {"id": "same", "question": "second"},
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate benchmark"):
        load_benchmark(path)


def test_v2_smoke_dataset_loads_and_normalizes_for_existing_runners():
    path = "data/benchmarks/researchbench_v2/dataset.json"

    bundle = load_benchmark_bundle(path)
    normalized = load_benchmark(path)

    assert bundle.legacy is False
    assert bundle.schema_version == "2.0"
    assert bundle.dataset_v2 is not None
    assert bundle.dataset_v2.metadata.version == "0.1.0"
    assert bundle.dataset_v2.metadata.track == "frozen"
    assert bundle.dataset_v2.metadata.release_tier == "development_draft"
    assert bundle.dataset_v2.metadata.human_calibrated is False
    assert bundle.dataset_v2.metadata.formal_benchmark_eligible is False
    assert {sample.task_type for sample in normalized} == {
        "fact_qa",
        "comparative_research",
        "controversy_multihop",
    }
    assert all(item.snapshot is not None for item in bundle.dataset_v2.samples)
    assert len(normalized) == 3
    assert all(sample.expected_claims for sample in normalized)
    assert all(sample.required_sources for sample in normalized)
    assert all(sample.judge_rubric for sample in normalized)


def test_engineering_automatic_metadata_cannot_claim_formal_release():
    source = json.loads(
        Path("data/benchmarks/researchbench_v2/dataset.json").read_text(encoding="utf-8")
    )
    source["metadata"].update(
        {
            "release_tier": "engineering_automatic",
            "human_calibrated": False,
            "formal_benchmark_eligible": False,
        }
    )
    dataset = BenchmarkDatasetV2.model_validate(source)
    assert dataset.metadata.release_tier == "engineering_automatic"

    source["metadata"]["formal_benchmark_eligible"] = True
    with pytest.raises(ValueError, match="human_calibrated"):
        BenchmarkDatasetV2.model_validate(source)


def test_metadata_revision_can_reuse_explicit_snapshot_base_version():
    bundle = load_benchmark_bundle(
        "data/benchmarks/researchbench_v2/dataset_v0.1.1.json"
    )

    assert bundle.dataset_v2 is not None
    assert bundle.dataset_v2.metadata.version == "0.1.1"
    assert bundle.dataset_v2.metadata.snapshot_base_version == "0.1.0"


def test_v013_routes_do_not_exceed_the_frozen_red_blue_sources():
    bundle = load_benchmark_bundle(
        "data/benchmarks/researchbench_v2/dataset_v0.1.3.json"
    )

    assert bundle.dataset_v2 is not None
    assert bundle.dataset_v2.metadata.version == "0.1.3"
    assert bundle.dataset_v2.metadata.snapshot_base_version == "0.1.0"
    sample = next(item for item in bundle.dataset_v2.samples if item.id == "bench_003")
    routes = {
        claim.id: claim.satisfaction_routes[0].text for claim in sample.claims
    }
    assert "新增 Evidence" not in routes["b003.c2"]
    assert "阻塞通过" not in routes["b003.c3"]
    assert "依据已有 Evidence 修订报告" in routes["b003.c2"]
    assert "标记仍然存在的不确定性" in routes["b003.c3"]


def test_v012_declares_complete_satisfaction_routes_without_rewriting_snapshots():
    bundle = load_benchmark_bundle(
        "data/benchmarks/researchbench_v2/dataset_v0.1.2.json"
    )

    assert bundle.dataset_v2 is not None
    assert bundle.dataset_v2.metadata.version == "0.1.2"
    assert bundle.dataset_v2.metadata.snapshot_base_version == "0.1.0"
    assert all(
        claim.satisfaction_routes
        for sample in bundle.dataset_v2.samples
        for claim in sample.claims
    )
    assert bundle.dataset_v2.samples[2].claims[1].satisfaction_routes[0].id == (
        "b003.c2.main"
    )


def test_legacy_atomic_claim_gets_stable_default_satisfaction_route():
    claim = AtomicClaim(id="gold.c1", text="完整主张", accepted_variants=["措辞别名"])

    assert [route.model_dump() for route in claim.effective_satisfaction_routes()] == [
        {"id": "gold.c1.main", "text": "完整主张"}
    ]


def test_legacy_benchmark_is_explicitly_marked():
    bundle = load_benchmark_bundle("data/researchbench.json")

    assert bundle.legacy is True
    assert bundle.schema_version == "1.0"
    assert bundle.dataset_v2 is None


def test_v2_rejects_duplicate_claim_ids_and_invalid_weights(tmp_path):
    source = json.loads(
        Path("data/benchmarks/researchbench_v2/dataset.json").read_text(encoding="utf-8")
    )
    duplicate = json.loads(json.dumps(source))
    duplicate["samples"][0]["claims"][1]["id"] = duplicate["samples"][0]["claims"][0]["id"]
    duplicate_path = tmp_path / "duplicate_claim.json"
    duplicate_path.write_text(json.dumps(duplicate), encoding="utf-8")

    invalid_weight = json.loads(json.dumps(source))
    invalid_weight["samples"][0]["claims"][0]["weight"] = 0
    invalid_weight_path = tmp_path / "invalid_weight.json"
    invalid_weight_path.write_text(json.dumps(invalid_weight), encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate benchmark ids"):
        load_benchmark_bundle(duplicate_path)
    with pytest.raises(ValueError, match="greater than 0"):
        load_benchmark_bundle(invalid_weight_path)


def test_frozen_v2_requires_snapshot_for_every_sample(tmp_path):
    source = json.loads(
        Path("data/benchmarks/researchbench_v2/dataset.json").read_text(encoding="utf-8")
    )
    source["metadata"]["track"] = "frozen"
    source["samples"][0]["snapshot"] = None
    path = tmp_path / "frozen_without_snapshot.json"
    path.write_text(json.dumps(source), encoding="utf-8")

    with pytest.raises(ValueError, match="require snapshots"):
        load_benchmark_bundle(path)


def test_v2_json_schema_can_be_exported(tmp_path):
    schema = benchmark_v2_json_schema()
    path = export_benchmark_v2_json_schema(tmp_path / "benchmark_v2.schema.json")
    exported = json.loads(path.read_text(encoding="utf-8"))

    assert schema["title"] == "BenchmarkDatasetV2"
    assert "BenchmarkSampleV2" in schema["$defs"]
    assert exported == schema


def test_frozen_v2_rejects_required_source_absent_from_snapshot(tmp_path):
    source = json.loads(
        Path("data/benchmarks/researchbench_v2/dataset.json").read_text(encoding="utf-8")
    )
    source["samples"][0]["evidence_requirements"][0]["required_source_ids"] = [
        "missing_source"
    ]
    # 保留原目录结构，使快照相对路径仍可解析。
    path = Path("data/benchmarks/researchbench_v2") / "invalid_required_source.tmp.json"
    try:
        path.write_text(json.dumps(source), encoding="utf-8")
        with pytest.raises(ValueError, match="Required source ids are absent"):
            load_benchmark_bundle(path)
    finally:
        path.unlink(missing_ok=True)


def test_frozen_v2_requires_source_ids_for_every_evidence_requirement(tmp_path):
    source = json.loads(
        Path("data/benchmarks/researchbench_v2/dataset.json").read_text(encoding="utf-8")
    )
    source["samples"][0]["evidence_requirements"][0]["required_source_ids"] = []
    path = tmp_path / "missing_required_source_annotation.json"
    path.write_text(json.dumps(source), encoding="utf-8")

    with pytest.raises(ValueError, match="need required_source_ids"):
        load_benchmark_bundle(path)

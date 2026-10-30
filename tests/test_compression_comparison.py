import json

from deepresearch.config import AppConfig, EvaluationConfig, MemoryConfig, RuntimeConfig
from deepresearch.evaluation import CompressionComparisonRunner


def test_compression_comparison_persists_both_variants(tmp_path):
    benchmark = tmp_path / "benchmark.json"
    benchmark.write_text(
        json.dumps(
            [
                {
                    "id": "sample",
                    "question": "如何评测引用质量和幻觉率？",
                    "expected_claims": ["引用质量"],
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    config = AppConfig(
        runtime=RuntimeConfig(
            run_dir=tmp_path / "runs",
            max_subtasks=1,
            enable_redteam=False,
        ),
        memory=MemoryConfig(enabled=True, compression_max_evidence=1),
        evaluation=EvaluationConfig(benchmark_path=benchmark),
    )
    runner = CompressionComparisonRunner(config)

    variants = runner.run(compressed_max_evidence=1)

    assert [item.strategy for item in variants] == ["compression_off", "compression_1"]
    assert all(item.succeeded_count == 1 for item in variants)
    assert variants[0].results[0].snapshot_id == variants[1].results[0].snapshot_id
    assert runner.last_output_path is not None and runner.last_output_path.exists()
    payload = json.loads(runner.last_output_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "2.0"
    assert payload["paired_evidence"] is True
    assert len(payload["results"]) == 2

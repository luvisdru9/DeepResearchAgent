import json
from pathlib import Path

from deepresearch.config import AppConfig, EvaluationConfig, RetrievalConfig, RuntimeConfig
from deepresearch.evaluation import SYSTEM_STRATEGIES, SystemComparisonRunner
from deepresearch.evaluation import load_benchmark_bundle


def test_system_comparison_runs_plan_matrix_and_writes_table(tmp_path):
    benchmark = tmp_path / "benchmark.json"
    benchmark.write_text(
        json.dumps(
            [
                {
                    "id": "sample",
                    "question": "共享记忆如何帮助研究？",
                    "domain": "agent_system",
                    "expected_claims": ["共享记忆"],
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    config = AppConfig(
        retrieval=RetrievalConfig(provider="local", corpus_dir="data/local_corpus", top_k=2),
        runtime=RuntimeConfig(run_dir=tmp_path / "runs", max_subtasks=2),
        evaluation=EvaluationConfig(benchmark_path=benchmark, provider="rule"),
    )
    runner = SystemComparisonRunner(config)

    variants = runner.run()

    assert [item.strategy for item in variants] == list(SYSTEM_STRATEGIES)
    assert all(item.succeeded_count == 1 for item in variants)
    assert variants[0].feature_flags["max_subtasks"] == 1
    assert variants[0].feature_flags["memory"] is False
    assert variants[2].feature_flags["memory"] is True
    assert variants[3].feature_flags["redteam"] is True
    assert variants[4].feature_flags["compression"] is True
    assert runner.memory_seed_path is not None
    assert runner.last_output_path is not None and runner.last_output_path.exists()
    assert runner.last_markdown_path is not None and runner.last_markdown_path.exists()
    assert "dag_memory_redteam_compress" in runner.last_markdown_path.read_text(encoding="utf-8")


def test_system_comparison_empty_seed_budgets_and_selective_judge(tmp_path):
    benchmark = tmp_path / "benchmark.json"
    benchmark.write_text(
        json.dumps(
            [
                {
                    "id": "sample",
                    "question": "证据如何支持结论？",
                    "expected_claims": ["证据"],
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    config = AppConfig(
        retrieval=RetrievalConfig(provider="local", corpus_dir="data/local_corpus", top_k=1),
        runtime=RuntimeConfig(run_dir=tmp_path / "runs", max_subtasks=1),
        evaluation=EvaluationConfig(benchmark_path=benchmark, provider="rule"),
    )
    runner = SystemComparisonRunner(config)

    variants = runner.run(
        strategies=["baseline_linear", "dag_memory_redteam_compress"],
        judge_strategies=["dag_memory_redteam_compress"],
        memory_seed_mode="empty",
        max_llm_calls=5,
        max_total_tokens=1000,
        max_runtime_seconds=30,
    )

    assert variants[0].results[0].semantic_judge_enabled is False
    assert variants[1].results[0].semantic_judge_enabled is True
    assert variants[0].results[0].source_count >= 0
    assert runner.memory_seed_sha256
    assert runner.last_output_path is not None
    payload = json.loads(runner.last_output_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "1.1"
    assert payload["memory_seed_mode"] == "empty"
    assert payload["task_budgets"]["max_llm_calls"] == 5
    assert len(payload["paired_deltas"]) == 1


def test_track_a_mock_three_samples_cover_all_fifteen_combinations(tmp_path):
    source = json.loads(Path("data/researchbench_full.json").read_text(encoding="utf-8"))
    selected_ids = {"full_005", "full_015", "full_025"}
    benchmark = tmp_path / "track_a_mock.json"
    benchmark.write_text(
        json.dumps([item for item in source if item["id"] in selected_ids], ensure_ascii=False),
        encoding="utf-8",
    )
    config = AppConfig(
        retrieval=RetrievalConfig(provider="local", corpus_dir="data/local_corpus", top_k=2),
        runtime=RuntimeConfig(run_dir=tmp_path / "runs", max_subtasks=2),
        evaluation=EvaluationConfig(benchmark_path=benchmark, provider="rule"),
    )
    runner = SystemComparisonRunner(config)

    variants = runner.run(
        judge_strategies=["baseline_linear", "dag_memory_redteam_compress"],
        memory_seed_mode="empty",
        max_llm_calls=40,
        max_total_tokens=120000,
        max_runtime_seconds=600,
    )

    assert len(variants) == 5
    assert sum(item.succeeded_count for item in variants) == 15
    assert sum(item.failed_count for item in variants) == 0
    assert {row.sample_id for item in variants for row in item.results} == selected_ids
    assert all(
        row.semantic_judge_enabled
        for item in variants
        if item.strategy in {"baseline_linear", "dag_memory_redteam_compress"}
        for row in item.results
    )
    replay = json.loads(
        next((tmp_path / "runs").glob("*_replay.json")).read_text(encoding="utf-8")
    )
    assert replay["task"]["max_llm_calls"] == 40
    assert replay["task"]["max_total_tokens"] == 120000


def test_system_comparison_filters_samples_without_substitution(tmp_path):
    benchmark = tmp_path / "benchmark.json"
    benchmark.write_text(
        json.dumps(
            [
                {"id": "one", "question": "first", "expected_claims": ["first"]},
                {"id": "two", "question": "second", "expected_claims": ["second"]},
            ]
        ),
        encoding="utf-8",
    )
    config = AppConfig(
        retrieval=RetrievalConfig(provider="local", corpus_dir="data/local_corpus", top_k=1),
        runtime=RuntimeConfig(run_dir=tmp_path / "runs", max_subtasks=1),
        evaluation=EvaluationConfig(benchmark_path=benchmark, provider="rule"),
    )
    runner = SystemComparisonRunner(config)

    results = runner.run(strategies=["baseline_linear"], sample_ids=["two"])

    assert [item.sample_id for item in results[0].results] == ["two"]
    payload = json.loads(runner.last_output_path.read_text(encoding="utf-8"))
    assert payload["sample_ids"] == ["two"]


def test_system_comparison_marks_v2_snapshot_provider_in_summary(tmp_path):
    config = AppConfig(
        runtime=RuntimeConfig(run_dir=tmp_path / "runs", max_subtasks=1),
        evaluation=EvaluationConfig(provider="rule"),
    )
    bundle = load_benchmark_bundle("data/benchmarks/researchbench_v2/dataset.json")
    authored = {item.id: item for item in bundle.dataset_v2.samples}
    sample = bundle.samples[0]
    runner = SystemComparisonRunner(config)

    result = runner._run_variant(
        "baseline_linear",
        [sample],
        benchmark_path=Path("data/benchmarks/researchbench_v2/dataset.json"),
        authored_samples=authored,
        semantic_judge=False,
        max_llm_calls=10,
        max_total_tokens=10000,
        max_runtime_seconds=60,
    )

    assert result.succeeded_count == 1
    assert result.feature_flags["retrieval_provider"] == "frozen_per_sample"
    assert result.feature_flags["frozen_snapshot_mode"] is True

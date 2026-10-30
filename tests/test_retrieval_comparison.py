import json

from deepresearch.config import AppConfig, EvaluationConfig, RetrievalConfig, RuntimeConfig
from deepresearch.evaluation import RetrievalComparisonRunner
from deepresearch.retrieval.base import SearchProvider
from deepresearch.schema import SourceDoc


class StaticProvider(SearchProvider):
    """Small deterministic provider used to compare the evaluation wiring."""

    def __init__(self, provider_name: str) -> None:
        self.provider_name = provider_name

    def search(self, query: str, top_k: int = 5) -> list[SourceDoc]:
        del top_k
        return [
            SourceDoc(
                id=f"{self.provider_name}_1",
                title=f"{self.provider_name} source",
                url=f"https://{self.provider_name}.example/source",
                content=f"{query} 的证据支持引用质量和可追溯性。",
                source_type=self.provider_name,
            )
        ]


def test_retrieval_comparison_runs_variants_and_aggregates_metrics(tmp_path):
    benchmark = tmp_path / "benchmark.json"
    benchmark.write_text(
        json.dumps(
            [
                {
                    "id": "sample_1",
                    "question": "如何提高引用质量？",
                    "expected_claims": ["引用质量"],
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    config = AppConfig(
        runtime=RuntimeConfig(run_dir=tmp_path / "runs", max_subtasks=1, enable_redteam=False),
        evaluation=EvaluationConfig(benchmark_path=benchmark),
    )
    runner = RetrievalComparisonRunner(
        config,
        provider_factory=lambda retrieval: StaticProvider(retrieval.provider),
    )

    variants = runner.run(["local", "tavily"])

    assert [item.provider for item in variants] == ["local", "tavily"]
    assert all(item.succeeded_count == 1 for item in variants)
    assert all(item.failed_count == 0 for item in variants)
    assert all(item.mean_citation_coverage == 1.0 for item in variants)
    assert all(item.mean_source_diversity == 1.0 for item in variants)
    assert runner.last_output_path is not None
    assert runner.last_output_path.parent == tmp_path / "runs" / "comparisons"
    assert runner.last_output_path.name.endswith("_retrieval_comparison.json")
    persisted = json.loads(runner.last_output_path.read_text(encoding="utf-8"))
    assert persisted["schema_version"] == "2.0"
    assert persisted["providers"] == ["local", "tavily"]
    assert persisted["memory_seed_path"] is None
    assert [item["provider"] for item in persisted["results"]] == ["local", "tavily"]


def test_retrieval_comparison_records_provider_initialization_failure(tmp_path):
    benchmark = tmp_path / "benchmark.json"
    benchmark.write_text(
        '[{"id":"sample_1","question":"test","expected_claims":[]}]',
        encoding="utf-8",
    )
    config = AppConfig(evaluation=EvaluationConfig(benchmark_path=benchmark))

    def failing_factory(retrieval):
        raise ValueError(f"missing credentials for {retrieval.provider}")

    result = RetrievalComparisonRunner(config, provider_factory=failing_factory).run(["tavily"])[0]

    assert result.succeeded_count == 0
    assert result.failed_count == 1
    assert result.failure_rate == 1.0
    assert result.failures[0].sample_id == "provider_initialization"


def test_retrieval_comparison_freezes_memory_before_memory_variant(tmp_path):
    benchmark = tmp_path / "benchmark.json"
    benchmark.write_text(
        '[{"id":"sample_1","question":"共享记忆","domain":"agent","expected_claims":[]}]',
        encoding="utf-8",
    )
    config = AppConfig(
        retrieval=RetrievalConfig(provider="hybrid", hybrid_providers=["local"]),
        runtime=RuntimeConfig(run_dir=tmp_path / "runs", max_subtasks=1),
        evaluation=EvaluationConfig(benchmark_path=benchmark),
    )
    runner = RetrievalComparisonRunner(config)

    runner.run(["hybrid+memory"])

    assert runner.memory_seed_path is not None and runner.memory_seed_path.exists()
    persisted = json.loads(runner.last_output_path.read_text(encoding="utf-8"))
    assert persisted["memory_seed_sha256"]

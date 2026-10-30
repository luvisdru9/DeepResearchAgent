import json

from deepresearch.config import AppConfig, EvaluationConfig, RetrievalConfig, RuntimeConfig
from deepresearch.evaluation import EvaluationRunner
from deepresearch.schema import EvalResult


def test_evaluation_runner_returns_results(tmp_path):
    config = AppConfig(
        retrieval=RetrievalConfig(corpus_dir="data/local_corpus", top_k=2),
        runtime=RuntimeConfig(run_dir=tmp_path, max_subtasks=2),
        evaluation=EvaluationConfig(benchmark_path="data/researchbench.json"),
    )
    runner = EvaluationRunner(config)
    results = runner.run()

    assert len(results) == 3
    assert all(result.run_id for result in results)
    assert runner.last_output_path is not None
    assert runner.last_output_path.exists()
    artifact = json.loads(runner.last_output_path.read_text(encoding="utf-8"))
    assert artifact["metric_protocol"] == "claim_citation_v2+replay_process_v1"
    assert artifact["judge_protocol"]["version"] == "4.0"
    assert artifact["judge_protocol"]["fingerprint"].startswith("sha256:")
    assert len(artifact["results"]) == 3


def test_frozen_evaluation_runner_ignores_online_provider_and_uses_snapshot(tmp_path, monkeypatch):
    """Frozen evaluation must remain runnable without a Tavily credential or network."""

    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    config = AppConfig(
        retrieval=RetrievalConfig(provider="tavily", top_k=5),
        runtime=RuntimeConfig(
            run_dir=tmp_path / "runs",
            max_subtasks=1,
            enable_redteam=False,
            enable_compression=False,
        ),
        evaluation=EvaluationConfig(
            benchmark_path="data/benchmarks/researchbench_v2/dataset.json"
        ),
    )

    results = EvaluationRunner(config).run()

    assert len(results) == 3
    replay_paths = sorted((tmp_path / "runs").glob("*_replay.json"))
    assert len(replay_paths) == 3
    for replay_path in replay_paths:
        replay = json.loads(replay_path.read_text(encoding="utf-8"))
        documents = [doc for docs in replay["docs_by_task"].values() for doc in docs]
        assert documents
        assert all(doc["metadata"]["provider"] == "frozen" for doc in documents)
        assert all(doc["metadata"]["snapshot_id"] for doc in documents)
    assert all(
        result.metadata["process_metrics"]["required_source_recall"] == 1.0
        for result in results
    )
    assert all(
        result.metadata["process_metrics"]["resources"]["latency_seconds"]
        == result.latency_seconds
        for result in results
    )


def test_automatic_consensus_artifact_is_not_labeled_as_human_reference(tmp_path):
    config = AppConfig(runtime=RuntimeConfig(run_dir=tmp_path))
    runner = EvaluationRunner(config)
    result = EvalResult(
        sample_id="sample_1",
        run_id="run_1",
        fact_accuracy=1.0,
        citation_coverage=1.0,
        citation_precision=1.0,
        hallucination_rate=0.0,
        answer_completeness=1.0,
        latency_seconds=0.1,
        metadata={
            "automatic_consensus": {
                "report_claims": {
                    "resolved_rate": 1.0,
                    "cross_model_agreement": 1.0,
                },
                "gold_claims": {
                    "resolved_rate": 1.0,
                    "cross_model_agreement": 1.0,
                },
                "gate": {"passed": True, "status": "pass", "failure_reasons": []},
            }
        },
    )

    runner._persist(tmp_path / "benchmark.json", "2.0", [result])
    artifact = json.loads(runner.last_output_path.read_text(encoding="utf-8"))

    assert artifact["schema_version"] == "1.1"
    assert artifact["evaluation_mode"] == "fully_automatic"
    assert artifact["reference_source"] == "multi_model_automatic_consensus"
    assert artifact["human_calibrated"] is False
    assert artifact["formal_benchmark_eligible"] is False
    assert artifact["automatic_consensus"]["b4_automatic_draft_allowed"] is True
    assert "human_reference" not in artifact
    assert "annotator_id" not in str(artifact)

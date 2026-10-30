"""Paired compression ablation over one reusable Evidence snapshot per sample."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from deepresearch.artifact_naming import build_compression_comparison_filename
from deepresearch.config import AppConfig
from deepresearch.evaluation.benchmark import BenchmarkSample, load_benchmark
from deepresearch.evaluation.metrics import score_report
from deepresearch.memory import EvidenceSnapshotStore
from deepresearch.orchestrator.pipeline import ResearchPipeline
from deepresearch.orchestrator.replay import ReplayStore
from deepresearch.schema import ResearchEvidenceSnapshot, ResearchTask


class CompressionSampleResult(BaseModel):
    """Quality, token, and context diagnostics for one fixed-snapshot trial."""

    sample_id: str
    snapshot_id: str
    run_id: str
    fact_accuracy: float
    citation_coverage: float
    citation_precision: float
    hallucination_rate: float
    latency_seconds: float
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    evidence_input_count: int = 0
    evidence_output_count: int = 0
    context_input_chars: int = 0
    context_output_chars: int = 0
    truncated: bool = False
    review_status: str = "skipped"


class CompressionVariantResult(BaseModel):
    """Aggregate one threshold while preserving per-sample paired results."""

    strategy: str
    compression_enabled: bool
    max_evidence: int | None = None
    sample_count: int
    succeeded_count: int
    failed_count: int
    mean_fact_accuracy: float
    mean_citation_coverage: float
    mean_hallucination_rate: float
    mean_latency_seconds: float
    mean_llm_input_tokens: float
    mean_llm_output_tokens: float
    mean_context_reduction_ratio: float
    results: list[CompressionSampleResult] = Field(default_factory=list)
    failures: list[dict[str, str]] = Field(default_factory=list)


class CompressionComparisonRunner:
    """Collect Evidence once, then run controlled synthesis at several compression thresholds."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.last_output_path: Path | None = None

    def run(
        self,
        benchmark_path: str | Path | None = None,
        *,
        provider: str | None = None,
        compressed_max_evidence: int | None = None,
        thresholds: list[int] | None = None,
    ) -> list[CompressionVariantResult]:
        path = Path(benchmark_path or self.config.evaluation.benchmark_path)
        samples = load_benchmark(path)
        active_thresholds = (
            [compressed_max_evidence]
            if compressed_max_evidence is not None
            else list(dict.fromkeys(thresholds or [4, 8, 12]))
        )
        if any(value < 1 for value in active_thresholds):
            raise ValueError("Compression thresholds must be at least 1.")
        snapshots, snapshot_paths, collection_failures = self._collect_snapshots(
            samples,
            provider=provider,
        )
        strategies: list[tuple[str, bool, int | None]] = [
            ("compression_off", False, None),
            *[(f"compression_{value}", True, value) for value in active_thresholds],
        ]
        variants = [
            self._run_variant(
                strategy,
                enabled,
                max_evidence,
                snapshots,
                samples,
                collection_failures,
                provider=provider,
            )
            for strategy, enabled, max_evidence in strategies
        ]
        created_at = datetime.now(timezone.utc)
        output_dir = self.config.runtime.run_dir / self.config.evaluation.comparison_subdir
        output_dir.mkdir(parents=True, exist_ok=True)
        self.last_output_path = output_dir / build_compression_comparison_filename(created_at)
        self.last_output_path.write_text(
            json.dumps(
                {
                    "schema_version": "2.0",
                    "created_at": created_at.isoformat(),
                    "benchmark_path": str(path),
                    "provider": provider or self.config.retrieval.provider,
                    "paired_evidence": True,
                    "thresholds": active_thresholds,
                    "snapshot_paths": snapshot_paths,
                    "results": [item.model_dump(mode="json") for item in variants],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return variants

    def _collect_snapshots(
        self,
        samples: list[BenchmarkSample],
        *,
        provider: str | None,
    ) -> tuple[dict[str, ResearchEvidenceSnapshot], dict[str, str], list[dict[str, str]]]:
        config = self.config.model_copy(deep=True)
        config.runtime.enable_compression = False
        config.memory.retrieval_enabled = False
        if provider:
            config.retrieval.provider = provider
        pipeline = ResearchPipeline(config)
        replay_store = ReplayStore(config.runtime.run_dir)
        snapshot_store = EvidenceSnapshotStore(config.runtime.run_dir)
        snapshots: dict[str, ResearchEvidenceSnapshot] = {}
        paths: dict[str, str] = {}
        failures: list[dict[str, str]] = []
        for sample in samples:
            sample_id = sample.id
            try:
                task = ResearchTask(
                    query=sample.question,
                    language=sample.language,
                    domain=sample.domain,
                    time_range=sample.time_range,
                )
                report = pipeline.run(task)
                replay_path = replay_store.path_for(report.run_id)
                replay = replay_store.read(report.run_id)
                snapshot = ResearchEvidenceSnapshot(
                    source_run_id=report.run_id,
                    task=task,
                    evidence=replay["evidence"],
                    source_replay_path=str(replay_path),
                )
                snapshot_path = snapshot_store.write(snapshot)
                snapshots[sample_id] = snapshot
                paths[sample_id] = str(snapshot_path)
            except Exception as exc:
                failures.append({"sample_id": sample_id, "error": f"{type(exc).__name__}: {exc}"})
        return snapshots, paths, failures

    def _run_variant(
        self,
        strategy: str,
        enabled: bool,
        max_evidence: int | None,
        snapshots: dict[str, ResearchEvidenceSnapshot],
        samples: list[BenchmarkSample],
        collection_failures: list[dict[str, str]],
        *,
        provider: str | None,
    ) -> CompressionVariantResult:
        config = self.config.model_copy(deep=True)
        config.memory.retrieval_enabled = False
        if provider:
            config.retrieval.provider = provider
        pipeline = ResearchPipeline(config)
        results: list[CompressionSampleResult] = []
        failures = list(collection_failures)
        for sample in samples:
            sample_id = sample.id
            snapshot = snapshots.get(sample_id)
            if snapshot is None:
                continue
            try:
                source_task = snapshot.task
                task = ResearchTask(
                    query=source_task.query,
                    language=source_task.language,
                    domain=source_task.domain,
                    time_range=source_task.time_range,
                    report_length=source_task.report_length,
                    require_citations=source_task.require_citations,
                    max_budget_usd=source_task.max_budget_usd,
                )
                trial = pipeline.synthesize_snapshot(
                    task,
                    snapshot.evidence,
                    compression_enabled=enabled,
                    max_evidence=max_evidence,
                )
                evaluation = score_report(
                    sample_id,
                    trial.report,
                    sample.expected_claims,
                    trial.latency_seconds,
                )
                input_tokens = sum(int(call.get("input_tokens") or 0) for call in trial.llm_calls)
                output_tokens = sum(int(call.get("output_tokens") or 0) for call in trial.llm_calls)
                results.append(
                    CompressionSampleResult(
                        sample_id=sample_id,
                        snapshot_id=snapshot.snapshot_id,
                        run_id=task.run_id,
                        fact_accuracy=evaluation.fact_accuracy,
                        citation_coverage=evaluation.citation_coverage,
                        citation_precision=evaluation.citation_precision,
                        hallucination_rate=evaluation.hallucination_rate,
                        latency_seconds=trial.latency_seconds,
                        llm_input_tokens=input_tokens,
                        llm_output_tokens=output_tokens,
                        evidence_input_count=len(snapshot.evidence),
                        evidence_output_count=len(trial.context_evidence),
                        context_input_chars=trial.context_input_chars,
                        context_output_chars=trial.context_output_chars,
                        truncated=trial.truncated,
                        review_status=trial.review_status,
                    )
                )
            except Exception as exc:
                failures.append({"sample_id": sample_id, "error": f"{type(exc).__name__}: {exc}"})
        return self._summarize(
            strategy,
            enabled,
            max_evidence,
            len(samples),
            results,
            failures,
        )

    @staticmethod
    def _summarize(
        strategy: str,
        enabled: bool,
        max_evidence: int | None,
        sample_count: int,
        results: list[CompressionSampleResult],
        failures: list[dict[str, str]],
    ) -> CompressionVariantResult:
        def mean(values: list[float]) -> float:
            return sum(values) / len(values) if values else 0.0

        reductions = [
            1.0 - item.context_output_chars / item.context_input_chars
            for item in results
            if item.context_input_chars
        ]
        return CompressionVariantResult(
            strategy=strategy,
            compression_enabled=enabled,
            max_evidence=max_evidence,
            sample_count=sample_count,
            succeeded_count=len(results),
            failed_count=len(failures),
            mean_fact_accuracy=mean([item.fact_accuracy for item in results]),
            mean_citation_coverage=mean([item.citation_coverage for item in results]),
            mean_hallucination_rate=mean([item.hallucination_rate for item in results]),
            mean_latency_seconds=mean([item.latency_seconds for item in results]),
            mean_llm_input_tokens=mean([float(item.llm_input_tokens) for item in results]),
            mean_llm_output_tokens=mean([float(item.llm_output_tokens) for item in results]),
            mean_context_reduction_ratio=mean(reductions),
            results=results,
            failures=failures,
        )

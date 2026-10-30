"""Ablation matrix for the five system configurations defined in PLAN.md."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from deepresearch.artifact_naming import (
    build_memory_seed_filename,
    build_system_comparison_filename,
)
from deepresearch.config import AppConfig
from deepresearch.evaluation.benchmark import (
    AtomicClaim,
    BenchmarkSample,
    BenchmarkSampleV2,
    load_benchmark_bundle,
    resolve_snapshot_manifest_path,
)
from deepresearch.evaluation.process_metrics import score_research_process
from deepresearch.evaluation.service import ReportEvaluationService
from deepresearch.memory import SQLiteMemoryStore
from deepresearch.orchestrator.pipeline import ResearchPipeline
from deepresearch.orchestrator.replay import ReplayStore
from deepresearch.schema import EvalResult, ResearchTask


SYSTEM_STRATEGIES = (
    "baseline_linear",
    "dag_multi_agent",
    "dag_memory",
    "dag_memory_redteam",
    "dag_memory_redteam_compress",
)


class SystemComparisonFailure(BaseModel):
    sample_id: str
    error: str


class SystemSampleResult(BaseModel):
    sample_id: str
    run_id: str
    fact_accuracy: float
    citation_coverage: float
    citation_precision: float
    hallucination_rate: float
    answer_completeness: float
    latency_seconds: float
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    cost_usd: float = 0.0
    evaluator: str | None = None
    semantic_judge_enabled: bool = False
    completion_status: str | None = None
    quality_status: str | None = None
    fallback_count: int = 0
    fallback_breakdown: dict[str, int] = Field(default_factory=dict)
    empty_result_rule_assist_count: int = 0
    evidence_count: int = 0
    source_count: int = 0
    independent_domain_count: int = 0
    required_source_recall: float | None = None
    llm_calls: int = 0


class SystemVariantResult(BaseModel):
    strategy: str
    feature_flags: dict[str, bool | int | str]
    sample_count: int
    succeeded_count: int
    failed_count: int
    failure_rate: float
    mean_fact_accuracy: float
    mean_citation_coverage: float
    mean_citation_precision: float
    mean_hallucination_rate: float
    mean_answer_completeness: float
    mean_latency_seconds: float
    mean_llm_input_tokens: float
    mean_llm_output_tokens: float
    total_cost_usd: float
    results: list[SystemSampleResult] = Field(default_factory=list)
    failures: list[SystemComparisonFailure] = Field(default_factory=list)


class SystemPairedDelta(BaseModel):
    """Within-sample full-system minus baseline measurements; not causal estimates."""

    sample_id: str
    baseline_run_id: str
    full_run_id: str
    fact_accuracy_delta: float
    citation_coverage_delta: float
    citation_precision_delta: float
    hallucination_rate_delta: float
    latency_seconds_delta: float
    llm_calls_delta: int
    total_tokens_delta: int


class SystemComparisonRunner:
    """Run controlled feature additions from linear baseline to the full system."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.last_output_path: Path | None = None
        self.last_markdown_path: Path | None = None
        self.memory_seed_path: Path | None = None
        self.memory_seed_sha256: str | None = None

    def run(
        self,
        benchmark_path: str | Path | None = None,
        *,
        strategies: list[str] | None = None,
        judge_strategies: list[str] | None = None,
        memory_seed_mode: Literal["current", "empty"] = "current",
        max_llm_calls: int | None = None,
        max_total_tokens: int | None = None,
        max_runtime_seconds: float | None = None,
        sample_ids: list[str] | None = None,
    ) -> list[SystemVariantResult]:
        selected = list(dict.fromkeys(strategies or list(SYSTEM_STRATEGIES)))
        unknown = sorted(set(selected) - set(SYSTEM_STRATEGIES))
        if unknown:
            raise ValueError(f"Unknown system strategies: {unknown}")
        selected_judges = list(dict.fromkeys(judge_strategies or selected))
        unknown_judges = sorted(set(selected_judges) - set(selected))
        if unknown_judges:
            raise ValueError(
                "Judge strategies must be included in --strategies: "
                f"{unknown_judges}"
            )
        if memory_seed_mode not in {"current", "empty"}:
            raise ValueError("memory_seed_mode must be 'current' or 'empty'.")
        path = Path(benchmark_path or self.config.evaluation.benchmark_path)
        bundle = load_benchmark_bundle(path)
        samples = bundle.samples
        selected_sample_ids = list(dict.fromkeys(sample_ids or [item.id for item in samples]))
        available_sample_ids = {item.id for item in samples}
        unknown_sample_ids = sorted(set(selected_sample_ids) - available_sample_ids)
        if unknown_sample_ids:
            raise ValueError(f"Unknown benchmark sample ids: {unknown_sample_ids}")
        by_sample_id = {item.id: item for item in samples}
        samples = [by_sample_id[sample_id] for sample_id in selected_sample_ids]
        authored_samples = {
            sample.id: sample
            for sample in (bundle.dataset_v2.samples if bundle.dataset_v2 else [])
        }
        created_at = datetime.now(timezone.utc)
        self._freeze_memory_seed(created_at, selected, mode=memory_seed_mode)
        results = [
            self._run_variant(
                name,
                samples,
                benchmark_path=path,
                authored_samples=authored_samples,
                semantic_judge=name in selected_judges,
                max_llm_calls=max_llm_calls,
                max_total_tokens=max_total_tokens,
                max_runtime_seconds=max_runtime_seconds,
            )
            for name in selected
        ]
        self._persist(
            created_at,
            path,
            selected,
            results,
            judge_strategies=selected_judges,
            memory_seed_mode=memory_seed_mode,
            task_budgets={
                "max_llm_calls": max_llm_calls,
                "max_total_tokens": max_total_tokens,
                "max_runtime_seconds": max_runtime_seconds,
            },
            sample_ids=selected_sample_ids,
        )
        return results

    def _freeze_memory_seed(
        self,
        created_at: datetime,
        strategies: list[str],
        *,
        mode: Literal["current", "empty"],
    ) -> None:
        self.memory_seed_path = None
        self.memory_seed_sha256 = None
        if not any("memory" in strategy for strategy in strategies):
            return
        output_dir = self.config.runtime.run_dir / self.config.evaluation.comparison_subdir
        output_dir.mkdir(parents=True, exist_ok=True)
        source_path = self.config.memory.sqlite_path or (
            self.config.runtime.run_dir / "deepresearch.sqlite3"
        )
        target = output_dir / build_memory_seed_filename(created_at)
        if mode == "empty":
            # 初始化全新 schema 后固定字节；每个 memory variant 都从该空白 seed 复制。
            SQLiteMemoryStore(target)
        else:
            SQLiteMemoryStore(source_path).backup_to(target)
        self.memory_seed_path = target
        self.memory_seed_sha256 = hashlib.sha256(target.read_bytes()).hexdigest()

    def _run_variant(
        self,
        strategy: str,
        samples: list[BenchmarkSample],
        *,
        benchmark_path: Path,
        authored_samples: dict[str, BenchmarkSampleV2],
        semantic_judge: bool,
        max_llm_calls: int | None,
        max_total_tokens: int | None,
        max_runtime_seconds: float | None,
    ) -> SystemVariantResult:
        config = self._strategy_config(strategy)
        results: list[SystemSampleResult] = []
        failures: list[SystemComparisonFailure] = []
        for sample in samples:
            try:
                sample_config = self._sample_config(
                    config,
                    benchmark_path=benchmark_path,
                    authored_sample=authored_samples.get(sample.id),
                )
                if sample_config.memory.retrieval_enabled:
                    if self.memory_seed_path is None:
                        raise RuntimeError("Memory strategy requires a frozen memory seed.")
                    with tempfile.TemporaryDirectory(prefix="deepresearch-system-memory-") as temp:
                        isolated_config = sample_config.model_copy(deep=True)
                        isolated_store = Path(temp) / "memory.sqlite3"
                        shutil.copy2(self.memory_seed_path, isolated_store)
                        isolated_config.memory.sqlite_path = isolated_store
                        results.append(
                            self._run_sample(
                                isolated_config,
                                sample,
                                authored_sample=authored_samples.get(sample.id),
                                semantic_judge=semantic_judge,
                                max_llm_calls=max_llm_calls,
                                max_total_tokens=max_total_tokens,
                                max_runtime_seconds=max_runtime_seconds,
                            )
                        )
                else:
                    results.append(
                        self._run_sample(
                            sample_config,
                            sample,
                            authored_sample=authored_samples.get(sample.id),
                            semantic_judge=semantic_judge,
                            max_llm_calls=max_llm_calls,
                            max_total_tokens=max_total_tokens,
                            max_runtime_seconds=max_runtime_seconds,
                        )
                    )
            except Exception as exc:
                failures.append(
                    SystemComparisonFailure(
                        sample_id=sample.id,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
        summary = self._summarize(strategy, config, len(samples), results, failures)
        if authored_samples and all(
            authored_samples.get(sample.id) is not None
            and authored_samples[sample.id].snapshot is not None
            for sample in samples
        ):
            summary.feature_flags["retrieval_provider"] = "frozen_per_sample"
            summary.feature_flags["frozen_snapshot_mode"] = True
        return summary

    @staticmethod
    def _sample_config(
        config: AppConfig,
        *,
        benchmark_path: Path,
        authored_sample: BenchmarkSampleV2 | None,
    ) -> AppConfig:
        """Pin v2 samples to their immutable source universe for every strategy."""

        if authored_sample is None or authored_sample.snapshot is None:
            return config
        manifest_path = resolve_snapshot_manifest_path(
            benchmark_path,
            authored_sample.snapshot,
        )
        sample_config = config.model_copy(deep=True)
        sample_config.retrieval.provider = "frozen"
        sample_config.retrieval.frozen_manifest_path = manifest_path
        return sample_config

    def _strategy_config(self, strategy: str) -> AppConfig:
        config = self.config.model_copy(deep=True)
        uses_memory = "memory" in strategy
        uses_redteam = "redteam" in strategy
        uses_compression = strategy.endswith("compress")
        config.memory.enabled = uses_memory
        config.memory.retrieval_enabled = uses_memory
        config.runtime.enable_redteam = uses_redteam
        config.runtime.enable_compression = uses_compression
        if strategy == "baseline_linear":
            # 在统一 Pipeline 内构造单子任务、单并发的线性近似基线。
            config.planning.provider = "rule"
            config.runtime.max_subtasks = 1
            config.runtime.max_concurrency = 1
        if uses_memory and config.retrieval.provider != "hybrid":
            config.retrieval.hybrid_providers = [config.retrieval.provider]
            config.retrieval.provider = "hybrid"
        return config

    @staticmethod
    def _run_sample(
        config: AppConfig,
        sample: BenchmarkSample,
        *,
        authored_sample: BenchmarkSampleV2 | None,
        semantic_judge: bool,
        max_llm_calls: int | None,
        max_total_tokens: int | None,
        max_runtime_seconds: float | None,
    ) -> SystemSampleResult:
        started = time.perf_counter()
        task = ResearchTask(
            query=sample.question,
            language=sample.language,
            domain=sample.domain,
            time_range=sample.time_range,
            max_llm_calls=max_llm_calls,
            max_total_tokens=max_total_tokens,
            max_runtime_seconds=max_runtime_seconds,
        )
        report = ResearchPipeline(config).run(task)
        pipeline_elapsed = time.perf_counter() - started
        evaluation_config = config
        if not semantic_judge:
            evaluation_config = config.model_copy(deep=True)
            evaluation_config.evaluation.provider = "rule"
            evaluation_config.evaluation.judge_profiles = []
        required_source_ids = [
            source_id
            for requirement in (authored_sample.evidence_requirements if authored_sample else [])
            for source_id in requirement.required_source_ids
        ]
        replay = ReplayStore(config.runtime.run_dir).read(report.run_id)
        process_metrics = score_research_process(
            replay,
            required_source_ids=required_source_ids,
        )
        evaluation = ReportEvaluationService(evaluation_config).evaluate(
            sample.id,
            task,
            report,
            sample.expected_claims,
            pipeline_elapsed,
            process_metrics,
            gold_claims=(authored_sample.claims if authored_sample else None),
        )
        calls = replay.get("llm_calls", [])
        input_tokens = sum(int(call.get("input_tokens") or 0) for call in calls)
        output_tokens = sum(int(call.get("output_tokens") or 0) for call in calls)
        pipeline_cost = sum(float(call.get("cost_usd") or 0.0) for call in calls)
        quality_cycle = replay.get("quality_cycle") or {}
        resources = process_metrics["resources"]
        judge_calls = int(evaluation.metadata.get("judge_count") or 0)
        return SystemSampleResult(
            sample_id=sample.id,
            run_id=report.run_id,
            fact_accuracy=evaluation.fact_accuracy,
            citation_coverage=evaluation.citation_coverage,
            citation_precision=evaluation.citation_precision,
            hallucination_rate=evaluation.hallucination_rate,
            answer_completeness=evaluation.answer_completeness,
            latency_seconds=time.perf_counter() - started,
            llm_input_tokens=input_tokens,
            llm_output_tokens=output_tokens,
            cost_usd=pipeline_cost + evaluation.cost_usd,
            evaluator=str(evaluation.metadata.get("evaluator")),
            semantic_judge_enabled=semantic_judge,
            completion_status=quality_cycle.get("completion_status"),
            quality_status=quality_cycle.get("effective_final_status"),
            fallback_count=int(process_metrics["fallback_count"]),
            fallback_breakdown=dict(process_metrics["fallback_breakdown"]),
            empty_result_rule_assist_count=int(
                process_metrics["empty_result_rule_assist_count"]
            ),
            evidence_count=int(process_metrics["evidence_count"]),
            source_count=int(process_metrics["document_count"]),
            independent_domain_count=int(process_metrics["independent_domain_count"]),
            required_source_recall=process_metrics["required_source_recall"],
            llm_calls=int(resources["llm_calls"]) + judge_calls,
        )

    @staticmethod
    def _summarize(
        strategy: str,
        config: AppConfig,
        sample_count: int,
        results: list[SystemSampleResult],
        failures: list[SystemComparisonFailure],
    ) -> SystemVariantResult:
        def average(name: str) -> float:
            return (
                sum(float(getattr(item, name)) for item in results) / len(results)
                if results
                else 0.0
            )

        return SystemVariantResult(
            strategy=strategy,
            feature_flags={
                "max_subtasks": config.runtime.max_subtasks,
                "max_concurrency": config.runtime.max_concurrency,
                "memory": config.memory.retrieval_enabled,
                "redteam": config.runtime.enable_redteam,
                "compression": config.runtime.enable_compression,
                "retrieval_provider": config.retrieval.provider,
            },
            sample_count=sample_count,
            succeeded_count=len(results),
            failed_count=len(failures),
            failure_rate=len(failures) / sample_count if sample_count else 0.0,
            mean_fact_accuracy=average("fact_accuracy"),
            mean_citation_coverage=average("citation_coverage"),
            mean_citation_precision=average("citation_precision"),
            mean_hallucination_rate=average("hallucination_rate"),
            mean_answer_completeness=average("answer_completeness"),
            mean_latency_seconds=average("latency_seconds"),
            mean_llm_input_tokens=average("llm_input_tokens"),
            mean_llm_output_tokens=average("llm_output_tokens"),
            total_cost_usd=sum(item.cost_usd for item in results),
            results=results,
            failures=failures,
        )

    def _persist(
        self,
        created_at: datetime,
        benchmark_path: Path,
        strategies: list[str],
        results: list[SystemVariantResult],
        *,
        judge_strategies: list[str],
        memory_seed_mode: str,
        task_budgets: dict[str, int | float | None],
        sample_ids: list[str],
    ) -> None:
        output_dir = self.config.runtime.run_dir / self.config.evaluation.comparison_subdir
        output_dir.mkdir(parents=True, exist_ok=True)
        self.last_output_path = output_dir / build_system_comparison_filename(created_at)
        payload = {
            "schema_version": "1.1",
            "created_at": created_at.isoformat(),
            "benchmark_path": str(benchmark_path),
            "strategies": strategies,
            "judge_strategies": judge_strategies,
            "memory_seed_mode": memory_seed_mode,
            "task_budgets": task_budgets,
            "sample_ids": sample_ids,
            "memory_seed_path": str(self.memory_seed_path) if self.memory_seed_path else None,
            "memory_seed_sha256": self.memory_seed_sha256,
            "results": [item.model_dump(mode="json") for item in results],
            "paired_deltas": [
                item.model_dump(mode="json") for item in self._paired_deltas(results)
            ],
        }
        self.last_output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self.last_markdown_path = output_dir / build_system_comparison_filename(
            created_at, markdown=True
        )
        self.last_markdown_path.write_text(self._markdown_table(results), encoding="utf-8")

    @staticmethod
    def _paired_deltas(results: list[SystemVariantResult]) -> list[SystemPairedDelta]:
        """Compare only matching baseline/full samples and preserve missing runs as failures."""

        by_strategy = {item.strategy: item for item in results}
        baseline = by_strategy.get("baseline_linear")
        full = by_strategy.get("dag_memory_redteam_compress")
        if baseline is None or full is None:
            return []
        baseline_by_sample = {item.sample_id: item for item in baseline.results}
        full_by_sample = {item.sample_id: item for item in full.results}
        paired: list[SystemPairedDelta] = []
        for sample_id in sorted(set(baseline_by_sample).intersection(full_by_sample)):
            left = baseline_by_sample[sample_id]
            right = full_by_sample[sample_id]
            paired.append(
                SystemPairedDelta(
                    sample_id=sample_id,
                    baseline_run_id=left.run_id,
                    full_run_id=right.run_id,
                    fact_accuracy_delta=right.fact_accuracy - left.fact_accuracy,
                    citation_coverage_delta=right.citation_coverage - left.citation_coverage,
                    citation_precision_delta=right.citation_precision - left.citation_precision,
                    hallucination_rate_delta=right.hallucination_rate - left.hallucination_rate,
                    latency_seconds_delta=right.latency_seconds - left.latency_seconds,
                    llm_calls_delta=right.llm_calls - left.llm_calls,
                    total_tokens_delta=(
                        right.llm_input_tokens
                        + right.llm_output_tokens
                        - left.llm_input_tokens
                        - left.llm_output_tokens
                    ),
                )
            )
        return paired

    @staticmethod
    def _markdown_table(results: list[SystemVariantResult]) -> str:
        lines = [
            "# DeepResearch System Comparison",
            "",
            "| Strategy | Success | Fact | Citation coverage | Citation precision | Hallucination | Fallback | LLM calls | Latency (s) | Input tokens | Output tokens |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for item in results:
            lines.append(
                f"| {item.strategy} | {item.succeeded_count}/{item.sample_count} | "
                f"{item.mean_fact_accuracy:.3f} | {item.mean_citation_coverage:.3f} | "
                f"{item.mean_citation_precision:.3f} | {item.mean_hallucination_rate:.3f} | "
                f"{sum(row.fallback_count for row in item.results)} | "
                f"{sum(row.llm_calls for row in item.results)} | "
                f"{item.mean_latency_seconds:.2f} | {item.mean_llm_input_tokens:.1f} | "
                f"{item.mean_llm_output_tokens:.1f} |"
            )
        return "\n".join(lines) + "\n"

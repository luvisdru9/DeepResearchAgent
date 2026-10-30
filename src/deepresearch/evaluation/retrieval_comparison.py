"""Run the same benchmark across retrieval providers and aggregate comparable metrics."""

from __future__ import annotations

import json
import hashlib
import shutil
import tempfile
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from deepresearch.config import AppConfig, RetrievalConfig
from deepresearch.artifact_naming import (
    build_memory_seed_filename,
    build_retrieval_comparison_filename,
)
from deepresearch.evaluation.benchmark import BenchmarkSample, load_benchmark
from deepresearch.evaluation.service import ReportEvaluationService
from deepresearch.memory import SQLiteMemoryStore
from deepresearch.orchestrator.pipeline import ResearchPipeline
from deepresearch.retrieval import SearchProvider, create_search_provider
from deepresearch.schema import EvalResult, ResearchTask

ProviderFactory = Callable[[RetrievalConfig], SearchProvider]


class RetrievalComparisonFailure(BaseModel):
    """One provider initialization or benchmark-sample failure."""

    sample_id: str
    error: str


class RetrievalVariantResult(BaseModel):
    """Per-provider results plus aggregate metrics for experiment tables."""

    provider: str
    sample_count: int
    succeeded_count: int
    failed_count: int
    failure_rate: float
    mean_fact_accuracy: float
    mean_citation_coverage: float
    mean_citation_precision: float
    mean_hallucination_rate: float
    mean_source_diversity: float
    mean_latency_seconds: float
    results: list[EvalResult] = Field(default_factory=list)
    failures: list[RetrievalComparisonFailure] = Field(default_factory=list)


class RetrievalComparisonRunner:
    """Compare local, Tavily, or future hybrid providers under one pipeline configuration."""

    def __init__(
        self,
        config: AppConfig,
        *,
        provider_factory: ProviderFactory = create_search_provider,
    ) -> None:
        self.config = config
        self.provider_factory = provider_factory
        self.last_output_path: Path | None = None
        self.memory_seed_path: Path | None = None
        self.memory_seed_sha256: str | None = None

    def run(
        self,
        providers: list[str],
        benchmark_path: str | Path | None = None,
    ) -> list[RetrievalVariantResult]:
        """Run providers independently so one unavailable backend does not erase other results."""

        path = Path(benchmark_path or self.config.evaluation.benchmark_path)
        samples = load_benchmark(path)
        created_at = datetime.now(timezone.utc)
        self._freeze_memory_seed(created_at, providers)
        results = [self._run_variant(provider, samples) for provider in dict.fromkeys(providers)]
        self.last_output_path = self._persist_results(
            created_at=created_at,
            benchmark_path=path,
            providers=providers,
            results=results,
        )
        return results

    def _persist_results(
        self,
        *,
        created_at: datetime,
        benchmark_path: Path,
        providers: list[str],
        results: list[RetrievalVariantResult],
    ) -> Path:
        """Persist comparison metadata and results without changing the public run() return type."""

        output_dir = self.config.runtime.run_dir / self.config.evaluation.comparison_subdir
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / build_retrieval_comparison_filename(created_at)
        payload = {
            "schema_version": "2.0",
            "created_at": created_at.isoformat(),
            "benchmark_path": str(benchmark_path),
            "providers": list(dict.fromkeys(providers)),
            "memory_seed_path": str(self.memory_seed_path) if self.memory_seed_path else None,
            "memory_seed_sha256": self.memory_seed_sha256,
            "results": [item.model_dump(mode="json") for item in results],
        }
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return output_path

    def _freeze_memory_seed(self, created_at: datetime, providers: list[str]) -> None:
        """Freeze pre-experiment memory so variants and samples cannot contaminate each other."""

        self.memory_seed_path = None
        self.memory_seed_sha256 = None
        if "hybrid+memory" not in providers:
            return
        output_dir = self.config.runtime.run_dir / self.config.evaluation.comparison_subdir
        output_dir.mkdir(parents=True, exist_ok=True)
        source_path = self.config.memory.sqlite_path or (
            self.config.runtime.run_dir / "deepresearch.sqlite3"
        )
        source = SQLiteMemoryStore(source_path)
        target = output_dir / build_memory_seed_filename(created_at)
        source.backup_to(target)
        self.memory_seed_path = target
        self.memory_seed_sha256 = hashlib.sha256(target.read_bytes()).hexdigest()

    def _run_variant(
        self, provider_name: str, samples: list[BenchmarkSample]
    ) -> RetrievalVariantResult:
        variant_config = self.config.model_copy(deep=True)
        actual_provider = "hybrid" if provider_name == "hybrid+memory" else provider_name
        variant_config.retrieval.provider = actual_provider
        if provider_name == "hybrid+memory":
            variant_config.memory.enabled = True
            variant_config.memory.retrieval_enabled = True
        elif not variant_config.evaluation.retrieval_comparison_use_memory:
            # 避免按顺序运行的后一个 variant 使用前一个 variant 新写入的历史 Evidence。
            variant_config.memory.retrieval_enabled = False
        failures: list[RetrievalComparisonFailure] = []
        results: list[EvalResult] = []

        try:
            # 预检 provider，保持凭据缺失时的单一初始化错误语义。
            self.provider_factory(variant_config.retrieval)
        except Exception as exc:  # Provider 缺密钥时仍返回可比较的失败摘要。
            failures.append(
                RetrievalComparisonFailure(
                    sample_id="provider_initialization",
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            return self._summarize(provider_name, len(samples), results, failures, len(samples))

        for sample in samples:
            sample_id = sample.id
            try:
                if provider_name == "hybrid+memory":
                    if self.memory_seed_path is None:
                        raise RuntimeError("hybrid+memory requires a frozen memory seed.")
                    with tempfile.TemporaryDirectory(prefix="deepresearch-memory-") as temp_dir:
                        sample_config = variant_config.model_copy(deep=True)
                        isolated_store = Path(temp_dir) / "memory.sqlite3"
                        shutil.copy2(self.memory_seed_path, isolated_store)
                        sample_config.memory.sqlite_path = isolated_store
                        evaluation = self._run_sample(sample_config, sample_id, sample)
                else:
                    evaluation = self._run_sample(variant_config, sample_id, sample)
                results.append(
                    evaluation.model_copy(
                        update={"latency_seconds": evaluation.latency_seconds}
                    )
                )
            except Exception as exc:  # 一个样本失败不应中断其他 provider/样本。
                failures.append(
                    RetrievalComparisonFailure(
                        sample_id=sample_id,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
        return self._summarize(
            provider_name,
            len(samples),
            results,
            failures,
            len(failures),
        )

    def _run_sample(
        self,
        config: AppConfig,
        sample_id: str,
        sample: BenchmarkSample,
    ) -> EvalResult:
        """Run one sample with fresh runtime objects and its assigned Memory snapshot."""

        provider = self.provider_factory(config.retrieval)
        pipeline = ResearchPipeline(config, search_provider=provider)
        evaluator = ReportEvaluationService(config)
        started = time.perf_counter()
        task = ResearchTask(
            query=sample.question,
            language=sample.language,
            domain=sample.domain,
            time_range=sample.time_range,
        )
        report = pipeline.run(task)
        pipeline_elapsed = time.perf_counter() - started
        result = evaluator.evaluate(
            sample_id,
            task,
            report,
            sample.expected_claims,
            pipeline_elapsed,
        )
        return result.model_copy(update={"latency_seconds": time.perf_counter() - started})

    @staticmethod
    def _summarize(
        provider: str,
        sample_count: int,
        results: list[EvalResult],
        failures: list[RetrievalComparisonFailure],
        failed_count: int,
    ) -> RetrievalVariantResult:
        def mean(values: list[float]) -> float:
            return sum(values) / len(values) if values else 0.0

        return RetrievalVariantResult(
            provider=provider,
            sample_count=sample_count,
            succeeded_count=len(results),
            failed_count=failed_count,
            failure_rate=failed_count / sample_count if sample_count else 0.0,
            mean_fact_accuracy=mean([item.fact_accuracy for item in results]),
            mean_citation_coverage=mean([item.citation_coverage for item in results]),
            mean_citation_precision=mean([item.citation_precision for item in results]),
            mean_hallucination_rate=mean([item.hallucination_rate for item in results]),
            mean_source_diversity=mean(
                [float(item.metadata.get("source_diversity", 0)) for item in results]
            ),
            mean_latency_seconds=mean([item.latency_seconds for item in results]),
            results=results,
            failures=failures,
        )

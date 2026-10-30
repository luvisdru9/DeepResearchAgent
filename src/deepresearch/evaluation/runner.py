from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from deepresearch.artifact_naming import build_benchmark_evaluation_filename
from deepresearch.config import AppConfig
from deepresearch.evaluation.benchmark import (
    BenchmarkSample,
    load_benchmark_bundle,
    resolve_snapshot_manifest_path,
)
from deepresearch.evaluation.service import (
    ReportEvaluationService,
    summarize_automatic_consensus,
)
from deepresearch.evaluation.process_metrics import score_research_process
from deepresearch.orchestrator.pipeline import ResearchPipeline
from deepresearch.orchestrator.replay import ReplayStore
from deepresearch.schema import EvalResult, ResearchTask


class EvaluationRunner:
    """Runs the pipeline against the local ResearchBench smoke dataset."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        # 延迟创建普通 Pipeline；冻结数据集不应先初始化 Tavily 等在线 provider。
        self.pipeline: ResearchPipeline | None = None
        self.evaluator = ReportEvaluationService(config)
        self.last_output_path: Path | None = None

    def run(self, benchmark_path: str | Path | None = None) -> list[EvalResult]:
        """Evaluate every benchmark sample and return structured metrics."""

        path = Path(benchmark_path or self.config.evaluation.benchmark_path)
        bundle = load_benchmark_bundle(path)
        v2_samples = {
            sample.id: sample
            for sample in (bundle.dataset_v2.samples if bundle.dataset_v2 else [])
        }
        results: list[EvalResult] = []
        for sample in bundle.samples:
            # 每个样本独立创建 ResearchTask，run_id 会用于关联 replay 和评测结果。
            started = time.perf_counter()
            task = build_generation_task(sample)
            pipeline = self.pipeline
            if bundle.dataset_v2 and bundle.dataset_v2.metadata.track == "frozen":
                authored_sample = v2_samples[sample.id]
                assert authored_sample.snapshot is not None
                manifest_path = resolve_snapshot_manifest_path(path, authored_sample.snapshot)
                # 冻结轨道同时关闭历史 Evidence 检索，避免前序运行污染固定来源集合。
                frozen_config = self.config.model_copy(
                    update={
                        "retrieval": self.config.retrieval.model_copy(
                            update={
                                "provider": "frozen",
                                "frozen_manifest_path": manifest_path,
                            }
                        ),
                        "memory": self.config.memory.model_copy(
                            update={"retrieval_enabled": False}
                        ),
                    }
                )
                pipeline = ResearchPipeline(frozen_config)
            elif pipeline is None:
                pipeline = ResearchPipeline(self.config)
                self.pipeline = pipeline
            assert pipeline is not None
            report = pipeline.run(task)
            elapsed = time.perf_counter() - started
            required_source_ids = (
                [
                    source_id
                    for requirement in v2_samples[sample.id].evidence_requirements
                    for source_id in requirement.required_source_ids
                ]
                if sample.id in v2_samples
                else []
            )
            replay = ReplayStore(self.config.runtime.run_dir).read(report.run_id)
            process_metrics = score_research_process(
                replay,
                required_source_ids=required_source_ids,
            )
            results.append(
                self.evaluator.evaluate(
                    sample.id,
                    task,
                    report,
                    sample.expected_claims,
                    elapsed,
                    process_metrics,
                    gold_claims=(v2_samples[sample.id].claims if sample.id in v2_samples else None),
                )
            )
        self._persist(path, bundle.schema_version, results)
        return results

    def _persist(
        self,
        benchmark_path: Path,
        benchmark_schema_version: str,
        results: list[EvalResult],
    ) -> None:
        """Persist the exact per-sample metrics so experiment records can reference them."""

        created_at = datetime.now(timezone.utc)
        output_dir = self.config.runtime.run_dir / "benchmarks"
        output_dir.mkdir(parents=True, exist_ok=True)
        self.last_output_path = output_dir / build_benchmark_evaluation_filename(created_at)
        automatic_consensus = summarize_automatic_consensus(results)
        payload = {
            "schema_version": "1.1" if automatic_consensus is not None else "1.0",
            "metric_protocol": "claim_citation_v2+replay_process_v1",
            "created_at": created_at.isoformat(),
            "benchmark_path": str(benchmark_path),
            "benchmark_schema_version": benchmark_schema_version,
            "judge_protocol": self.evaluator.protocol_manifest().model_dump(mode="json"),
            "results": [item.model_dump(mode="json") for item in results],
        }
        if automatic_consensus is not None:
            # 自动共识与人工标签使用不同字段，避免把模型互评误写为人工校准。
            payload.update(
                {
                    "evaluation_mode": "fully_automatic",
                    "reference_source": automatic_consensus["reference_source"],
                    "independent_model_versions": automatic_consensus[
                        "independent_model_versions"
                    ],
                    "release_tier": "engineering_automatic",
                    "human_calibrated": False,
                    "formal_benchmark_eligible": False,
                    "automatic_consensus": automatic_consensus,
                }
            )
        self.last_output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def build_generation_task(sample: BenchmarkSample) -> ResearchTask:
    """Expose only user-visible inputs to generation; gold claims/rubrics stay evaluator-side."""

    return ResearchTask(
        query=sample.question,
        language=sample.language,
        domain=sample.domain,
        time_range=sample.time_range,
    )

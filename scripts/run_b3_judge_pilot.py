"""Run only the B3 LLM Judge over reports from an existing benchmark artifact."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from deepresearch.artifact_naming import build_benchmark_evaluation_filename
from deepresearch.config import load_config
from deepresearch.evaluation.benchmark import load_benchmark_bundle
from deepresearch.evaluation.runner import build_generation_task
from deepresearch.evaluation.service import ReportEvaluationService
from deepresearch.orchestrator.replay import ReplayStore
from deepresearch.schema import ResearchReport


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--base-results", required=True)
    parser.add_argument("--output-dir", default="runs/benchmarks")
    args = parser.parse_args()

    config = load_config(args.config)
    if config.evaluation.provider != "llm":
        raise ValueError("B3 Judge pilot requires evaluation.provider=llm.")
    bundle = load_benchmark_bundle(args.benchmark)
    if bundle.dataset_v2 is None:
        raise ValueError("B3 Judge pilot requires Benchmark v2 authored claims.")
    base_payload = json.loads(Path(args.base_results).read_text(encoding="utf-8"))
    base_by_sample = {item["sample_id"]: item for item in base_payload["results"]}
    sample_by_id = {item.id: item for item in bundle.dataset_v2.samples}
    normalized_by_id = {item.id: item for item in bundle.samples}
    replay_store = ReplayStore(config.runtime.run_dir)
    evaluator = ReportEvaluationService(config)
    results = []
    for sample_id, authored in sample_by_id.items():
        base_result = base_by_sample[sample_id]
        replay = replay_store.read(base_result["run_id"])
        report = ResearchReport.model_validate(replay["report"])
        normalized = normalized_by_id[sample_id]
        results.append(
            evaluator.evaluate(
                sample_id,
                build_generation_task(normalized),
                report,
                normalized.expected_claims,
                float(base_result.get("latency_seconds", 0.0)),
                base_result.get("metadata", {}).get("process_metrics"),
                gold_claims=authored.claims,
            )
        )

    created_at = datetime.now(timezone.utc)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / build_benchmark_evaluation_filename(created_at)
    output.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "metric_protocol": "claim_citation_v2+gold_claim_v1+judge_b3.1",
                "created_at": created_at.isoformat(),
                "benchmark_path": str(args.benchmark),
                "base_results_path": str(args.base_results),
                "judge_protocol": evaluator.protocol_manifest().model_dump(mode="json"),
                "results": [item.model_dump(mode="json") for item in results],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

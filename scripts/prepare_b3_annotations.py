"""Create independent pending B3 annotation packets from a benchmark result artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from deepresearch.evaluation.annotations import (
    HumanAnnotationBundle,
    build_annotation_set,
    write_annotation_bundle,
)
from deepresearch.evaluation.benchmark import load_benchmark_bundle
from deepresearch.orchestrator.replay import ReplayStore
from deepresearch.schema import ResearchReport


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--run-dir", default="runs")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--annotators",
        default="annotator_a_pending,annotator_b_pending",
        help="Comma-separated independent annotator IDs.",
    )
    parser.add_argument(
        "--adjudicator",
        default="adjudicator_pending",
        help="ID for the separate pending adjudication packet.",
    )
    args = parser.parse_args()

    benchmark_path = Path(args.benchmark)
    bundle = load_benchmark_bundle(benchmark_path)
    if bundle.dataset_v2 is None:
        raise ValueError("B3 annotation packets require Benchmark v2 authored claims.")
    result_payload = json.loads(Path(args.results).read_text(encoding="utf-8"))
    result_by_sample = {item["sample_id"]: item for item in result_payload["results"]}
    replay_store = ReplayStore(args.run_dir)
    annotator_ids = [item.strip() for item in args.annotators.split(",") if item.strip()]
    if len(annotator_ids) < 2:
        raise ValueError("B3 requires at least two independent annotator IDs.")

    assignments = [(annotator_id, "annotator") for annotator_id in annotator_ids]
    assignments.append((args.adjudicator, "adjudicator"))
    for annotator_id, role in assignments:
        sets = []
        for sample in bundle.dataset_v2.samples:
            result = result_by_sample[sample.id]
            replay = replay_store.read(result["run_id"])
            report = ResearchReport.model_validate(replay["report"])
            sets.append(
                build_annotation_set(
                    benchmark_id=bundle.dataset_v2.metadata.benchmark_id,
                    benchmark_version=bundle.dataset_v2.metadata.version,
                    sample_id=sample.id,
                    report=report,
                    gold_claims=sample.claims,
                    annotator_id=annotator_id,
                    role=role,
                )
            )
        packet = HumanAnnotationBundle(annotator_id=annotator_id, role=role, samples=sets)
        write_annotation_bundle(packet, Path(args.output_dir) / f"{annotator_id}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

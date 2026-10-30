"""Create an auditable item-level comparison between two B3 Judge artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from deepresearch.evaluation.annotations import HumanAnnotationBundle
from deepresearch.evaluation.calibration import compare_b3_protocol_results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-results", required=True)
    parser.add_argument("--candidate-results", required=True)
    parser.add_argument("--annotator-a", required=True)
    parser.add_argument("--annotator-b", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    baseline = json.loads(Path(args.baseline_results).read_text(encoding="utf-8"))
    candidate = json.loads(Path(args.candidate_results).read_text(encoding="utf-8"))
    annotator_a = HumanAnnotationBundle.model_validate_json(
        Path(args.annotator_a).read_text(encoding="utf-8")
    )
    annotator_b = HumanAnnotationBundle.model_validate_json(
        Path(args.annotator_b).read_text(encoding="utf-8")
    )
    payload = compare_b3_protocol_results(
        baseline,
        candidate,
        annotator_a,
        annotator_b,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

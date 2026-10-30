"""Compare repeated Judge labels with completed human B3 annotations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from deepresearch.evaluation.annotations import (
    HumanAnnotationBundle,
)
from deepresearch.evaluation.calibration import build_b3_calibration


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--automatic-results", required=True)
    parser.add_argument("--annotator-a", required=True)
    parser.add_argument("--annotator-b", required=True)
    parser.add_argument(
        "--adjudication",
        help="Completed adjudication JSON; required only when A/B labels disagree.",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    automatic = json.loads(Path(args.automatic_results).read_text(encoding="utf-8"))
    annotator_a = HumanAnnotationBundle.model_validate_json(
        Path(args.annotator_a).read_text(encoding="utf-8")
    )
    annotator_b = HumanAnnotationBundle.model_validate_json(
        Path(args.annotator_b).read_text(encoding="utf-8")
    )
    adjudication = (
        HumanAnnotationBundle.model_validate_json(
            Path(args.adjudication).read_text(encoding="utf-8")
        )
        if args.adjudication
        else None
    )
    payload = build_b3_calibration(
        automatic,
        annotator_a,
        annotator_b,
        adjudication=adjudication,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0
if __name__ == "__main__":
    raise SystemExit(main())

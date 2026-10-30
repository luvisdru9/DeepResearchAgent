"""Create a disagreement-only B3 adjudication packet and optional HTML form."""

from __future__ import annotations

import argparse
from pathlib import Path

from deepresearch.evaluation.annotation_form import write_annotation_form
from deepresearch.evaluation.annotations import (
    HumanAnnotationBundle,
    write_annotation_bundle,
)
from deepresearch.evaluation.calibration import (
    build_disagreement_adjudication_bundle,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two completed independent B3 bundles and create a blank "
            "third-human packet containing only their disagreements."
        )
    )
    parser.add_argument("--annotator-a", required=True)
    parser.add_argument("--annotator-b", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--form-output")
    parser.add_argument("--adjudicator-id", default="adjudicator_pending")
    args = parser.parse_args()

    annotator_a = HumanAnnotationBundle.model_validate_json(
        Path(args.annotator_a).read_text(encoding="utf-8")
    )
    annotator_b = HumanAnnotationBundle.model_validate_json(
        Path(args.annotator_b).read_text(encoding="utf-8")
    )
    packet = build_disagreement_adjudication_bundle(
        annotator_a,
        annotator_b,
        adjudicator_id=args.adjudicator_id,
    )
    destination = write_annotation_bundle(packet, args.output)
    print(destination)
    if args.form_output:
        form = write_annotation_form(
            packet,
            args.form_output,
            source_name=destination.name,
        )
        print(form)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

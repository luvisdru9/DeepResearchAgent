"""Generate self-contained browser forms from frozen B3 annotation packets."""

from __future__ import annotations

import argparse
from pathlib import Path

from deepresearch.evaluation.annotation_form import write_annotation_form
from deepresearch.evaluation.annotations import HumanAnnotationBundle


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate local HTML forms without modifying source annotation JSON."
    )
    parser.add_argument("packets", nargs="+", help="Pending annotation JSON packets.")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    for packet_name in args.packets:
        packet_path = Path(packet_name)
        bundle = HumanAnnotationBundle.model_validate_json(
            packet_path.read_text(encoding="utf-8")
        )
        form_name = f"{packet_path.stem}_form.html"
        destination = write_annotation_form(
            bundle,
            output_dir / form_name,
            source_name=packet_path.name,
        )
        print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

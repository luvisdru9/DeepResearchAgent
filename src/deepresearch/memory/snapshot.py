"""Persistence for reusable Evidence snapshots used by paired experiments."""

from __future__ import annotations

import json
from pathlib import Path

from deepresearch.artifact_naming import build_run_artifact_stem
from deepresearch.schema import ResearchEvidenceSnapshot


class EvidenceSnapshotStore:
    """Write/read snapshot JSON independently from normal Replay artifacts."""

    def __init__(self, run_dir: str | Path, *, subdir: str = "snapshots") -> None:
        self.output_dir = Path(run_dir) / subdir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write(self, snapshot: ResearchEvidenceSnapshot) -> Path:
        stem = build_run_artifact_stem(snapshot.snapshot_id, snapshot.created_at)
        path = self.output_dir / f"{stem}_evidence_snapshot.json"
        path.write_text(snapshot.model_dump_json(indent=2), encoding="utf-8")
        return path

    def read(self, path: str | Path) -> ResearchEvidenceSnapshot:
        return ResearchEvidenceSnapshot.model_validate(
            json.loads(Path(path).read_text(encoding="utf-8"))
        )

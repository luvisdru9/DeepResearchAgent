"""Export reviewable replay/report examples from a comparison artifact."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from deepresearch.orchestrator.replay import ReplayStore


def export_comparison_examples(
    comparison_path: str | Path,
    *,
    run_dir: str | Path,
    output_dir: str | Path = "examples/generated",
    strategy: str = "dag_memory_redteam_compress",
    limit: int = 3,
) -> Path:
    """Export complete replays and reports without copying raw LLM trace files."""

    if limit < 1:
        raise ValueError("Example export limit must be at least 1.")
    comparison = json.loads(Path(comparison_path).read_text(encoding="utf-8"))
    variant = next(
        (item for item in comparison.get("results", []) if item.get("strategy") == strategy),
        None,
    )
    if variant is None:
        raise ValueError(f"Strategy '{strategy}' is absent from comparison artifact.")

    destination = Path(output_dir)
    replay_dir = destination / "replays"
    report_dir = destination / "reports"
    replay_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    store = ReplayStore(run_dir)
    exported: list[dict[str, Any]] = []
    for item in variant.get("results", [])[:limit]:
        sample_id = str(item["sample_id"])
        run_id = str(item["run_id"])
        replay = store.read(run_id)
        replay_path = replay_dir / f"{sample_id}_{run_id}_replay.json"
        report_path = report_dir / f"{sample_id}_{run_id}_report.md"
        replay_path.write_text(
            json.dumps(replay, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        report_path.write_text(replay["report"]["markdown"], encoding="utf-8")
        exported.append(
            {
                "sample_id": sample_id,
                "run_id": run_id,
                "replay_path": str(replay_path),
                "report_path": str(report_path),
                "source_comparison": str(comparison_path),
                "strategy": strategy,
            }
        )
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "source_comparison": str(comparison_path),
                "strategy": strategy,
                "examples": exported,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return manifest_path


def export_run_examples(
    run_ids: list[str],
    *,
    run_dir: str | Path,
    output_dir: str | Path = "examples/real",
) -> Path:
    """Export explicitly selected real runs as human-review candidates."""

    if not run_ids:
        raise ValueError("At least one run id is required.")
    destination = Path(output_dir)
    replay_dir = destination / "replays"
    report_dir = destination / "reports"
    replay_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    store = ReplayStore(run_dir)
    exported: list[dict[str, Any]] = []
    for run_id in dict.fromkeys(run_ids):
        replay = store.read(run_id)
        language = str(replay.get("task", {}).get("language") or "unknown")
        replay_path = replay_dir / f"{language}_{run_id}_replay.json"
        report_path = report_dir / f"{language}_{run_id}_report.md"
        replay_path.write_text(
            json.dumps(replay, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        report_path.write_text(replay["report"]["markdown"], encoding="utf-8")
        exported.append(
            {
                "run_id": run_id,
                "language": language,
                "query": replay.get("task", {}).get("query"),
                "replay_path": str(replay_path),
                "report_path": str(report_path),
                "review_status": "candidate_requires_human_review",
            }
        )
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "kind": "real_pipeline_candidates",
                "examples": exported,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return manifest_path

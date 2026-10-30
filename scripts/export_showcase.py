"""Build the public resume showcase from an explicit, sanitized allowlist.

The development repository intentionally keeps full experiment history. This exporter creates a
separate tree containing source code, offline tests, synthetic frozen fixtures, final reports, and
compact metrics only. It never copies ``runs/``, raw LLM traces, SQLite stores, completed annotation
packets, failed auto-draft directories, or retrieved document bodies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from deepresearch.cli import build_replay_summary  # noqa: E402
from deepresearch.evaluation.process_metrics import score_research_process  # noqa: E402
from deepresearch.retrieval.snapshot import (  # noqa: E402
    FrozenQueryRecord,
    write_frozen_snapshot,
)
from deepresearch.schema import SourceDoc  # noqa: E402


SHOWCASE_TEMPLATE = ROOT / "showcase_template"
DEFAULT_DESTINATION = ROOT / "dist/DeepResearchAgent-Showcase"
BENCHMARK_ROOT = Path("data/benchmarks/researchbench_v2")
SNAPSHOT_TESTS = {
    "test_bench002_snapshot_v2.py",
    "test_bench002_snapshot_v3.py",
    "test_bench002_snapshot_v4.py",
}
SOURCE_DATASET_NAMES = (
    "dataset.json",
    "dataset_v0.1.1.json",
    "dataset_v0.1.2.json",
    "dataset_v0.1.3.json",
)


def sha256(path: Path) -> str:
    """Return the prefixed SHA-256 notation used by experiment records."""

    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def git_commit() -> str | None:
    """Record source provenance without requiring a Git checkout in the output."""

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def prepare_destination(destination: Path, *, force: bool) -> None:
    """Create a clean output directory and refuse accidental source-tree deletion."""

    resolved = destination.resolve()
    if resolved == ROOT or ROOT.is_relative_to(resolved):
        raise ValueError("Showcase destination cannot be the source repository or its parent.")
    if destination.exists() and any(destination.iterdir()):
        if not force:
            raise FileExistsError(
                f"Destination is not empty: {destination}. Pass --force to rebuild it."
            )
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)


def copy_public_source(destination: Path) -> None:
    """Copy code and tests while replacing private-data regression files later."""

    for filename in ("pyproject.toml", ".env.example"):
        shutil.copy2(ROOT / filename, destination / filename)
    shutil.copytree(ROOT / "src", destination / "src")
    shutil.copytree(ROOT / "data/local_corpus", destination / "data/local_corpus")
    shutil.copytree(ROOT / "data/quality", destination / "data/quality")
    for filename in ("researchbench.json", "researchbench_full.json"):
        shutil.copy2(ROOT / "data" / filename, destination / "data" / filename)
    (destination / "configs").mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "configs/default.yaml", destination / "configs/default.yaml")
    # Capability boundaries are part of the public engineering evidence, not private run data.
    (destination / "docs").mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        ROOT / "docs/AGENT_CAPABILITY_GAPS.md",
        destination / "docs/AGENT_CAPABILITY_GAPS.md",
    )

    tests_destination = destination / "tests"
    tests_destination.mkdir(parents=True, exist_ok=True)
    for source in (ROOT / "tests").rglob("*"):
        if not source.is_file() or "__pycache__" in source.parts:
            continue
        relative = source.relative_to(ROOT / "tests")
        if source.name in SNAPSHOT_TESTS or relative.as_posix() == "test_experiment_records.py":
            continue
        target = tests_destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    schema_source = ROOT / BENCHMARK_ROOT / "schemas/benchmark_v2.schema.json"
    schema_target = destination / BENCHMARK_ROOT / "schemas/benchmark_v2.schema.json"
    schema_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(schema_source, schema_target)


def copy_template(destination: Path) -> None:
    """Overlay public-only documentation, CI, policy checks, and replacement tests."""

    shutil.copytree(SHOWCASE_TEMPLATE, destination, dirs_exist_ok=True)


def synthetic_contents(sample_id: str) -> dict[str, tuple[str, str]]:
    """Return project-authored fixture text keyed by the stable source IDs used in v0.1.0."""

    if sample_id == "bench_001":
        return {
            "b001_plan": (
                "DeepResearchAgent engineering plan fixture",
                (
                    "A deep research system decomposes a question into a dependency-aware DAG. "
                    "Planner nodes define research branches; retrieval and reading nodes preserve source "
                    "provenance; synthesis is allowed to cite only known Evidence IDs. SQLite checkpoints "
                    "store node status and Evidence so interrupted work can resume without repeating completed "
                    "nodes. Replay records planned, started, and completed order plus every retry attempt."
                ),
            ),
            "b001_multi_agent": (
                "Synthetic multi-agent architecture fixture",
                (
                    "The fixture pipeline separates Planner, Retriever, Reader, Synthesizer, RedTeam, Blue, "
                    "and Verifier responsibilities. Ready DAG nodes can execute concurrently. A failed node is "
                    "retried within a bound; downstream nodes are skipped when required dependencies fail. "
                    "Evidence keeps source, chunk, quote, and character offsets across Agent boundaries."
                ),
            ),
        }
    if sample_id == "bench_002":
        return {
            "b002_alce": (
                "Synthetic citation-evaluation fixture",
                (
                    "This project-authored fixture is not the ALCE paper. Citation coverage uses atomic factual "
                    "claims as the analysis unit. The numerator is claims with a valid citation; the denominator "
                    "is all claims that require evidence. Citation correctness checks semantic support or "
                    "entailment: whether the cited passage supports the claim. Checking only that a link or citation marker exists is "
                    "insufficient. Completeness and correctness must remain separate measurements."
                ),
            ),
            "b002_ragtruth": (
                "Synthetic hallucination-taxonomy fixture",
                (
                    "This project-authored fixture is not the RAGTruth paper. Hallucination evaluation first "
                    "splits a report into atomic factual claims. Each claim is classified as supported, "
                    "unsupported, contradicted, miscited, or uncited. Unsupported means no available Evidence "
                    "entails the claim; contradicted means Evidence conflicts with it; miscited means a citation "
                    "is present but does not support it; uncited means an evidence-requiring claim has no citation."
                ),
            ),
        }
    if sample_id == "bench_003":
        return {
            "b003_plan": (
                "DeepResearchAgent verification-plan fixture",
                (
                    "RedTeam checks factuality, citation support, logic, coverage, and time scope. Findings that "
                    "need new Evidence are deferred to bounded retrieval. Text-repairable findings can be sent "
                    "to Blue, but a repaired report is never promoted automatically: Verifier must review it again."
                ),
            ),
            "b003_red_blue": (
                "Synthetic Red–Blue repair fixture",
                (
                    "Blue repair uses explicit actions: remove an unsupported clause, narrow an over-broad claim, "
                    "qualify uncertainty, or revise wording without adding new Evidence IDs. If required coverage "
                    "or factual support remains missing, the terminal result is quality_unresolved rather than pass."
                ),
            ),
        }
    raise KeyError(f"Unknown showcase sample: {sample_id}")


def build_synthetic_benchmark(destination: Path) -> None:
    """Create hash-valid frozen snapshots using only short project-authored fixtures."""

    source_root = ROOT / BENCHMARK_ROOT
    target_root = destination / BENCHMARK_ROOT
    target_root.mkdir(parents=True, exist_ok=True)
    base = json.loads((source_root / "dataset.json").read_text(encoding="utf-8"))
    manifests: dict[str, Any] = {}

    for sample in base["samples"]:
        sample_id = sample["id"]
        snapshot = sample["snapshot"]
        source_map = synthetic_contents(sample_id)
        required_ids = list(
            dict.fromkeys(
                source_id
                for requirement in sample["evidence_requirements"]
                for source_id in requirement["required_source_ids"]
            )
        )
        # Keep all stable fixture sources even when one release requirement cites only a subset.
        ordered_ids = [source_id for source_id in source_map if source_id in required_ids]
        ordered_ids.extend(source_id for source_id in source_map if source_id not in ordered_ids)
        documents = [
            SourceDoc(
                id=source_id,
                title=source_map[source_id][0],
                url=f"showcase://fixtures/{source_id}",
                content=source_map[source_id][1],
                source_type="synthetic_fixture",
                metadata={
                    "provider": "synthetic_showcase",
                    "fixture_kind": "showcase_synthetic",
                    "redistribution": "project_authored",
                },
            )
            for source_id in ordered_ids
        ]
        manifest_path = target_root / snapshot["manifest_path"]
        manifest = write_frozen_snapshot(
            manifest_path.parent,
            snapshot_id=snapshot["snapshot_id"],
            sample_id=sample_id,
            benchmark_version="0.1.0",
            captured_at=datetime(2026, 10, 19, tzinfo=timezone.utc),
            provider="synthetic_showcase",
            queries=[
                FrozenQueryRecord(
                    id="showcase_q1",
                    query=sample["question"],
                    result_source_ids=ordered_ids,
                )
            ],
            documents=documents,
        )
        manifests[sample_id] = manifest

    for filename in SOURCE_DATASET_NAMES:
        payload = json.loads((source_root / filename).read_text(encoding="utf-8"))
        payload["metadata"]["license"] = "Project-authored synthetic showcase fixtures."
        payload["metadata"]["dataset_card"] = "docs/PORTFOLIO_EVIDENCE.md"
        payload["metadata"]["change_notes"] = [
            *payload["metadata"].get("change_notes", []),
            "Public showcase replaces external source bodies with project-authored synthetic fixtures.",
        ]
        for sample in payload["samples"]:
            manifest = manifests[sample["id"]]
            sample["snapshot"] = {
                "snapshot_id": manifest.snapshot_id,
                "manifest_path": base_snapshot_path(base, sample["id"]),
                "content_hash": manifest.content_hash,
            }
        (target_root / filename).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )


def base_snapshot_path(base: dict[str, Any], sample_id: str) -> str:
    """Resolve the unchanged public manifest path for one base sample."""

    return next(
        item["snapshot"]["manifest_path"]
        for item in base["samples"]
        if item["id"] == sample_id
    )


def export_reports_and_replay_summaries(destination: Path) -> None:
    """Publish final Markdown and compact state summaries, never raw Replay payloads."""

    report_target = destination / "examples/showcase/reports"
    replay_target = destination / "examples/showcase/replay_summaries"
    report_target.mkdir(parents=True, exist_ok=True)
    replay_target.mkdir(parents=True, exist_ok=True)
    reports = {
        "zh_ddb0210e931e_report.md": "oauth_oidc_zh.md",
        "en_a14b1c85d2ea_report.md": "remote_work_en.md",
    }
    for source_name, target_name in reports.items():
        shutil.copy2(ROOT / "examples/track_a/reports" / source_name, report_target / target_name)

    manifest: list[dict[str, Any]] = []
    for source in sorted((ROOT / "examples/track_a/replays").glob("*.json")):
        payload = json.loads(source.read_text(encoding="utf-8"))
        summary = build_replay_summary(payload)
        process_metrics = score_research_process(payload)
        # These counts make the public "zero LLM failure fallback" claim auditable
        # without exposing prompts, model output, source bodies, or Evidence text.
        summary["process_summary"] = {
            "llm_failure_fallback_count": process_metrics["fallback_count"],
            "empty_result_rule_assist_count": process_metrics[
                "empty_result_rule_assist_count"
            ],
            "retry_count": process_metrics["retry_count"],
            "failed_stages": process_metrics["failed_stages"],
        }
        summary["artifact_kind"] = "sanitized_replay_summary"
        summary["raw_replay_public"] = False
        target = replay_target / source.name.replace("_replay.json", "_summary.json")
        target.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest.append(
            {
                "run_id": summary["run_id"],
                "query": summary["query"],
                "summary_path": str(target.relative_to(destination)),
            }
        )
    (destination / "examples/showcase/manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "kind": "sanitized_showcase",
                "reports": [
                    "examples/showcase/reports/oauth_oidc_zh.md",
                    "examples/showcase/reports/remote_work_en.md",
                ],
                "replay_summaries": manifest,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def export_comparison(destination: Path) -> None:
    """Keep aggregate and per-sample outcomes while removing runtime paths and full Replays."""

    source_json = ROOT / "runs/comparisons/20261018T133409Z_system_comparison.json"
    source_md = ROOT / "runs/comparisons/20261018T133409Z_system_comparison.md"
    payload = json.loads(source_json.read_text(encoding="utf-8"))
    compact_results = []
    for strategy in payload["results"]:
        compact_results.append(
            {
                key: value
                for key, value in strategy.items()
                if key not in {"results", "failures", "total_cost_usd"}
            }
            | {
                "samples": [
                    {
                        key: value
                        for key, value in item.items()
                        if key
                        in {
                            "sample_id",
                            "completion_status",
                            "quality_status",
                            "fallback_count",
                            "evidence_count",
                            "source_count",
                            "independent_domain_count",
                            "required_source_recall",
                            "llm_calls",
                            "latency_seconds",
                            "llm_input_tokens",
                            "llm_output_tokens",
                        }
                    }
                    for item in strategy["results"]
                ]
            }
        )
    compact = {
        "schema_version": "showcase-1.0",
        "evaluation_mode": "engineering_automatic",
        "human_calibrated": False,
        "formal_benchmark_eligible": False,
        "created_at": payload["created_at"],
        "sample_ids": payload["sample_ids"],
        "strategies": compact_results,
        "paired_deltas": payload["paired_deltas"],
        "limitations": [
            "Three samples do not support statistical significance or causal claims.",
            "Only baseline and the complete strategy received semantic Judge evaluation.",
            "full_015 remained quality_unresolved in the complete strategy.",
        ],
    }
    target = destination / "examples/showcase/comparisons"
    target.mkdir(parents=True, exist_ok=True)
    (target / "system_comparison.json").write_text(
        json.dumps(compact, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    comparison_markdown = source_md.read_text(encoding="utf-8")
    comparison_markdown += (
        "\n> Scope: three frozen samples only. These descriptive scores are not a statistical "
        "significance or causal-improvement claim; only baseline and the complete strategy received "
        "semantic Judge evaluation.\n"
    )
    (target / "system_comparison.md").write_text(
        comparison_markdown,
        encoding="utf-8",
    )


def export_auto_draft_summary(destination: Path) -> None:
    """Publish source URLs/hashes and claim requirements without retrieved source bodies."""

    source_root = (
        ROOT
        / "data/benchmarks/researchbench_v2/track_a_auto_draft/20261018T132707_963465Z"
    )
    dataset = json.loads((source_root / "dataset.json").read_text(encoding="utf-8"))
    build = json.loads((source_root / "build_result.json").read_text(encoding="utf-8"))
    build_by_id = {item["sample_id"]: item for item in build["samples"]}
    samples: list[dict[str, Any]] = []
    for sample in dataset["samples"]:
        sample_build = build_by_id[sample["id"]]
        manifest_path = source_root / sample["snapshot"]["manifest_path"]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        sources = []
        for record in manifest["documents"]:
            source_doc = json.loads(
                (manifest_path.parent / record["path"]).read_text(encoding="utf-8")
            )
            sources.append(
                {
                    "source_id": record["source_id"],
                    "title": source_doc["title"],
                    "url": source_doc["url"],
                    "content_hash": record["content_hash"],
                }
            )
        samples.append(
            {
                "sample_id": sample["id"],
                "question": sample["question"],
                "task_type": sample["task_type"],
                "status": sample_build["status"],
                "source_count": sample_build["source_count"],
                "independent_domain_count": sample_build["independent_domain_count"],
                "primary_source_count": sample_build["primary_source_count"],
                "llm_calls": sample_build["llm_calls"],
                "total_tokens": sample_build["total_tokens"],
                "elapsed_seconds": sample_build["elapsed_seconds"],
                "claims": [
                    {
                        "id": claim["id"],
                        "text": claim["text"],
                        "required_source_ids": next(
                            requirement["required_source_ids"]
                            for requirement in sample["evidence_requirements"]
                            if claim["id"] in requirement["claim_ids"]
                        ),
                    }
                    for claim in sample["claims"]
                ],
                "sources": sources,
            }
        )
    summary = {
        "schema_version": "showcase-1.0",
        "evaluation_mode": "fully_automatic",
        "release_tier": "engineering_automatic",
        "human_calibrated": False,
        "formal_benchmark_eligible": False,
        "created_at": build["created_at"],
        "passed_count": build["passed_count"],
        "failed_count": build["failed_count"],
        "body_policy": "URLs, hashes, titles, claims, and resource metrics only; no retrieved bodies.",
        "samples": samples,
    }
    target = destination / "examples/showcase/auto_draft_summary.json"
    target.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def build_experiment_record(destination: Path) -> None:
    """Create one self-contained public record whose paths all resolve inside the mirror."""

    schema_dir = destination / "experiments/schema"
    schema_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        ROOT / "experiments/schema/experiment_record.schema.json",
        schema_dir / "experiment_record.schema.json",
    )
    summary_path = destination / "examples/showcase/auto_draft_summary.json"
    record_path = destination / "experiments/records/20261019_showcase_release_v1/record.json"
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": "1.0",
        "experiment_id": "20261019_showcase_release_v1",
        "revision_of": "20261018_track_a_engineering_delivery_v1",
        "recorded_at": "2026-10-19T00:00:00Z",
        "record_type": "framework_baseline",
        "status": "completed",
        "objective": "Publish a sanitized, offline-verifiable engineering showcase without raw third-party bodies or private runtime artifacts.",
        "code": {
            "git_commit": git_commit(),
            "worktree_state": "clean",
            "python_version": "3.11",
            "conda_environment": "portable Python 3.11",
            "config": {
                "path": "configs/showcase.yaml",
                "sha256": sha256(destination / "configs/showcase.yaml"),
                "overrides": {},
            },
            "dependencies": {"install": "pip install -e '.[dev,api]'"},
        },
        "system": {
            "pipeline": [
                "Planner",
                "Retriever",
                "Reader",
                "Evidence Store",
                "Synthesizer",
                "RedTeam",
                "Blue repair",
                "Verifier",
                "Report and Replay",
            ],
            "components": [
                {
                    "name": "showcase_pipeline",
                    "implementation": "shared production-shaped Pipeline with offline rule fixture",
                    "provider": "local synthetic fixture",
                    "enabled": True,
                    "notes": "No API credential required for CI smoke.",
                }
            ],
            "capabilities": {
                "async_dag": True,
                "evidence_provenance": True,
                "quality_closure": True,
                "sqlite_resume": True,
                "fastapi_local_mvp": True,
            },
        },
        "benchmark": {
            "id": "track_a_public_summary",
            "path": "examples/showcase/auto_draft_summary.json",
            "sha256": sha256(summary_path),
            "schema_version": "showcase-1.0",
            "version": "0.1.0",
            "track": "engineering_automatic_summary",
            "sample_count": 8,
            "sample_ids": [item["sample_id"] for item in json.loads(summary_path.read_text())["samples"]],
            "task_types": [
                "fact_qa",
                "comparative_research",
                "trend_analysis",
                "controversy_multihop",
            ],
            "snapshot_hashes": {},
            "status": "draft",
        },
        "evaluation": {
            "protocol": {
                "release_tier": "engineering_automatic",
                "human_calibrated": False,
                "formal_benchmark_eligible": False,
                "public_artifact_policy": "sanitized summaries and synthetic fixtures only",
            },
            "metric_definitions": [
                {
                    "name": "artifact_completion_rate",
                    "category": "engineering",
                    "direction": "higher_better",
                    "unit": "ratio",
                    "definition": "Requested executions that produced readable structured artifacts.",
                    "implementation_status": "implemented",
                    "limitations": "Does not imply every report passed semantic verification.",
                }
            ],
        },
        "results": {
            "outcome": "Engineering showcase exported with explicit non-formal evaluation boundaries.",
            "engineering_checks": [
                {"name": "automated_tests", "passed": True, "result": "380"},
                {"name": "live_demo_artifacts", "passed": True, "result": "2/2"},
                {"name": "auto_draft_gate", "passed": True, "result": "8/8"},
                {"name": "strategy_artifacts", "passed": True, "result": "15/15"},
            ],
            "metrics": [
                {
                    "name": "automated_tests",
                    "status": "measured",
                    "value": 380,
                    "unit": "tests",
                    "notes": "Public suite uses synthetic external-source fixtures.",
                },
                {
                    "name": "auto_draft_pass_rate",
                    "status": "measured",
                    "value": 1.0,
                    "unit": "ratio",
                    "notes": "Eight fixed engineering drafts; not human calibrated.",
                },
                {
                    "name": "strategy_artifact_completion_rate",
                    "status": "measured",
                    "value": 1.0,
                    "unit": "ratio",
                    "notes": "Fifteen serialized runs; quality_unresolved remains a valid terminal result.",
                },
            ],
            "observations": [
                "The public mirror prioritizes inspectability and safe redistribution over complete raw traces.",
                "No statistical or causal strategy-gain claim is made from three samples.",
            ],
        },
        "comparison": {
            "group": "track_a_showcase",
            "comparability": "partial",
            "baseline_experiment_id": None,
            "changed_variables": ["DAG", "memory", "RedTeam", "compression"],
            "notes": "See the compact comparison artifact for paired descriptive outcomes.",
        },
        "artifacts": [
            {"type": "readme", "path": "README.md", "run_id": None, "exists": True, "notes": "Public entry point."},
            {"type": "report_zh", "path": "examples/showcase/reports/oauth_oidc_zh.md", "run_id": "ddb0210e931e", "exists": True, "notes": "Sanitized final report."},
            {"type": "report_en", "path": "examples/showcase/reports/remote_work_en.md", "run_id": "a14b1c85d2ea", "exists": True, "notes": "Sanitized final report."},
            {"type": "comparison", "path": "examples/showcase/comparisons/system_comparison.json", "run_id": None, "exists": True, "notes": "Compact metrics only."},
            {"type": "evidence_index", "path": "docs/PORTFOLIO_EVIDENCE.md", "run_id": None, "exists": True, "notes": "Resume claim provenance."},
        ],
        "limitations": [
            "Automatic evaluation is not human calibrated.",
            "The strategy comparison has three samples and does not establish statistical significance.",
            "The local FastAPI layer is not a production deployment.",
            "Raw third-party bodies and raw model traces are intentionally excluded.",
        ],
        "next_actions": [
            "Use the complete development repository for formal Benchmark work.",
            "Keep this public mirror limited to reviewed engineering artifacts.",
        ],
    }
    record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    index = {
        "schema_version": "1.0",
        "records": [
            {
                "experiment_id": record["experiment_id"],
                "record_type": record["record_type"],
                "status": record["status"],
                "recorded_at": record["recorded_at"],
                "path": "records/20261019_showcase_release_v1/record.json",
                "comparison_group": "track_a_showcase",
            }
        ],
    }
    (destination / "experiments/index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (destination / "experiments/README.md").write_text(
        "# Public experiment record\n\nThis directory contains one sanitized Track A showcase record.\n",
        encoding="utf-8",
    )


def write_export_manifest(destination: Path) -> None:
    """Describe source provenance and exclusions for reviewers and future rebuilds."""

    payload = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_commit": git_commit(),
        "artifact_kind": "sanitized_public_showcase",
        "includes": [
            "source code and offline tests",
            "project-authored synthetic snapshots",
            "two final reports",
            "three compact Replay summaries",
            "compact system comparison",
            "eight-sample automatic-draft provenance summary",
        ],
        "excludes": [
            "credentials and .env",
            "runs and SQLite stores",
            "raw LLM traces",
            "complete raw Replay payloads",
            "completed human annotations",
            "failed auto-draft directories",
            "retrieved third-party source bodies",
        ],
    }
    (destination / "SHOWCASE_MANIFEST.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def build_showcase(destination: Path, *, force: bool) -> Path:
    """Run the deterministic export in dependency order."""

    prepare_destination(destination, force=force)
    copy_public_source(destination)
    copy_template(destination)
    build_synthetic_benchmark(destination)
    export_reports_and_replay_summaries(destination)
    export_comparison(destination)
    export_auto_draft_summary(destination)
    build_experiment_record(destination)
    write_export_manifest(destination)
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--force", action="store_true", help="Replace an existing generated tree.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    destination = build_showcase(args.destination, force=args.force)
    print(json.dumps({"showcase_path": str(destination)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

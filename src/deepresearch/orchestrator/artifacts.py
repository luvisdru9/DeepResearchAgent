"""Persist one self-contained directory for every completed research report."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from deepresearch.artifact_naming import build_run_artifact_stem
from deepresearch.schema import ResearchReport


REPORT_ARTIFACT_SCHEMA_VERSION = "1.1"


@dataclass(frozen=True, slots=True)
class ReportArtifactPaths:
    """Deterministic locations recorded in Replay before files are committed."""

    directory: Path
    markdown: Path
    report_json: Path
    manifest: Path

    def as_dict(self) -> dict[str, str]:
        return {key: str(value) for key, value in asdict(self).items()}


class ReportArtifactStore:
    """Write human-readable and structured report artifacts into one run folder."""

    def __init__(self, run_dir: str | Path, *, subdir: str = "reports") -> None:
        relative_subdir = Path(subdir)
        if relative_subdir.is_absolute() or ".." in relative_subdir.parts:
            raise ValueError("runtime.report_subdir must stay inside runtime.run_dir.")
        self.root = Path(run_dir) / relative_subdir

    def paths_for(self, run_id: str, created_at: Any) -> ReportArtifactPaths:
        directory = self.root / build_run_artifact_stem(run_id, created_at)
        return ReportArtifactPaths(
            directory=directory,
            markdown=directory / "report.md",
            report_json=directory / "report.json",
            manifest=directory / "manifest.json",
        )

    def write(
        self,
        report: ResearchReport,
        *,
        created_at: Any,
        replay_path: str | Path,
        quality_status: str,
        stop_reason: str,
        completion_status: str,
        coverage_required: bool,
        coverage_complete: bool,
        missing_dimension_count: int,
        open_gap_count: int,
    ) -> ReportArtifactPaths:
        """Commit a complete report folder; callers treat write failure as run failure."""

        paths = self.paths_for(report.run_id, created_at)
        paths.directory.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema_version": REPORT_ARTIFACT_SCHEMA_VERSION,
            "run_id": report.run_id,
            "query": report.query,
            "created_at": report.created_at.isoformat(),
            "quality_status": quality_status,
            # quality_status 只表示最终核验结论；completion_status 才是面向
            # API/CLI/实验消费者的整体交付状态，防止 pass 被误读为覆盖完整。
            "completion_status": completion_status,
            "stop_reason": stop_reason,
            "coverage": {
                "required": coverage_required,
                "complete": coverage_complete,
                "missing_dimension_count": missing_dimension_count,
                "open_gap_count": open_gap_count,
            },
            "replay_path": str(replay_path),
            "files": {
                "markdown": paths.markdown.name,
                "report_json": paths.report_json.name,
            },
        }
        self._atomic_write(paths.markdown, report.markdown)
        self._atomic_write(
            paths.report_json,
            # 用户报告 JSON 与 Markdown 使用同一交付边界；内部核验明细仅保存在 Replay。
            json.dumps(
                report.model_dump(mode="json", exclude={"redteam_findings"}),
                ensure_ascii=False,
                indent=2,
            ),
        )
        # manifest 最后提交，可把它的存在视为该目录完整写入的标志。
        self._atomic_write(
            paths.manifest,
            json.dumps(manifest, ensure_ascii=False, indent=2),
        )
        return paths

    @staticmethod
    def _atomic_write(path: Path, content: str) -> None:
        """Replace one artifact atomically so interrupted writes do not leave partial JSON."""

        temporary = path.with_suffix(f"{path.suffix}.tmp")
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)

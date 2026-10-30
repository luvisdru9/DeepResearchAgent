"""Persistence for full LLM request/response traces kept outside replay payloads."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from deepresearch.artifact_naming import build_llm_trace_filename, build_run_artifact_stem
from deepresearch.llm.types import LLMRawTrace


class LLMTraceStore:
    """Write one JSON file per logical LLM call under a run-scoped trace directory."""

    def __init__(self, run_dir: str | Path, subdir: str = "llm_traces") -> None:
        self.run_dir = Path(run_dir)
        self.subdir = subdir
        self.base_dir = self.run_dir / subdir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def write(self, trace: LLMRawTrace, *, run_created_at: Any | None = None) -> Path:
        """Persist one raw trace and return its path."""

        path = self.path_for(trace, run_created_at=run_created_at)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self._jsonable(trace), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path

    def path_for(self, trace: LLMRawTrace, *, run_created_at: Any | None = None) -> Path:
        """Map one raw trace to a readable run directory and trace filename."""

        run_dirname = build_run_artifact_stem(trace.run_id, run_created_at)
        filename = build_llm_trace_filename(
            started_at=trace.started_at,
            agent=trace.agent,
            operation=trace.operation,
            call_id=trace.call_id,
        )
        return self.base_dir / run_dirname / filename

    def relative_path_for(self, trace: LLMRawTrace, *, run_created_at: Any | None = None) -> str:
        """Return the replay-facing relative path for one raw trace."""

        return str(self.path_for(trace, run_created_at=run_created_at).relative_to(self.run_dir))

    @classmethod
    def _jsonable(cls, value: Any) -> Any:
        """Recursively convert Pydantic models into JSON-serializable values."""

        if isinstance(value, BaseModel):
            return value.model_dump(mode="json")
        if isinstance(value, list):
            return [cls._jsonable(item) for item in value]
        if isinstance(value, dict):
            return {key: cls._jsonable(item) for key, item in value.items()}
        return value

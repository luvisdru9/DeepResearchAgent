"""
负责把每次运行的 task、subtasks、docs、evidence、report、metrics 写成 JSON 文件。
这个是后续调试、复盘、评测可复现的基础。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from deepresearch.artifact_naming import build_replay_filename
from deepresearch.schema import NodeTrace, SubTask

REPLAY_SCHEMA_VERSION = "2.0"


def build_dag_replay(
    subtasks: list[SubTask],
    traces: list[NodeTrace],
    *,
    started_order: list[str] | None = None,
    completed_order: list[str] | None = None,
) -> dict[str, Any]:
    """Build a compact, human-readable DAG view from planning and execution data."""

    trace_by_id = {trace.subtask_id: trace for trace in traces}
    dependent_ids = {dependency for subtask in subtasks for dependency in subtask.depends_on}
    status_counts: dict[str, int] = {}
    nodes: list[dict[str, Any]] = []

    for subtask in subtasks:
        trace = trace_by_id.get(subtask.id)
        status = trace.status if trace is not None else subtask.status
        status_counts[status.value] = status_counts.get(status.value, 0) + 1

        # nodes 提供快速总览；完整错误和时间信息仍保留在 node_traces 中。
        nodes.append(
            {
                "id": subtask.id,
                "question": subtask.question,
                "status": status,
                "depends_on": subtask.depends_on,
                "attempts": trace.attempts if trace is not None else 0,
                "latency_seconds": trace.latency_seconds if trace is not None else 0.0,
            }
        )

    # 显式边比 depends_on 更适合画图，也便于后续异步执行器重建任务拓扑。
    edges = [
        {"from": dependency, "to": subtask.id}
        for subtask in subtasks
        for dependency in subtask.depends_on
    ]

    return {
        "summary": {
            "total_nodes": len(subtasks),
            "total_edges": len(edges),
            "status_counts": status_counts,
        },
        "root_nodes": [subtask.id for subtask in subtasks if not subtask.depends_on],
        "leaf_nodes": [subtask.id for subtask in subtasks if subtask.id not in dependent_ids],
        # 三种顺序分别表达 Planner 意图、调度启动和实际结束，异步执行时不再混为一谈。
        "planned_order": [subtask.id for subtask in subtasks],
        "started_order": started_order
        if started_order is not None
        else [trace.subtask_id for trace in traces if trace.started_at is not None],
        "completed_order": completed_order
        if completed_order is not None
        else [trace.subtask_id for trace in traces],
        "nodes": nodes,
        "edges": edges,
    }


class ReplayStore:
    """File-based replay store for reproducible local experiments."""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def write(self, run_id: str, payload: dict[str, Any]) -> Path:
        """Write one run payload as formatted JSON."""

        versioned_payload = {**payload, "schema_version": REPLAY_SCHEMA_VERSION}
        path = self.path_for(run_id, payload=versioned_payload)
        # Store controls the on-disk version; callers cannot accidentally override it.
        path.write_text(json.dumps(self._jsonable(versioned_payload), ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def read(self, run_id: str) -> dict[str, Any]:
        """Load replay JSON and identify legacy files without inventing missing fields."""

        path = self.path_for(run_id)
        payload = json.loads(path.read_text(encoding="utf-8"))
        # 旧文件保持原内容可读，只补充版本标识，避免伪造当时并未记录的 DAG 信息。
        payload.setdefault("schema_version", "1.0")
        return payload

    def path_for(self, run_id: str, payload: dict[str, Any] | None = None) -> Path:
        """Map a stable run id to its replay file, while keeping readable filenames."""

        if payload is not None:
            task_payload = payload.get("task")
            if isinstance(task_payload, BaseModel):
                created_at = getattr(task_payload, "created_at", None)
            elif isinstance(task_payload, dict):
                created_at = task_payload.get("created_at")
            else:
                created_at = None
            return self.run_dir / build_replay_filename(run_id, created_at)

        # 读取时兼容三类用户输入：
        # 1. 短 run_id，例如 46b1cb50196a；
        # 2. 新版可读文件 stem，例如 20260730T040135Z_46b1cb50196a；
        # 3. 完整文件名，例如 20260730T040135Z_46b1cb50196a_replay.json。
        direct_path = self.run_dir / run_id
        if direct_path.exists():
            return direct_path

        readable_stem_path = self.run_dir / f"{run_id}_replay.json"
        if readable_stem_path.exists():
            return readable_stem_path

        legacy_path = self.run_dir / f"{run_id}.json"
        if legacy_path.exists():
            return legacy_path

        matches = sorted(self.run_dir.glob(f"*_{run_id}_replay.json"))
        if matches:
            return matches[-1]
        return legacy_path

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

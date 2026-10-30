"""Compact tool-call ledger with optional SQLite-backed restart idempotency."""

from __future__ import annotations

from typing import Protocol
from typing import Any

from pydantic import BaseModel, Field

from deepresearch.tools.models import ToolCallRecord


class ToolCallRecordStore(Protocol):
    """Persistence contract that stores audit metadata, never raw arguments/results."""

    def save_tool_call_record(self, record: ToolCallRecord) -> None: ...

    def find_completed_tool_call(
        self,
        *,
        run_id: str,
        action_signature: str,
    ) -> ToolCallRecord | None: ...


class PersistedToolResult(BaseModel):
    """Safe placeholder when a restart can recover metadata but not a full result body."""

    persisted_result_unavailable: bool = True
    original_call_id: str
    result_summary: dict[str, Any] = Field(default_factory=dict)


class ToolCallLedger:
    """Keep compact audit records and private runtime results separately."""

    def __init__(self, persistent_store: ToolCallRecordStore | None = None) -> None:
        self._records: list[ToolCallRecord] = []
        self._completed: dict[tuple[str, str], tuple[ToolCallRecord, Any]] = {}
        self._persistent_store = persistent_store

    def record(self, record: ToolCallRecord, *, result: Any = None) -> None:
        self._records.append(record)
        if self._persistent_store is not None:
            self._persistent_store.save_tool_call_record(record)
        if record.status == "DONE":
            self._completed[(record.run_id, record.action_signature)] = (record, result)

    def reusable(
        self,
        *,
        run_id: str,
        action_signature: str,
    ) -> tuple[ToolCallRecord, Any] | None:
        in_memory = self._completed.get((run_id, action_signature))
        if in_memory is not None:
            return in_memory
        if self._persistent_store is None:
            return None
        record = self._persistent_store.find_completed_tool_call(
            run_id=run_id,
            action_signature=action_signature,
        )
        if record is None:
            return None
        return (
            record,
            PersistedToolResult(
                original_call_id=record.call_id,
                result_summary=record.result_summary,
            ),
        )

    def records(self) -> list[ToolCallRecord]:
        return list(self._records)

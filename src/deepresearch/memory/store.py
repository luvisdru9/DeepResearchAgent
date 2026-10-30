"""SQLite-backed task state and Evidence Store for local reproducible runs."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from deepresearch.schema import (
    Evidence,
    NodeTrace,
    ResearchNodeResult,
    ResearchRunStatus,
    ResearchState,
    ResearchStopReason,
    ResearchTask,
    SubTask,
    utc_now,
)
from deepresearch.tools.approvals import ApprovalRequest, ApprovalStatus
from deepresearch.tools.models import ToolRiskLevel
from deepresearch.tools.models import ToolCallRecord


@dataclass(frozen=True, slots=True)
class StoredEvidence:
    """Evidence plus the originating run needed for cross-run provenance."""

    run_id: str
    evidence: Evidence
    task_query: str
    domain: str | None
    created_at: str


class SQLiteMemoryStore:
    """Persist compact run state using one short-lived SQLite connection per operation."""

    SCHEMA_VERSION = "1.0"

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS research_runs (
                    run_id TEXT PRIMARY KEY,
                    query TEXT NOT NULL,
                    status TEXT NOT NULL,
                    task_json TEXT NOT NULL,
                    replay_path TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    schema_version TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS node_states (
                    run_id TEXT NOT NULL,
                    subtask_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    subtask_json TEXT NOT NULL,
                    trace_json TEXT,
                    result_json TEXT,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (run_id, subtask_id),
                    FOREIGN KEY (run_id) REFERENCES research_runs(run_id)
                );
                CREATE TABLE IF NOT EXISTS evidence (
                    run_id TEXT NOT NULL,
                    evidence_id TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    source_url TEXT NOT NULL,
                    extraction_method TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    evidence_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (run_id, evidence_id),
                    FOREIGN KEY (run_id) REFERENCES research_runs(run_id)
                );
                CREATE INDEX IF NOT EXISTS idx_evidence_run_source
                    ON evidence(run_id, source_id);
                CREATE TABLE IF NOT EXISTS evidence_vectors (
                    run_id TEXT NOT NULL,
                    evidence_id TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    version TEXT NOT NULL,
                    dimensions INTEGER NOT NULL,
                    content_hash TEXT NOT NULL,
                    vector BLOB NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (run_id, evidence_id, provider, model, version)
                );
                CREATE TABLE IF NOT EXISTS research_states (
                    run_id TEXT PRIMARY KEY,
                    state_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    schema_version TEXT NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES research_runs(run_id)
                );
                CREATE TABLE IF NOT EXISTS approval_requests (
                    approval_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    action_signature TEXT NOT NULL,
                    status TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE (run_id, action_signature)
                );
                CREATE INDEX IF NOT EXISTS idx_approval_requests_run_status
                    ON approval_requests(run_id, status);
                CREATE TABLE IF NOT EXISTS tool_call_records (
                    call_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    action_signature TEXT NOT NULL,
                    status TEXT NOT NULL,
                    record_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_tool_call_records_action
                    ON tool_call_records(run_id, action_signature, status);
                """
            )

    def save_tool_call_record(self, record: ToolCallRecord) -> None:
        """Persist compact audit metadata without raw arguments or result bodies."""

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO tool_call_records
                    (call_id, run_id, action_signature, status, record_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(call_id) DO UPDATE SET
                    status=excluded.status,
                    record_json=excluded.record_json
                """,
                (
                    record.call_id,
                    record.run_id,
                    record.action_signature,
                    record.status,
                    record.model_dump_json(),
                    record.started_at.isoformat(),
                ),
            )

    def find_completed_tool_call(
        self,
        *,
        run_id: str,
        action_signature: str,
    ) -> ToolCallRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT record_json FROM tool_call_records
                WHERE run_id=? AND action_signature=? AND status='DONE'
                ORDER BY created_at DESC LIMIT 1
                """,
                (run_id, action_signature),
            ).fetchone()
        return ToolCallRecord.model_validate_json(row[0]) if row is not None else None

    def list_tool_call_records(
        self,
        *,
        run_id: str,
        limit: int = 100,
    ) -> list[ToolCallRecord]:
        bounded_limit = max(1, min(limit, 500))
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT record_json FROM tool_call_records
                WHERE run_id=? ORDER BY created_at, rowid LIMIT ?
                """,
                (run_id, bounded_limit),
            ).fetchall()
        return [ToolCallRecord.model_validate_json(row[0]) for row in rows]

    def request_approval(
        self,
        *,
        run_id: str,
        agent: str,
        tool_name: str,
        action_signature: str,
        argument_keys: list[str],
        risk_level: ToolRiskLevel,
        reason: str,
        expires_in_seconds: float | None = None,
    ) -> ApprovalRequest:
        """Create or return the stable request for one run-scoped action."""

        created_at = utc_now()
        request = ApprovalRequest(
            run_id=run_id,
            agent=agent,
            tool_name=tool_name,
            action_signature=action_signature,
            argument_keys=sorted(argument_keys),
            risk_level=risk_level,
            reason=reason,
            created_at=created_at,
            expires_at=(
                created_at + timedelta(seconds=expires_in_seconds)
                if expires_in_seconds is not None
                else None
            ),
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO approval_requests
                    (approval_id, run_id, action_signature, status, request_json,
                     created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    request.approval_id,
                    request.run_id,
                    request.action_signature,
                    request.status.value,
                    request.model_dump_json(),
                    request.created_at.isoformat(),
                    created_at.isoformat(),
                ),
            )
            row = connection.execute(
                """
                SELECT request_json FROM approval_requests
                WHERE run_id=? AND action_signature=?
                """,
                (run_id, action_signature),
            ).fetchone()
        if row is None:  # Defensive: INSERT/SELECT share one transaction.
            raise RuntimeError("approval request could not be persisted")
        return self._expire_approval_if_needed(ApprovalRequest.model_validate_json(row[0]))

    def get_approval(self, approval_id: str) -> ApprovalRequest | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT request_json FROM approval_requests WHERE approval_id=?",
                (approval_id,),
            ).fetchone()
        if row is None:
            return None
        return self._expire_approval_if_needed(ApprovalRequest.model_validate_json(row[0]))

    def get_approval_for_action(
        self,
        *,
        run_id: str,
        action_signature: str,
    ) -> ApprovalRequest | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT request_json FROM approval_requests
                WHERE run_id=? AND action_signature=?
                """,
                (run_id, action_signature),
            ).fetchone()
        if row is None:
            return None
        return self._expire_approval_if_needed(ApprovalRequest.model_validate_json(row[0]))

    def list_approvals(
        self,
        *,
        run_id: str | None = None,
        status: ApprovalStatus | None = None,
        limit: int = 100,
    ) -> list[ApprovalRequest]:
        """List compact approval metadata for the local operator UI/API."""

        bounded_limit = max(1, min(limit, 500))
        clauses: list[str] = []
        parameters: list[Any] = []
        if run_id is not None:
            clauses.append("run_id=?")
            parameters.append(run_id)
        if status is not None:
            clauses.append("status=?")
            parameters.append(status.value)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        parameters.append(bounded_limit)
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT request_json FROM approval_requests{where} "  # noqa: S608 - clauses are fixed.
                "ORDER BY created_at DESC LIMIT ?",
                tuple(parameters),
            ).fetchall()
        return [
            self._expire_approval_if_needed(ApprovalRequest.model_validate_json(row[0]))
            for row in rows
        ]

    def decide_approval(
        self,
        approval_id: str,
        *,
        decision: ApprovalStatus,
        decision_key: str,
        decided_by: str = "local_operator",
        reason: str | None = None,
    ) -> ApprovalRequest:
        """Approve/reject once; an identical idempotency key may safely repeat."""

        if decision not in {ApprovalStatus.APPROVED, ApprovalStatus.REJECTED}:
            raise ValueError("decision must be APPROVED or REJECTED")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            request = self._load_approval_in_connection(connection, approval_id)
            if request is None:
                raise KeyError(approval_id)
            request = self._expire_approval_in_connection(connection, request)
            if request.status == decision and request.decision_key == decision_key:
                return request
            if request.status != ApprovalStatus.PENDING:
                raise ValueError(f"approval is already {request.status.value}")
            updated = request.model_copy(
                update={
                    "status": decision,
                    "decided_at": utc_now(),
                    "decided_by": decided_by,
                    "decision_reason": reason,
                    "decision_key": decision_key,
                }
            )
            self._save_approval_in_connection(connection, updated)
            return updated

    def claim_approval_execution(self, approval_id: str) -> ApprovalRequest | None:
        """Atomically reserve an approved side effect so concurrent calls cannot duplicate it."""

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            request = self._load_approval_in_connection(connection, approval_id)
            if request is None:
                return None
            request = self._expire_approval_in_connection(connection, request)
            if request.status != ApprovalStatus.APPROVED:
                return None
            updated = request.model_copy(update={"status": ApprovalStatus.EXECUTING})
            self._save_approval_in_connection(connection, updated)
            return updated

    def mark_approval_executed(
        self,
        approval_id: str,
        *,
        call_id: str,
    ) -> ApprovalRequest:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            request = self._load_approval_in_connection(connection, approval_id)
            if request is None:
                raise KeyError(approval_id)
            if request.status == ApprovalStatus.EXECUTED:
                if request.execution_call_id != call_id:
                    raise ValueError("approval already executed by another call")
                return request
            if request.status != ApprovalStatus.EXECUTING:
                raise ValueError(f"approval is not executing: {request.status.value}")
            updated = request.model_copy(
                update={
                    "status": ApprovalStatus.EXECUTED,
                    "execution_call_id": call_id,
                }
            )
            self._save_approval_in_connection(connection, updated)
            return updated

    def mark_approval_failed(self, approval_id: str) -> ApprovalRequest:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            request = self._load_approval_in_connection(connection, approval_id)
            if request is None:
                raise KeyError(approval_id)
            if request.status == ApprovalStatus.FAILED:
                return request
            if request.status != ApprovalStatus.EXECUTING:
                raise ValueError(f"approval is not executing: {request.status.value}")
            updated = request.model_copy(update={"status": ApprovalStatus.FAILED})
            self._save_approval_in_connection(connection, updated)
            return updated

    def _expire_approval_if_needed(self, request: ApprovalRequest) -> ApprovalRequest:
        if request.status != ApprovalStatus.PENDING or not request.is_expired():
            return request
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            latest = self._load_approval_in_connection(connection, request.approval_id)
            if latest is None:
                return request
            return self._expire_approval_in_connection(connection, latest)

    @staticmethod
    def _load_approval_in_connection(
        connection: sqlite3.Connection,
        approval_id: str,
    ) -> ApprovalRequest | None:
        row = connection.execute(
            "SELECT request_json FROM approval_requests WHERE approval_id=?",
            (approval_id,),
        ).fetchone()
        return ApprovalRequest.model_validate_json(row[0]) if row is not None else None

    @staticmethod
    def _save_approval_in_connection(
        connection: sqlite3.Connection,
        request: ApprovalRequest,
    ) -> None:
        connection.execute(
            """
            UPDATE approval_requests
            SET status=?, request_json=?, updated_at=?
            WHERE approval_id=?
            """,
            (
                request.status.value,
                request.model_dump_json(),
                utc_now().isoformat(),
                request.approval_id,
            ),
        )

    def _expire_approval_in_connection(
        self,
        connection: sqlite3.Connection,
        request: ApprovalRequest,
    ) -> ApprovalRequest:
        if request.status == ApprovalStatus.PENDING and request.is_expired():
            request = request.model_copy(update={"status": ApprovalStatus.EXPIRED})
            self._save_approval_in_connection(connection, request)
        return request

    def backup_to(self, destination: str | Path) -> Path:
        """Create a transactionally consistent experiment snapshot of this SQLite store."""

        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as source, sqlite3.connect(target) as destination_connection:
            source.backup(destination_connection)
        return target

    def start_run(self, task: ResearchTask) -> None:
        now = utc_now().isoformat()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO research_runs
                    (run_id, query, status, task_json, created_at, updated_at, schema_version)
                VALUES (?, ?, 'RUNNING', ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    query=excluded.query,
                    status='RUNNING',
                    task_json=excluded.task_json,
                    error=NULL,
                    updated_at=excluded.updated_at
                """,
                (
                    task.run_id,
                    task.query,
                    task.model_dump_json(),
                    task.created_at.isoformat(),
                    now,
                    self.SCHEMA_VERSION,
                ),
            )

    def save_node(
        self,
        run_id: str,
        subtask: SubTask,
        trace: NodeTrace | None,
        result: ResearchNodeResult | None,
    ) -> None:
        status = trace.status.value if trace else subtask.status.value
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO node_states
                    (run_id, subtask_id, status, subtask_json, trace_json, result_json, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, subtask_id) DO UPDATE SET
                    status=excluded.status,
                    subtask_json=excluded.subtask_json,
                    trace_json=excluded.trace_json,
                    result_json=excluded.result_json,
                    updated_at=excluded.updated_at
                """,
                (
                    run_id,
                    subtask.id,
                    status,
                    subtask.model_dump_json(),
                    trace.model_dump_json() if trace else None,
                    result.model_dump_json() if result else None,
                    utc_now().isoformat(),
                ),
            )

    def save_evidence(self, run_id: str, evidence: list[Evidence]) -> None:
        rows = [
            (
                run_id,
                item.id,
                item.source_id,
                item.source_url,
                item.extraction_method,
                item.confidence,
                item.model_dump_json(),
                utc_now().isoformat(),
            )
            for item in evidence
        ]
        if not rows:
            return
        with self._connect() as connection:
            connection.executemany(
                """
                INSERT INTO evidence
                    (run_id, evidence_id, source_id, source_url, extraction_method,
                     confidence, evidence_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, evidence_id) DO UPDATE SET
                    source_id=excluded.source_id,
                    source_url=excluded.source_url,
                    extraction_method=excluded.extraction_method,
                    confidence=excluded.confidence,
                    evidence_json=excluded.evidence_json
                """,
                rows,
            )

    def save_research_state(self, state: ResearchState) -> None:
        """Upsert the unified controller state without changing legacy run rows."""

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO research_states (run_id, state_json, updated_at, schema_version)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    state_json=excluded.state_json,
                    updated_at=excluded.updated_at,
                    schema_version=excluded.schema_version
                """,
                (
                    state.run_id,
                    state.model_dump_json(),
                    state.updated_at.isoformat(),
                    state.schema_version,
                ),
            )

    def load_research_state(self, run_id: str) -> ResearchState | None:
        """Load the latest controller state for replay, inspection, or future resume."""

        with self._connect() as connection:
            row = connection.execute(
                "SELECT state_json FROM research_states WHERE run_id=?", (run_id,)
            ).fetchone()
        if row is None:
            return None
        return ResearchState.model_validate(json.loads(row["state_json"]))

    def finish_run(self, run_id: str, *, replay_path: str | Path) -> None:
        self._set_run_status(run_id, "DONE", replay_path=str(replay_path))

    def fail_run(self, run_id: str, error: str) -> None:
        # 错误只保存异常类型/消息；调用方不应把请求头或密钥传入。
        self._set_run_status(run_id, "FAILED", error=error[:2000])

    def interrupt_run(self, run_id: str, error: str) -> None:
        """Mark one active run resumable after API shutdown or process loss."""

        safe_error = error[:2000]
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE research_runs
                SET status='INTERRUPTED', error=?, updated_at=?
                WHERE run_id=? AND status='RUNNING'
                """,
                (safe_error, utc_now().isoformat(), run_id),
            )
        if cursor.rowcount:
            self._interrupt_research_state(run_id, safe_error)

    def interrupt_running_runs(self, error: str) -> list[str]:
        """Recover stale RUNNING rows left by a previous local API process."""

        with self._connect() as connection:
            rows = connection.execute(
                "SELECT run_id FROM research_runs WHERE status='RUNNING' ORDER BY created_at"
            ).fetchall()
        run_ids = [str(row["run_id"]) for row in rows]
        for run_id in run_ids:
            self.interrupt_run(run_id, error)
        return run_ids

    def _interrupt_research_state(self, run_id: str, error: str) -> None:
        state = self.load_research_state(run_id)
        if state is None or state.status != ResearchRunStatus.RUNNING:
            return
        interrupted = state.model_copy(
            update={
                "status": ResearchRunStatus.INTERRUPTED,
                "stop_reason": ResearchStopReason.INTERRUPTED,
                "last_error": error,
                "updated_at": utc_now(),
            }
        )
        self.save_research_state(interrupted)

    def _set_run_status(
        self,
        run_id: str,
        status: str,
        *,
        replay_path: str | None = None,
        error: str | None = None,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE research_runs
                SET status=?, replay_path=COALESCE(?, replay_path), error=?, updated_at=?
                WHERE run_id=?
                """,
                (status, replay_path, error, utc_now().isoformat(), run_id),
            )

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM research_runs WHERE run_id=?", (run_id,)
            ).fetchone()
        return dict(row) if row else None

    def list_runs(self, *, status: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        """List newest task rows for API polling without loading large Replay payloads."""

        bounded_limit = max(1, min(limit, 500))
        if status is None:
            statement = "SELECT * FROM research_runs ORDER BY created_at DESC LIMIT ?"
            parameters: tuple[Any, ...] = (bounded_limit,)
        else:
            statement = (
                "SELECT * FROM research_runs WHERE status=? "
                "ORDER BY created_at DESC LIMIT ?"
            )
            parameters = (status, bounded_limit)
        with self._connect() as connection:
            rows = connection.execute(statement, parameters).fetchall()
        return [dict(row) for row in rows]

    def load_task(self, run_id: str) -> ResearchTask | None:
        """Load the original request used by API resume without exposing credentials."""

        row = self.get_run(run_id)
        if row is None:
            return None
        return ResearchTask.model_validate(json.loads(row["task_json"]))

    def load_evidence(self, run_id: str) -> list[Evidence]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT evidence_json FROM evidence WHERE run_id=? ORDER BY rowid", (run_id,)
            ).fetchall()
        return [Evidence.model_validate(json.loads(row["evidence_json"])) for row in rows]

    def load_all_evidence(
        self,
        *,
        limit: int | None = None,
        domain: str | None = None,
        max_age_days: int | None = None,
        exclude_run_id: str | None = None,
    ) -> list[StoredEvidence]:
        """Load newest historical evidence for rebuilding the in-process numpy index."""

        query = """
            SELECT e.run_id, e.evidence_json, e.created_at, r.task_json
            FROM evidence AS e
            JOIN research_runs AS r ON r.run_id = e.run_id
            ORDER BY e.rowid DESC
        """
        with self._connect() as connection:
            rows = connection.execute(query).fetchall()
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=max_age_days)
            if max_age_days is not None
            else None
        )
        records: list[StoredEvidence] = []
        for row in rows:
            if exclude_run_id and row["run_id"] == exclude_run_id:
                continue
            task = ResearchTask.model_validate(json.loads(row["task_json"]))
            if domain is not None and task.domain != domain:
                continue
            if cutoff is not None:
                created_at = datetime.fromisoformat(row["created_at"])
                if created_at.tzinfo is None:
                    created_at = created_at.replace(tzinfo=timezone.utc)
                if created_at < cutoff:
                    continue
            records.append(
                StoredEvidence(
                    run_id=row["run_id"],
                    evidence=Evidence.model_validate(json.loads(row["evidence_json"])),
                    task_query=task.query,
                    domain=task.domain,
                    created_at=row["created_at"],
                )
            )
            if limit is not None and len(records) >= limit:
                break
        return records

    def load_nodes(self, run_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM node_states WHERE run_id=? ORDER BY rowid", (run_id,)
            ).fetchall()
        return [dict(row) for row in rows]

    def load_evidence_vector(
        self,
        *,
        run_id: str,
        evidence_id: str,
        provider: str,
        model: str,
        version: str,
        content_hash: str,
    ) -> tuple[int, bytes] | None:
        """Load a vector only when its embedding identity and source content still match."""

        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT dimensions, vector FROM evidence_vectors
                WHERE run_id=? AND evidence_id=? AND provider=? AND model=?
                  AND version=? AND content_hash=?
                """,
                (run_id, evidence_id, provider, model, version, content_hash),
            ).fetchone()
        return (int(row["dimensions"]), bytes(row["vector"])) if row else None

    def save_evidence_vector(
        self,
        *,
        run_id: str,
        evidence_id: str,
        provider: str,
        model: str,
        version: str,
        dimensions: int,
        content_hash: str,
        vector: bytes,
    ) -> None:
        """Upsert one float32 vector cache entry without storing external credentials."""

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO evidence_vectors
                    (run_id, evidence_id, provider, model, version, dimensions,
                     content_hash, vector, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, evidence_id, provider, model, version) DO UPDATE SET
                    dimensions=excluded.dimensions,
                    content_hash=excluded.content_hash,
                    vector=excluded.vector,
                    created_at=excluded.created_at
                """,
                (
                    run_id,
                    evidence_id,
                    provider,
                    model,
                    version,
                    dimensions,
                    content_hash,
                    vector,
                    utc_now().isoformat(),
                ),
            )

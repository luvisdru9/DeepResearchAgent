"""Approval lifecycle used by local HITL-aware tool execution.

Approval records intentionally contain only compact action metadata. Raw arguments,
credentials, source bodies, and full tool results remain outside the approval store.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from enum import StrEnum
from threading import Lock
from typing import Protocol
from uuid import uuid4

from pydantic import BaseModel, Field

from deepresearch.schema import utc_now
from deepresearch.tools.models import ToolRiskLevel


class ApprovalStatus(StrEnum):
    """Lifecycle independent from DAG node and whole-run status."""

    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    EXECUTING = "EXECUTING"
    EXECUTED = "EXECUTED"
    FAILED = "FAILED"


class ApprovalRequest(BaseModel):
    """Replay-safe request for one stable tool action."""

    approval_id: str = Field(default_factory=lambda: f"approval_{uuid4().hex[:12]}")
    run_id: str
    agent: str
    tool_name: str
    action_signature: str
    argument_keys: list[str] = Field(default_factory=list)
    risk_level: ToolRiskLevel
    reason: str
    status: ApprovalStatus = ApprovalStatus.PENDING
    created_at: datetime = Field(default_factory=utc_now)
    expires_at: datetime | None = None
    decided_at: datetime | None = None
    decided_by: str | None = None
    decision_reason: str | None = None
    decision_key: str | None = None
    execution_call_id: str | None = None

    def is_expired(self, *, now: datetime | None = None) -> bool:
        return self.expires_at is not None and self.expires_at <= (now or utc_now())


class ApprovalStore(Protocol):
    """Storage contract shared by in-memory tests and SQLite persistence."""

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
    ) -> ApprovalRequest: ...

    def get_approval_for_action(
        self,
        *,
        run_id: str,
        action_signature: str,
    ) -> ApprovalRequest | None: ...

    def claim_approval_execution(self, approval_id: str) -> ApprovalRequest | None: ...

    def mark_approval_executed(
        self,
        approval_id: str,
        *,
        call_id: str,
    ) -> ApprovalRequest: ...

    def mark_approval_failed(self, approval_id: str) -> ApprovalRequest: ...


class InMemoryApprovalStore:
    """Thread-safe local implementation used by isolated ToolExecutor workflows."""

    def __init__(self) -> None:
        self._by_id: dict[str, ApprovalRequest] = {}
        self._by_action: dict[tuple[str, str], str] = {}
        self._lock = Lock()

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
        with self._lock:
            key = (run_id, action_signature)
            existing_id = self._by_action.get(key)
            if existing_id is not None:
                return self._expire_if_needed(self._by_id[existing_id])
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
            self._by_id[request.approval_id] = request
            self._by_action[key] = request.approval_id
            return request

    def get_approval(self, approval_id: str) -> ApprovalRequest | None:
        with self._lock:
            request = self._by_id.get(approval_id)
            return self._expire_if_needed(request) if request is not None else None

    def get_approval_for_action(
        self,
        *,
        run_id: str,
        action_signature: str,
    ) -> ApprovalRequest | None:
        with self._lock:
            approval_id = self._by_action.get((run_id, action_signature))
            if approval_id is None:
                return None
            return self._expire_if_needed(self._by_id[approval_id])

    def list_approvals(
        self,
        *,
        run_id: str | None = None,
        status: ApprovalStatus | None = None,
    ) -> list[ApprovalRequest]:
        with self._lock:
            requests = [self._expire_if_needed(item) for item in self._by_id.values()]
        return [
            item
            for item in requests
            if (run_id is None or item.run_id == run_id)
            and (status is None or item.status == status)
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
        if decision not in {ApprovalStatus.APPROVED, ApprovalStatus.REJECTED}:
            raise ValueError("decision must be APPROVED or REJECTED")
        with self._lock:
            request = self._require(approval_id)
            request = self._expire_if_needed(request)
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
            self._by_id[approval_id] = updated
            return updated

    def claim_approval_execution(self, approval_id: str) -> ApprovalRequest | None:
        with self._lock:
            request = self._by_id.get(approval_id)
            if request is None:
                return None
            request = self._expire_if_needed(request)
            if request.status != ApprovalStatus.APPROVED:
                return None
            updated = request.model_copy(update={"status": ApprovalStatus.EXECUTING})
            self._by_id[approval_id] = updated
            return updated

    def mark_approval_executed(
        self,
        approval_id: str,
        *,
        call_id: str,
    ) -> ApprovalRequest:
        with self._lock:
            request = self._require(approval_id)
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
            self._by_id[approval_id] = updated
            return updated

    def mark_approval_failed(self, approval_id: str) -> ApprovalRequest:
        with self._lock:
            request = self._require(approval_id)
            if request.status == ApprovalStatus.FAILED:
                return request
            if request.status != ApprovalStatus.EXECUTING:
                raise ValueError(f"approval is not executing: {request.status.value}")
            updated = request.model_copy(update={"status": ApprovalStatus.FAILED})
            self._by_id[approval_id] = updated
            return updated

    def _require(self, approval_id: str) -> ApprovalRequest:
        request = self._by_id.get(approval_id)
        if request is None:
            raise KeyError(approval_id)
        return request

    def _expire_if_needed(self, request: ApprovalRequest) -> ApprovalRequest:
        if request.status == ApprovalStatus.PENDING and request.is_expired():
            request = request.model_copy(update={"status": ApprovalStatus.EXPIRED})
            self._by_id[request.approval_id] = request
        return request

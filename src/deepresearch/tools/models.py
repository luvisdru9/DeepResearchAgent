"""Provider-neutral models for controlled tool registration and audit."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from deepresearch.schema import utc_now


ToolRiskLevel = Literal["read_only", "write", "high_risk"]
ToolCallStatus = Literal[
    "DONE",
    "DENIED",
    "PENDING_APPROVAL",
    "INVALID",
    "FAILED",
    "TIMEOUT",
    "REUSED",
]


class ToolSpec(BaseModel):
    """Serializable contract exposed to an Agent before it proposes a tool call."""

    name: str = Field(pattern=r"^[a-z][a-z0-9_]{1,63}$")
    description: str = Field(min_length=1, max_length=500)
    input_schema: dict[str, Any]
    risk_level: ToolRiskLevel = "read_only"
    allowed_agents: list[str] = Field(default_factory=list)
    timeout_seconds: float = Field(default=30.0, gt=0.0, le=300.0)
    idempotent: bool = True
    requires_approval: bool = False


class ToolPolicyDecision(BaseModel):
    """Deterministic authorization result; no model output can override it."""

    allowed: bool
    reason: str
    requires_approval: bool = False


class ToolCallRecord(BaseModel):
    """Replay-safe ledger entry that excludes raw arguments and full tool results."""

    call_id: str = Field(default_factory=lambda: f"tool_{uuid4().hex[:12]}")
    run_id: str
    agent: str
    tool_name: str
    risk_level: ToolRiskLevel
    status: ToolCallStatus
    action_signature: str
    argument_keys: list[str] = Field(default_factory=list)
    result_summary: dict[str, Any] = Field(default_factory=dict)
    policy_reason: str | None = None
    approval_id: str | None = None
    error_type: str | None = None
    reused_from_call_id: str | None = None
    latency_seconds: float = Field(default=0.0, ge=0.0)
    started_at: datetime = Field(default_factory=utc_now)
    finished_at: datetime = Field(default_factory=utc_now)


@dataclass(frozen=True, slots=True)
class ToolExecutionOutcome:
    """Runtime result returned to deterministic orchestration code."""

    result: Any
    record: ToolCallRecord

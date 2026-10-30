"""Deterministic least-privilege policy for proposed tool calls."""

from __future__ import annotations

from collections.abc import Iterable

from deepresearch.tools.approvals import ApprovalRequest, ApprovalStatus
from deepresearch.tools.models import ToolPolicyDecision, ToolRiskLevel, ToolSpec


class ToolPolicy:
    """Authorize by local role/risk rules; LLM text is never treated as approval."""

    def __init__(
        self,
        *,
        allowed_risk_levels: Iterable[ToolRiskLevel] = ("read_only",),
        approvals_enabled: bool = False,
    ) -> None:
        self.allowed_risk_levels = set(allowed_risk_levels)
        self.approvals_enabled = approvals_enabled

    def evaluate(
        self,
        spec: ToolSpec,
        *,
        agent: str,
        approval: ApprovalRequest | None = None,
    ) -> ToolPolicyDecision:
        if agent not in spec.allowed_agents:
            return ToolPolicyDecision(allowed=False, reason="agent_not_allowed")
        if spec.risk_level not in self.allowed_risk_levels:
            return ToolPolicyDecision(allowed=False, reason="risk_level_not_allowed")
        approval_required = spec.requires_approval or spec.risk_level == "high_risk"
        if approval_required:
            if approval is not None and approval.status == ApprovalStatus.APPROVED:
                return ToolPolicyDecision(allowed=True, reason="approved")
            if approval is not None and approval.status in {
                ApprovalStatus.EXECUTING,
                ApprovalStatus.EXECUTED,
            }:
                return ToolPolicyDecision(allowed=False, reason="action_already_claimed")
            return ToolPolicyDecision(
                allowed=False,
                reason=(
                    "approval_flow_not_implemented"
                    if not self.approvals_enabled
                    else (
                        f"approval_{approval.status.value.lower()}"
                        if approval is not None
                        else "approval_required"
                    )
                ),
                requires_approval=True,
            )
        return ToolPolicyDecision(allowed=True, reason="allowed")

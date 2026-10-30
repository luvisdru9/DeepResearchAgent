"""Policy-enforced, schema-validated, timeout-bounded tool execution."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from datetime import datetime
from typing import Any

from pydantic import ValidationError

from deepresearch.schema import utc_now
from deepresearch.tools.approvals import ApprovalStatus, ApprovalStore
from deepresearch.tools.errors import (
    ToolApprovalRequiredError,
    ToolExecutionError,
    ToolInputValidationError,
    ToolPolicyDeniedError,
    ToolTimeoutError,
)
from deepresearch.tools.ledger import ToolCallLedger
from deepresearch.tools.models import ToolCallRecord, ToolCallStatus, ToolExecutionOutcome
from deepresearch.tools.policy import ToolPolicy
from deepresearch.tools.registry import RegisteredTool, ToolRegistry


class ToolExecutor:
    """Execute only registered and locally authorized actions."""

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        policy: ToolPolicy | None = None,
        ledger: ToolCallLedger | None = None,
        approval_store: ApprovalStore | None = None,
        approval_ttl_seconds: float | None = 3600.0,
    ) -> None:
        self.registry = registry
        self.policy = policy or ToolPolicy()
        self.ledger = ledger or ToolCallLedger()
        self.approval_store = approval_store
        self.approval_ttl_seconds = approval_ttl_seconds

    async def execute(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        agent: str,
        run_id: str,
    ) -> ToolExecutionOutcome:
        tool = self.registry.get(name)
        started_at = utc_now()
        started = time.perf_counter()
        argument_keys = sorted(str(key) for key in arguments)
        pre_validation_signature = self._signature(name, agent, arguments)

        decision = self.policy.evaluate(tool.spec, agent=agent)
        # Unauthorized roles/risks fail before validation. Approval-required calls
        # continue only far enough to validate and hash compact action metadata.
        if not decision.allowed and not decision.requires_approval:
            record = self._record(
                tool,
                run_id=run_id,
                agent=agent,
                status="DENIED",
                signature=pre_validation_signature,
                argument_keys=argument_keys,
                started_at=started_at,
                started=started,
                policy_reason=decision.reason,
            )
            raise ToolPolicyDeniedError(
                f"Tool '{name}' denied for agent '{agent}': {record.policy_reason}."
            )

        try:
            payload = tool.input_model.model_validate(arguments)
        except ValidationError as exc:
            self._record(
                tool,
                run_id=run_id,
                agent=agent,
                status="INVALID",
                signature=pre_validation_signature,
                argument_keys=argument_keys,
                started_at=started_at,
                started=started,
                error_type=type(exc).__name__,
            )
            raise ToolInputValidationError(
                f"Arguments for tool '{name}' failed schema validation."
            ) from exc

        signature = self._signature(
            name,
            agent,
            payload.model_dump(mode="json"),
        )
        if tool.spec.idempotent:
            reusable = self.ledger.reusable(
                run_id=run_id,
                action_signature=signature,
            )
            if reusable is not None:
                original, result = reusable
                # A crash may occur after the DONE ledger write but before approval state
                # reaches EXECUTED. Reconciliation uses the original call id and never
                # re-runs the side effect.
                if original.approval_id and self.approval_store is not None:
                    try:
                        self.approval_store.mark_approval_executed(
                            original.approval_id,
                            call_id=original.call_id,
                        )
                    except ValueError:
                        pass
                record = self._record(
                    tool,
                    run_id=run_id,
                    agent=agent,
                    status="REUSED",
                    signature=signature,
                    argument_keys=argument_keys,
                    started_at=started_at,
                    started=started,
                    result_summary=original.result_summary,
                    reused_from_call_id=original.call_id,
                    approval_id=original.approval_id,
                )
                return ToolExecutionOutcome(result=result, record=record)

        approval_id: str | None = None
        if decision.requires_approval:
            if self.approval_store is None or not self.policy.approvals_enabled:
                record = self._record(
                    tool,
                    run_id=run_id,
                    agent=agent,
                    status="DENIED",
                    signature=signature,
                    argument_keys=argument_keys,
                    started_at=started_at,
                    started=started,
                    policy_reason="approval_flow_not_implemented",
                )
                raise ToolPolicyDeniedError(
                    f"Tool '{name}' denied for agent '{agent}': {record.policy_reason}."
                )

            approval = self.approval_store.get_approval_for_action(
                run_id=run_id,
                action_signature=signature,
            )
            if approval is None:
                approval = self.approval_store.request_approval(
                    run_id=run_id,
                    agent=agent,
                    tool_name=name,
                    action_signature=signature,
                    argument_keys=argument_keys,
                    risk_level=tool.spec.risk_level,
                    reason="tool_requires_operator_approval",
                    expires_in_seconds=self.approval_ttl_seconds,
                )
            approval_id = approval.approval_id
            decision = self.policy.evaluate(tool.spec, agent=agent, approval=approval)
            if not decision.allowed:
                record_status: ToolCallStatus = (
                    "PENDING_APPROVAL"
                    if approval.status == ApprovalStatus.PENDING
                    else "DENIED"
                )
                record = self._record(
                    tool,
                    run_id=run_id,
                    agent=agent,
                    status=record_status,
                    signature=signature,
                    argument_keys=argument_keys,
                    started_at=started_at,
                    started=started,
                    policy_reason=decision.reason,
                    approval_id=approval_id,
                )
                if approval.status == ApprovalStatus.PENDING:
                    raise ToolApprovalRequiredError(
                        f"Tool '{name}' is waiting for local approval '{approval_id}'.",
                        approval_id=approval_id,
                    )
                raise ToolPolicyDeniedError(
                    f"Tool '{name}' denied for agent '{agent}': {record.policy_reason}."
                )

            claimed = self.approval_store.claim_approval_execution(approval_id)
            if claimed is None:
                record = self._record(
                    tool,
                    run_id=run_id,
                    agent=agent,
                    status="DENIED",
                    signature=signature,
                    argument_keys=argument_keys,
                    started_at=started_at,
                    started=started,
                    policy_reason="approval_execution_already_claimed",
                    approval_id=approval_id,
                )
                raise ToolPolicyDeniedError(
                    f"Tool '{name}' denied for agent '{agent}': {record.policy_reason}."
                )

        try:
            result = await asyncio.wait_for(
                tool.handler(payload),
                timeout=tool.spec.timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            if approval_id is not None and self.approval_store is not None:
                self.approval_store.mark_approval_failed(approval_id)
            self._record(
                tool,
                run_id=run_id,
                agent=agent,
                status="TIMEOUT",
                signature=signature,
                argument_keys=argument_keys,
                started_at=started_at,
                started=started,
                error_type="ToolTimeoutError",
                approval_id=approval_id,
            )
            raise ToolTimeoutError(
                f"Tool '{name}' exceeded {tool.spec.timeout_seconds:.3f}s timeout."
            ) from exc
        except Exception as exc:  # noqa: BLE001 - adapters normalize provider-specific failures.
            if approval_id is not None and self.approval_store is not None:
                self.approval_store.mark_approval_failed(approval_id)
            self._record(
                tool,
                run_id=run_id,
                agent=agent,
                status="FAILED",
                signature=signature,
                argument_keys=argument_keys,
                started_at=started_at,
                started=started,
                error_type=type(exc).__name__,
                approval_id=approval_id,
            )
            raise ToolExecutionError(
                f"Tool '{name}' failed with {type(exc).__name__}."
            ) from exc

        record = self._record(
            tool,
            run_id=run_id,
            agent=agent,
            status="DONE",
            signature=signature,
            argument_keys=argument_keys,
            started_at=started_at,
            started=started,
            result_summary=tool.summarize_result(result),
            result=result,
            approval_id=approval_id,
        )
        if approval_id is not None and self.approval_store is not None:
            self.approval_store.mark_approval_executed(
                approval_id,
                call_id=record.call_id,
            )
        return ToolExecutionOutcome(result=result, record=record)

    def _record(
        self,
        tool: RegisteredTool,
        *,
        run_id: str,
        agent: str,
        status: ToolCallStatus,
        signature: str,
        argument_keys: list[str],
        started_at: datetime,
        started: float,
        result_summary: dict[str, Any] | None = None,
        policy_reason: str | None = None,
        error_type: str | None = None,
        reused_from_call_id: str | None = None,
        approval_id: str | None = None,
        result: Any = None,
    ) -> ToolCallRecord:
        record = ToolCallRecord(
            run_id=run_id,
            agent=agent,
            tool_name=tool.spec.name,
            risk_level=tool.spec.risk_level,
            status=status,
            action_signature=signature,
            argument_keys=argument_keys,
            result_summary=result_summary or {},
            policy_reason=policy_reason,
            error_type=error_type,
            reused_from_call_id=reused_from_call_id,
            approval_id=approval_id,
            latency_seconds=time.perf_counter() - started,
            started_at=started_at,
            finished_at=utc_now(),
        )
        self.ledger.record(record, result=result)
        return record

    @staticmethod
    def _signature(name: str, agent: str, arguments: dict[str, Any]) -> str:
        canonical = json.dumps(
            {"tool": name, "agent": agent, "arguments": arguments},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

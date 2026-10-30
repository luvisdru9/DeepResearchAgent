import asyncio

import pytest
from pydantic import BaseModel

from deepresearch.retrieval import SearchProvider
from deepresearch.schema import SourceDoc
from deepresearch.tools import (
    ApprovalStatus,
    InMemoryApprovalStore,
    PersistedToolResult,
    RegisteredTool,
    SearchToolResult,
    ToolAlreadyRegisteredError,
    ToolApprovalRequiredError,
    ToolExecutor,
    ToolCallLedger,
    ToolInputValidationError,
    ToolNotFoundError,
    ToolPolicyDeniedError,
    ToolRegistry,
    ToolSpec,
    ToolPolicy,
    ToolTimeoutError,
    register_search_tool,
)


class CountingSearchProvider(SearchProvider):
    def __init__(self) -> None:
        self.calls = 0

    def search(self, query: str, top_k: int = 5) -> list[SourceDoc]:
        self.calls += 1
        return [
            SourceDoc(
                id="doc_1",
                title="Evidence source",
                url="local://evidence.md",
                content="Sensitive full body must not be copied into the ledger summary.",
            )
        ][:top_k]


def _search_executor() -> tuple[ToolExecutor, CountingSearchProvider]:
    provider = CountingSearchProvider()
    registry = ToolRegistry()
    register_search_tool(registry, provider)
    return ToolExecutor(registry), provider


def test_registry_rejects_duplicate_names_and_filters_specs_by_agent():
    registry = ToolRegistry()
    provider = CountingSearchProvider()
    spec = register_search_tool(registry, provider)

    assert [item.name for item in registry.specs_for_agent("retriever")] == [spec.name]
    assert registry.specs_for_agent("writer") == []
    with pytest.raises(ToolAlreadyRegisteredError):
        register_search_tool(registry, provider)


def test_unknown_tool_fails_closed():
    executor = ToolExecutor(ToolRegistry())

    with pytest.raises(ToolNotFoundError):
        asyncio.run(
            executor.execute("missing_tool", {}, agent="retriever", run_id="run_1")
        )


def test_policy_denies_unauthorized_agent_without_calling_provider():
    executor, provider = _search_executor()

    with pytest.raises(ToolPolicyDeniedError, match="agent_not_allowed"):
        asyncio.run(
            executor.execute(
                "search_sources",
                {"query": "evidence"},
                agent="writer",
                run_id="run_1",
            )
        )

    assert provider.calls == 0
    record = executor.ledger.records()[0]
    assert record.status == "DENIED"
    assert record.argument_keys == ["query"]
    assert "evidence" not in record.model_dump_json()


def test_invalid_search_arguments_are_rejected_before_provider_call():
    executor, provider = _search_executor()

    with pytest.raises(ToolInputValidationError):
        asyncio.run(
            executor.execute(
                "search_sources",
                {"query": "evidence", "top_k": 0},
                agent="retriever",
                run_id="run_1",
            )
        )

    assert provider.calls == 0
    assert executor.ledger.records()[0].status == "INVALID"


def test_read_only_search_executes_and_ledger_omits_full_document_content():
    executor, provider = _search_executor()

    outcome = asyncio.run(
        executor.execute(
            "search_sources",
            {"query": "evidence", "top_k": 1},
            agent="retriever",
            run_id="run_1",
        )
    )

    assert provider.calls == 1
    assert isinstance(outcome.result, SearchToolResult)
    assert outcome.record.status == "DONE"
    assert outcome.record.result_summary["source_ids"] == ["doc_1"]
    assert "Sensitive full body" not in outcome.record.model_dump_json()


def test_idempotent_search_reuses_result_only_within_same_run():
    executor, provider = _search_executor()
    arguments = {"query": "evidence", "top_k": 1}

    first = asyncio.run(
        executor.execute(
            "search_sources", arguments, agent="retriever", run_id="run_1"
        )
    )
    reused = asyncio.run(
        executor.execute(
            "search_sources", arguments, agent="retriever", run_id="run_1"
        )
    )
    second_run = asyncio.run(
        executor.execute(
            "search_sources", arguments, agent="retriever", run_id="run_2"
        )
    )

    assert provider.calls == 2
    assert first.record.status == "DONE"
    assert reused.record.status == "REUSED"
    assert reused.record.reused_from_call_id == first.record.call_id
    assert second_run.record.status == "DONE"


def test_sqlite_ledger_prevents_repeat_after_process_restart(tmp_path):
    from deepresearch.memory import SQLiteMemoryStore

    provider = CountingSearchProvider()
    registry = ToolRegistry()
    register_search_tool(registry, provider)
    database = tmp_path / "tool_ledger.sqlite3"
    first_store = SQLiteMemoryStore(database)
    first = ToolExecutor(
        registry,
        ledger=ToolCallLedger(persistent_store=first_store),
    )
    arguments = {"query": "private query text", "top_k": 1}

    original = asyncio.run(
        first.execute("search_sources", arguments, agent="retriever", run_id="run_1")
    )
    restarted_store = SQLiteMemoryStore(database)
    restarted = ToolExecutor(
        registry,
        ledger=ToolCallLedger(persistent_store=restarted_store),
    )
    reused = asyncio.run(
        restarted.execute("search_sources", arguments, agent="retriever", run_id="run_1")
    )

    assert provider.calls == 1
    assert original.record.status == "DONE"
    assert reused.record.status == "REUSED"
    assert reused.record.reused_from_call_id == original.record.call_id
    assert isinstance(reused.result, PersistedToolResult)
    assert reused.result.result_summary["source_ids"] == ["doc_1"]
    stored_json = "\n".join(
        item.model_dump_json()
        for item in restarted_store.list_tool_call_records(run_id="run_1")
    )
    assert "private query text" not in stored_json
    assert "Sensitive full body" not in stored_json


def test_persisted_approval_and_ledger_prevent_duplicate_side_effect_after_restart(tmp_path):
    from deepresearch.memory import SQLiteMemoryStore

    class PublishInput(BaseModel):
        report_id: str

    calls = 0

    async def handler(payload: PublishInput) -> str:
        nonlocal calls
        calls += 1
        return "published"

    registry = ToolRegistry()
    registry.register(
        RegisteredTool(
            spec=ToolSpec(
                name="publish_report",
                description="Persisted approval test.",
                input_schema=PublishInput.model_json_schema(),
                allowed_agents=["writer"],
                requires_approval=True,
            ),
            input_model=PublishInput,
            handler=handler,
        )
    )
    database = tmp_path / "tool_ledger.sqlite3"
    store = SQLiteMemoryStore(database)
    executor = ToolExecutor(
        registry,
        policy=ToolPolicy(approvals_enabled=True),
        approval_store=store,
        ledger=ToolCallLedger(persistent_store=store),
    )
    arguments = {"report_id": "report_1"}

    with pytest.raises(ToolApprovalRequiredError) as raised:
        asyncio.run(
            executor.execute(
                "publish_report", arguments, agent="writer", run_id="run_1"
            )
        )
    store.decide_approval(
        raised.value.approval_id,
        decision=ApprovalStatus.APPROVED,
        decision_key="approve_once",
    )
    executed = asyncio.run(
        executor.execute("publish_report", arguments, agent="writer", run_id="run_1")
    )

    reopened = SQLiteMemoryStore(database)
    restarted = ToolExecutor(
        registry,
        policy=ToolPolicy(approvals_enabled=True),
        approval_store=reopened,
        ledger=ToolCallLedger(persistent_store=reopened),
    )
    reused = asyncio.run(
        restarted.execute("publish_report", arguments, agent="writer", run_id="run_1")
    )

    assert calls == 1
    assert executed.record.status == "DONE"
    assert reused.record.status == "REUSED"
    assert isinstance(reused.result, PersistedToolResult)
    approval = reopened.get_approval(raised.value.approval_id)
    assert approval is not None
    assert approval.status == ApprovalStatus.EXECUTED


def test_write_risk_tool_is_denied_by_default_policy():
    class EmptyInput(BaseModel):
        pass

    called = False

    async def handler(payload: BaseModel) -> str:
        nonlocal called
        called = True
        return "written"

    registry = ToolRegistry()
    registry.register(
        RegisteredTool(
            spec=ToolSpec(
                name="write_file",
                description="Test-only write action.",
                input_schema=EmptyInput.model_json_schema(),
                risk_level="write",
                allowed_agents=["researcher"],
                idempotent=False,
            ),
            input_model=EmptyInput,
            handler=handler,
        )
    )
    executor = ToolExecutor(registry)

    with pytest.raises(ToolPolicyDeniedError, match="risk_level_not_allowed"):
        asyncio.run(
            executor.execute("write_file", {}, agent="researcher", run_id="run_1")
        )

    assert called is False
    assert executor.ledger.records()[0].status == "DENIED"


def test_approval_required_tool_stays_denied_until_hitl_exists():
    class EmptyInput(BaseModel):
        pass

    async def handler(payload: BaseModel) -> str:
        return "must not run"

    registry = ToolRegistry()
    registry.register(
        RegisteredTool(
            spec=ToolSpec(
                name="publish_report",
                description="Test-only publication action.",
                input_schema=EmptyInput.model_json_schema(),
                risk_level="read_only",
                allowed_agents=["writer"],
                requires_approval=True,
            ),
            input_model=EmptyInput,
            handler=handler,
        )
    )
    executor = ToolExecutor(registry)

    with pytest.raises(ToolPolicyDeniedError, match="approval_flow_not_implemented"):
        asyncio.run(
            executor.execute("publish_report", {}, agent="writer", run_id="run_1")
        )

    record = executor.ledger.records()[0]
    assert record.status == "DENIED"
    assert record.policy_reason == "approval_flow_not_implemented"


def test_approval_required_tool_waits_then_executes_once_after_local_decision():
    class PublishInput(BaseModel):
        report_id: str

    calls = 0

    async def handler(payload: PublishInput) -> str:
        nonlocal calls
        calls += 1
        return f"published:{payload.report_id}"

    registry = ToolRegistry()
    registry.register(
        RegisteredTool(
            spec=ToolSpec(
                name="publish_report",
                description="Publish one report after local approval.",
                input_schema=PublishInput.model_json_schema(),
                risk_level="read_only",
                allowed_agents=["writer"],
                requires_approval=True,
            ),
            input_model=PublishInput,
            handler=handler,
        )
    )
    approvals = InMemoryApprovalStore()
    executor = ToolExecutor(
        registry,
        policy=ToolPolicy(approvals_enabled=True),
        approval_store=approvals,
    )
    arguments = {"report_id": "report_1"}

    with pytest.raises(ToolApprovalRequiredError) as raised:
        asyncio.run(
            executor.execute(
                "publish_report", arguments, agent="writer", run_id="run_1"
            )
        )

    assert calls == 0
    approval = approvals.get_approval(raised.value.approval_id)
    assert approval is not None
    assert approval.status == ApprovalStatus.PENDING
    assert approval.argument_keys == ["report_id"]
    assert "report_1" not in approval.model_dump_json()

    approvals.decide_approval(
        approval.approval_id,
        decision=ApprovalStatus.APPROVED,
        decision_key="decision_1",
    )
    executed = asyncio.run(
        executor.execute("publish_report", arguments, agent="writer", run_id="run_1")
    )
    reused = asyncio.run(
        executor.execute("publish_report", arguments, agent="writer", run_id="run_1")
    )

    assert executed.result == "published:report_1"
    assert executed.record.approval_id == approval.approval_id
    assert reused.record.status == "REUSED"
    assert calls == 1
    completed = approvals.get_approval(approval.approval_id)
    assert completed is not None
    assert completed.status == ApprovalStatus.EXECUTED
    assert completed.execution_call_id == executed.record.call_id


def test_rejected_approval_never_executes_tool():
    class EmptyInput(BaseModel):
        pass

    called = False

    async def handler(payload: EmptyInput) -> str:
        nonlocal called
        called = True
        return "unsafe"

    registry = ToolRegistry()
    registry.register(
        RegisteredTool(
            spec=ToolSpec(
                name="publish_report",
                description="Approval rejection test.",
                input_schema=EmptyInput.model_json_schema(),
                allowed_agents=["writer"],
                requires_approval=True,
            ),
            input_model=EmptyInput,
            handler=handler,
        )
    )
    approvals = InMemoryApprovalStore()
    executor = ToolExecutor(
        registry,
        policy=ToolPolicy(approvals_enabled=True),
        approval_store=approvals,
    )

    with pytest.raises(ToolApprovalRequiredError) as raised:
        asyncio.run(executor.execute("publish_report", {}, agent="writer", run_id="run_1"))
    approvals.decide_approval(
        raised.value.approval_id,
        decision=ApprovalStatus.REJECTED,
        decision_key="reject_1",
        reason="operator rejected publication",
    )

    with pytest.raises(ToolPolicyDeniedError, match="approval_rejected"):
        asyncio.run(executor.execute("publish_report", {}, agent="writer", run_id="run_1"))
    assert called is False


def test_high_risk_tool_requires_approval_even_without_explicit_flag():
    class EmptyInput(BaseModel):
        pass

    async def handler(payload: EmptyInput) -> str:
        return "high risk result"

    registry = ToolRegistry()
    registry.register(
        RegisteredTool(
            spec=ToolSpec(
                name="high_risk_action",
                description="High-risk test action.",
                input_schema=EmptyInput.model_json_schema(),
                risk_level="high_risk",
                allowed_agents=["operator"],
            ),
            input_model=EmptyInput,
            handler=handler,
        )
    )
    approvals = InMemoryApprovalStore()
    executor = ToolExecutor(
        registry,
        policy=ToolPolicy(
            allowed_risk_levels=("read_only", "high_risk"),
            approvals_enabled=True,
        ),
        approval_store=approvals,
    )

    with pytest.raises(ToolApprovalRequiredError) as raised:
        asyncio.run(
            executor.execute("high_risk_action", {}, agent="operator", run_id="run_1")
        )

    approval = approvals.get_approval(raised.value.approval_id)
    assert approval is not None
    assert approval.risk_level == "high_risk"
    assert approval.status == ApprovalStatus.PENDING


def test_tool_timeout_is_bounded_and_audited():
    class EmptyInput(BaseModel):
        pass

    async def slow_handler(payload: BaseModel) -> str:
        await asyncio.sleep(0.05)
        return "late"

    registry = ToolRegistry()
    registry.register(
        RegisteredTool(
            spec=ToolSpec(
                name="slow_read",
                description="Test-only slow read.",
                input_schema=EmptyInput.model_json_schema(),
                risk_level="read_only",
                allowed_agents=["researcher"],
                timeout_seconds=0.001,
            ),
            input_model=EmptyInput,
            handler=slow_handler,
        )
    )
    executor = ToolExecutor(registry)

    with pytest.raises(ToolTimeoutError):
        asyncio.run(
            executor.execute("slow_read", {}, agent="researcher", run_id="run_1")
        )

    record = executor.ledger.records()[0]
    assert record.status == "TIMEOUT"
    assert record.error_type == "ToolTimeoutError"

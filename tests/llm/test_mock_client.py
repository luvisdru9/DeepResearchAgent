import asyncio
import json
from datetime import datetime, timezone

import pytest
from pydantic import BaseModel

from deepresearch.llm import (
    ChatMessage,
    LLMCallBudget,
    LLMCallTrace,
    LLMTraceStore,
    MockLLMClient,
    MockReply,
)
from deepresearch.llm.errors import (
    LLMBudgetExceededError,
    LLMRequestError,
    LLMResponseParseError,
    LLMResponseValidationError,
    LLMTimeoutError,
)


class PlannedAnswer(BaseModel):
    answer: str
    confidence: float


def test_mock_client_returns_normalized_chat_response_and_trace():
    client = MockLLMClient(
        responses=[
            MockReply(
                content="offline answer",
                input_tokens=12,
                output_tokens=3,
                cost_usd=0.0,
                request_id="mock-request",
                finish_reason="length",
            )
        ]
    )

    response = asyncio.run(
        client.chat([ChatMessage(role="user", content="research question")], agent="planner")
    )

    assert response.content == "offline answer"
    assert response.input_tokens == 12
    assert response.output_tokens == 3
    assert response.finish_reason == "length"
    trace = client.trace_collector.since(0)[0]
    assert trace.status == "DONE"
    assert trace.agent == "planner"
    assert trace.operation == "chat"
    assert trace.attempt_count == 1
    assert trace.request_id == "mock-request"
    assert trace.finish_reason == "length"


def test_json_chat_parses_fenced_json_into_pydantic_model():
    client = MockLLMClient(
        responses=['```json\n{"answer": "use evidence", "confidence": 0.9}\n```']
    )

    result = asyncio.run(
        client.json_chat(
            [ChatMessage(role="user", content="return JSON")],
            PlannedAnswer,
            agent="planner",
        )
    )

    assert result == PlannedAnswer(answer="use evidence", confidence=0.9)
    assert client.trace_collector.since(0)[0].operation == "json_chat"


def test_json_chat_retries_parse_failure_then_succeeds():
    client = MockLLMClient(
        responses=[
            "not-json",
            '{"answer": "recovered", "confidence": 0.8}',
        ],
        max_retries=1,
    )

    result = asyncio.run(
        client.json_chat(
            [ChatMessage(role="user", content="return JSON")],
            PlannedAnswer,
            agent="planner",
        )
    )

    assert result.answer == "recovered"
    trace = client.trace_collector.since(0)[0]
    assert trace.attempt_count == 2
    assert trace.retry_delays_seconds == [0.0]
    assert trace.retry_error_types == ["LLMResponseParseError"]


def test_json_parse_retry_never_waits_even_with_network_backoff_enabled():
    client = MockLLMClient(
        responses=["not-json", '{"answer": "recovered", "confidence": 0.8}'],
        max_retries=1,
        retry_base_delay_seconds=5.0,
        retry_max_delay_seconds=5.0,
        retry_jitter_ratio=0.0,
    )
    waits: list[float] = []

    async def capture_sleep(delay: float) -> None:
        waits.append(delay)

    client._sleep = capture_sleep

    result = asyncio.run(
        client.json_chat(
            [ChatMessage(role="user", content="return JSON")],
            PlannedAnswer,
            agent="planner",
        )
    )

    assert result.answer == "recovered"
    assert waits == []
    assert client.trace_collector.since(0)[0].retry_delays_seconds == [0.0]


def test_json_chat_raises_parse_error_after_retry_budget():
    client = MockLLMClient(responses=["bad", "still bad"], max_retries=1)

    with pytest.raises(LLMResponseParseError):
        asyncio.run(
            client.json_chat(
                [ChatMessage(role="user", content="return JSON")],
                PlannedAnswer,
                agent="planner",
            )
        )

    trace = client.trace_collector.since(0)[0]
    assert trace.status == "FAILED"
    assert trace.attempt_count == 2
    assert "LLMResponseParseError" in (trace.error or "")


def test_json_chat_reports_schema_validation_failure():
    client = MockLLMClient(
        responses=['{"answer": "missing confidence"}'],
        max_retries=0,
    )

    with pytest.raises(LLMResponseValidationError):
        asyncio.run(
            client.json_chat(
                [ChatMessage(role="user", content="return JSON")],
                PlannedAnswer,
                agent="planner",
            )
        )

    trace = client.trace_collector.since(0)[0]
    assert trace.status == "FAILED"
    assert "LLMResponseValidationError" in (trace.error or "")


def test_mock_client_retries_provider_exception():
    client = MockLLMClient(
        responses=[RuntimeError("temporary"), "recovered"],
        max_retries=1,
    )

    response = asyncio.run(
        client.chat([ChatMessage(role="user", content="question")], agent="reader")
    )

    assert response.content == "recovered"
    trace = client.trace_collector.since(0)[0]
    assert trace.attempt_count == 2
    assert trace.retry_delays_seconds == [0.0]
    assert trace.retry_error_types == ["RuntimeError"]


def test_mock_client_wraps_provider_error_after_retries():
    client = MockLLMClient(
        responses=[RuntimeError("down"), RuntimeError("still down")],
        max_retries=1,
    )

    with pytest.raises(LLMRequestError):
        asyncio.run(
            client.chat([ChatMessage(role="user", content="question")], agent="reader")
        )

    trace = client.trace_collector.since(0)[0]
    assert trace.status == "FAILED"
    assert trace.attempt_count == 2


def test_mock_client_timeout_is_cancelled_and_traced():
    client = MockLLMClient(
        responses=[MockReply(content="late", delay_seconds=1)],
        timeout_seconds=0.01,
        max_retries=0,
    )

    with pytest.raises(LLMTimeoutError):
        asyncio.run(
            client.chat([ChatMessage(role="user", content="question")], agent="reader")
        )

    trace = client.trace_collector.since(0)[0]
    assert trace.status == "TIMEOUT"
    assert trace.attempt_count == 1


def test_timeout_retries_with_network_backoff_without_real_wait():
    client = MockLLMClient(
        responses=[MockReply(content="late", delay_seconds=0.02), "recovered"],
        timeout_seconds=0.001,
        max_retries=1,
        retry_base_delay_seconds=0.75,
        retry_max_delay_seconds=2.0,
        retry_jitter_ratio=0.0,
    )
    waits: list[float] = []

    async def capture_sleep(delay: float) -> None:
        waits.append(delay)

    client._sleep = capture_sleep

    response = asyncio.run(
        client.chat([ChatMessage(role="user", content="question")], agent="reader")
    )

    assert response.content == "recovered"
    assert waits == [0.75]
    trace = client.trace_collector.since(0)[0]
    assert trace.retry_delays_seconds == [0.75]
    assert trace.retry_error_types == ["LLMTimeoutError"]


def test_mock_client_persists_raw_io_trace_when_enabled(tmp_path):
    client = MockLLMClient(
        responses=['{"answer": "use evidence", "confidence": 0.9}'],
    )
    client.configure_raw_trace(enabled=True, store=LLMTraceStore(tmp_path))
    client.bind_run("run_123", created_at=datetime(2026, 6, 20, 1, 51, 33, tzinfo=timezone.utc))

    result = asyncio.run(
        client.json_chat(
            [ChatMessage(role="user", content="return JSON")],
            PlannedAnswer,
            agent="planner",
        )
    )

    assert result.answer == "use evidence"
    trace = client.trace_collector.since(0)[0]
    assert trace.trace_path is not None
    assert trace.trace_path.startswith("llm_traces/20260620T015133Z_run_123/")
    assert trace.trace_path.endswith(f"_{trace.call_id}.json")
    raw_trace = json.loads((tmp_path / trace.trace_path).read_text(encoding="utf-8"))
    assert raw_trace["messages"] == [{"role": "user", "content": "return JSON"}]
    assert raw_trace["parsed_response"] == {"answer": "use evidence", "confidence": 0.9}
    assert raw_trace["raw_response_text"] == '{"answer": "use evidence", "confidence": 0.9}'
    assert raw_trace["finish_reason"] == "stop"
    assert raw_trace["retry_delays_seconds"] == []
    assert raw_trace["retry_error_types"] == []


def test_old_call_trace_without_retry_diagnostics_remains_readable():
    trace = LLMCallTrace.model_validate(
        {
            "agent": "reader",
            "operation": "chat",
            "model": "legacy-model",
            "status": "DONE",
            "attempt_count": 1,
            "latency_seconds": 0.1,
        }
    )

    assert trace.retry_delays_seconds == []
    assert trace.retry_error_types == []


def test_shared_call_budget_prevents_concurrent_provider_overshoot():
    client = MockLLMClient(
        responses=[
            MockReply(content="first", delay_seconds=0.01),
            MockReply(content="must not be consumed"),
        ]
    )
    client.configure_call_budget(LLMCallBudget(1))

    async def run_calls():
        return await asyncio.gather(
            client.chat([ChatMessage(role="user", content="one")], agent="reader"),
            client.chat([ChatMessage(role="user", content="two")], agent="reader"),
            return_exceptions=True,
        )

    results = asyncio.run(run_calls())

    assert sum(isinstance(item, LLMBudgetExceededError) for item in results) == 1
    assert len(client.trace_collector.since(0)) == 1


def test_call_budget_can_reserve_capacity_for_a_later_phase():
    client = MockLLMClient(responses=["research", "synthesis", "verification"])
    budget = LLMCallBudget(3)
    client.configure_call_budget(budget)

    async def run_phases():
        async with budget.reserve_for_later(2):
            first = await client.chat(
                [ChatMessage(role="user", content="research")],
                agent="reader",
            )
            with pytest.raises(LLMBudgetExceededError):
                await client.chat(
                    [ChatMessage(role="user", content="extra research")],
                    agent="reader",
                )
        second = await client.chat(
            [ChatMessage(role="user", content="synthesize")],
            agent="synthesizer",
        )
        third = await client.chat(
            [ChatMessage(role="user", content="verify")],
            agent="redteam",
        )
        return first, second, third

    responses = asyncio.run(run_phases())

    assert [item.content for item in responses] == ["research", "synthesis", "verification"]
    assert budget.used_calls == 3
    assert len(client.trace_collector.since(0)) == 3

import asyncio
from types import SimpleNamespace

import pytest

from deepresearch.config import LLMConfig
from deepresearch.llm import ChatMessage, OpenAICompatibleClient
from deepresearch.llm.errors import LLMRequestError


class FakeCompletions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests: list[dict] = []

    async def create(self, **kwargs):
        self.requests.append(kwargs)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def fake_sdk(responses):
    completions = FakeCompletions(responses)
    return SimpleNamespace(
        chat=SimpleNamespace(completions=completions),
        completions_api=completions,
    )


def fake_response(content="answer"):
    return SimpleNamespace(
        id="deepseek-request",
        model="deepseek-v4-pro",
        choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=20, completion_tokens=5),
    )


def deepseek_config(**overrides):
    values = {
        "provider": "openai_compatible",
        "model": "deepseek-v4-pro",
        "base_url": "https://api.deepseek.com",
        "api_key_env": "DEEPSEEK_API_KEY",
        "max_retries": 1,
        "temperature": None,
        "request_options": {
            "reasoning_effort": "high",
            "max_tokens": 2000,
            "response_format": {"type": "json_object"},
            "extra_body": {"thinking": {"type": "enabled"}},
        },
    }
    values.update(overrides)
    return LLMConfig(**values)


def test_deepseek_request_options_and_response_are_normalized():
    sdk = fake_sdk([fake_response()])
    client = OpenAICompatibleClient(deepseek_config(), sdk_client=sdk)

    response = asyncio.run(
        client.chat(
            [
                ChatMessage(role="system", content="You are helpful"),
                ChatMessage(role="user", content="Hello"),
            ],
            agent="planner",
        )
    )

    request = sdk.completions_api.requests[0]
    assert request == {
        "model": "deepseek-v4-pro",
        "messages": [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
        ],
        "stream": False,
        "reasoning_effort": "high",
        "max_tokens": 2000,
        "response_format": {"type": "json_object"},
        "extra_body": {"thinking": {"type": "enabled"}},
    }
    assert response.content == "answer"
    assert response.input_tokens == 20
    assert response.output_tokens == 5
    assert response.request_id == "deepseek-request"
    assert response.cost_usd is None
    assert response.finish_reason == "stop"


def test_deepseek_pro_judge_uses_thinking_json_options():
    sdk = fake_sdk([fake_response('{"ok": true}')])
    config = LLMConfig(
        provider="openai_compatible",
        model="deepseek-v4-pro",
        base_url="https://api.deepseek.com",
        api_key_env="DEEPSEEK_API_KEY",
        temperature=None,
        request_options={
            "reasoning_effort": "high",
            "max_tokens": 12000,
            "response_format": {"type": "json_object"},
            "extra_body": {"thinking": {"type": "enabled"}},
        },
    )
    client = OpenAICompatibleClient(
        config,
        profile_name="evaluator_deepseek_pro",
        sdk_client=sdk,
    )

    response = asyncio.run(
        client.chat([ChatMessage(role="user", content="Return JSON")], agent="evaluator")
    )

    assert response.content == '{"ok": true}'
    assert sdk.completions_api.requests[0] == {
        "model": "deepseek-v4-pro",
        "messages": [{"role": "user", "content": "Return JSON"}],
        "stream": False,
        "reasoning_effort": "high",
        "max_tokens": 12000,
        "response_format": {"type": "json_object"},
        "extra_body": {"thinking": {"type": "enabled"}},
    }
    assert "temperature" not in sdk.completions_api.requests[0]


def test_transient_status_is_retried():
    transient = RuntimeError("rate limited")
    transient.status_code = 429
    sdk = fake_sdk([transient, fake_response("recovered")])
    client = OpenAICompatibleClient(deepseek_config(), sdk_client=sdk)
    delays: list[float] = []

    async def capture_sleep(delay: float) -> None:
        delays.append(delay)

    client._sleep = capture_sleep
    client._random_uniform = lambda low, high: 0.0

    response = asyncio.run(
        client.chat([ChatMessage(role="user", content="Hello")], agent="reader")
    )

    assert response.content == "recovered"
    assert len(sdk.completions_api.requests) == 2
    trace = client.trace_collector.since(0)[0]
    assert trace.attempt_count == 2
    assert delays == [0.5]
    assert trace.retry_delays_seconds == [0.5]
    assert trace.retry_error_types == ["RuntimeError"]


def test_auth_error_fails_without_retry_and_does_not_leak_key():
    unauthorized = RuntimeError("invalid authentication")
    unauthorized.status_code = 401
    sdk = fake_sdk([unauthorized])
    client = OpenAICompatibleClient(deepseek_config(max_retries=3), sdk_client=sdk)

    with pytest.raises(LLMRequestError):
        asyncio.run(
            client.chat([ChatMessage(role="user", content="Hello")], agent="reader")
        )

    trace = client.trace_collector.since(0)[0]
    assert len(sdk.completions_api.requests) == 1
    assert trace.attempt_count == 1
    assert trace.retry_delays_seconds == []
    assert trace.retry_error_types == []
    assert "DEEPSEEK_API_KEY" not in (trace.error or "")


def test_transient_failures_use_bounded_exponential_backoff():
    first = RuntimeError("server unavailable")
    first.status_code = 503
    second = RuntimeError("gateway unavailable")
    second.status_code = 502
    sdk = fake_sdk([first, second, fake_response("recovered")])
    client = OpenAICompatibleClient(
        deepseek_config(
            max_retries=2,
            retry_base_delay_seconds=2.0,
            retry_max_delay_seconds=3.0,
            retry_jitter_ratio=0.5,
        ),
        sdk_client=sdk,
    )
    delays: list[float] = []

    async def capture_sleep(delay: float) -> None:
        delays.append(delay)

    client._sleep = capture_sleep
    client._random_uniform = lambda low, high: 0.5

    response = asyncio.run(
        client.chat([ChatMessage(role="user", content="Hello")], agent="reader")
    )

    assert response.content == "recovered"
    assert delays == [3.0, 3.0]
    trace = client.trace_collector.since(0)[0]
    assert trace.retry_delays_seconds == [3.0, 3.0]
    assert trace.retry_error_types == ["RuntimeError", "RuntimeError"]


def test_final_transient_failure_does_not_sleep_after_budget_is_exhausted():
    failure = RuntimeError("rate limited")
    failure.status_code = 429
    sdk = fake_sdk([failure, failure])
    client = OpenAICompatibleClient(deepseek_config(max_retries=1), sdk_client=sdk)
    delays: list[float] = []

    async def capture_sleep(delay: float) -> None:
        delays.append(delay)

    client._sleep = capture_sleep
    client._random_uniform = lambda low, high: 0.0

    with pytest.raises(LLMRequestError):
        asyncio.run(
            client.chat([ChatMessage(role="user", content="Hello")], agent="reader")
        )

    assert delays == [0.5]
    trace = client.trace_collector.since(0)[0]
    assert trace.attempt_count == 2
    assert trace.retry_delays_seconds == [0.5]


def test_provider_error_redacts_api_key_from_trace_and_exception(monkeypatch):
    secret = "secret-deepseek-key"
    monkeypatch.setenv("DEEPSEEK_API_KEY", secret)
    failure = RuntimeError(f"request rejected for bearer {secret}")
    failure.status_code = 401
    sdk = fake_sdk([failure])
    client = OpenAICompatibleClient(deepseek_config(max_retries=0), sdk_client=sdk)

    with pytest.raises(LLMRequestError) as raised:
        asyncio.run(
            client.chat([ChatMessage(role="user", content="Hello")], agent="reader")
        )

    trace = client.trace_collector.since(0)[0]
    assert secret not in str(raised.value)
    assert secret not in (trace.error or "")
    assert "[REDACTED]" in (trace.error or "")


def test_empty_choices_is_reported_as_request_error():
    empty = SimpleNamespace(id="empty", model="deepseek-v4-pro", choices=[], usage=None)
    sdk = fake_sdk([empty])
    client = OpenAICompatibleClient(deepseek_config(max_retries=0), sdk_client=sdk)

    with pytest.raises(LLMRequestError, match="no choices"):
        asyncio.run(
            client.chat([ChatMessage(role="user", content="Hello")], agent="reader")
        )

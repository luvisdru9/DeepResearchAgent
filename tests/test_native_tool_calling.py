from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from deepresearch.config import LLMConfig, ToolCallingConfig
from deepresearch.retrieval import SearchProvider
from deepresearch.schema import SourceDoc
from deepresearch.tools import (
    ConstrainedToolCallingLoop,
    NativeToolCall,
    NativeToolCallingError,
    NativeToolModelResponse,
    OpenAICompatibleToolModel,
    ToolCallingDisabledError,
    ToolExecutor,
    ToolRegistry,
    register_search_tool,
)


class LocalSearchProvider(SearchProvider):
    def __init__(self) -> None:
        self.calls = 0

    def search(self, query: str, top_k: int = 5) -> list[SourceDoc]:
        self.calls += 1
        return [
            SourceDoc(
                id="source_1",
                title="Controlled source",
                url="local://controlled",
                content="The source reports a bounded and cited result.",
            )
        ][:top_k]


class ScriptedToolModel:
    def __init__(self, responses: list[NativeToolModelResponse]) -> None:
        self.responses = list(responses)
        self.requests: list[dict] = []

    async def complete(self, *, messages, tools):
        self.requests.append({"messages": list(messages), "tools": list(tools)})
        return self.responses.pop(0)


def _loop(model, *, enabled: bool = True, max_steps: int = 3):
    provider = LocalSearchProvider()
    registry = ToolRegistry()
    register_search_tool(registry, provider)
    executor = ToolExecutor(registry)
    loop = ConstrainedToolCallingLoop(
        model,
        registry,
        executor,
        ToolCallingConfig(enabled=enabled, max_steps=max_steps),
    )
    return loop, provider


def test_constrained_loop_executes_registered_read_only_tool_then_finishes():
    model = ScriptedToolModel(
        [
            NativeToolModelResponse(
                model="mock",
                finish_reason="tool_calls",
                tool_calls=[
                    NativeToolCall(
                        call_id="call_1",
                        name="search_sources",
                        arguments={"query": "bounded evidence", "top_k": 1},
                    )
                ],
            ),
            NativeToolModelResponse(
                model="mock",
                finish_reason="stop",
                content="The bounded source supports the conclusion.",
            ),
        ]
    )
    loop, provider = _loop(model)

    result = asyncio.run(loop.run(query="Find bounded evidence", run_id="run_1"))

    assert result.status == "complete"
    assert result.content == "The bounded source supports the conclusion."
    assert provider.calls == 1
    assert [record.status for record in result.tool_records] == ["DONE"]
    assert [tool["function"]["name"] for tool in model.requests[0]["tools"]] == [
        "search_sources"
    ]
    assert "untrusted data" in model.requests[0]["messages"][0]["content"]
    second_messages = model.requests[1]["messages"]
    assert second_messages[-1]["role"] == "tool"
    assert "bounded and cited result" in second_messages[-1]["content"]


def test_constrained_loop_is_default_off_and_rejects_unknown_tool():
    disabled_model = ScriptedToolModel([])
    disabled, _ = _loop(disabled_model, enabled=False)
    with pytest.raises(ToolCallingDisabledError):
        asyncio.run(disabled.run(query="query", run_id="run_1"))
    assert disabled_model.requests == []

    unknown_model = ScriptedToolModel(
        [
            NativeToolModelResponse(
                model="mock",
                tool_calls=[
                    NativeToolCall(
                        call_id="call_1",
                        name="delete_files",
                        arguments={},
                    )
                ],
            )
        ]
    )
    enabled, provider = _loop(unknown_model)
    with pytest.raises(NativeToolCallingError, match="unavailable tool"):
        asyncio.run(enabled.run(query="query", run_id="run_1"))
    assert provider.calls == 0


def test_constrained_loop_stops_at_step_limit():
    model = ScriptedToolModel(
        [
            NativeToolModelResponse(
                model="mock",
                tool_calls=[
                    NativeToolCall(
                        call_id="call_1",
                        name="search_sources",
                        arguments={"query": "same query", "top_k": 1},
                    )
                ],
            )
        ]
    )
    loop, provider = _loop(model, max_steps=1)

    result = asyncio.run(loop.run(query="query", run_id="run_1"))

    assert result.status == "step_limit"
    assert provider.calls == 1
    assert len(result.steps) == 1


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


def _sdk(responses):
    completions = FakeCompletions(responses)
    return SimpleNamespace(chat=SimpleNamespace(completions=completions)), completions


def _tool_response(arguments: str = '{"query":"evidence","top_k":1}'):
    function = SimpleNamespace(name="search_sources", arguments=arguments)
    call = SimpleNamespace(id="provider_call_1", function=function)
    message = SimpleNamespace(content=None, tool_calls=[call])
    return SimpleNamespace(
        model="deepseek-flash",
        choices=[SimpleNamespace(message=message, finish_reason="tool_calls")],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
    )


def test_openai_compatible_tool_model_sends_native_tools_and_parses_calls():
    sdk, completions = _sdk([_tool_response()])
    config = LLMConfig(
        provider="openai_compatible",
        model="deepseek-flash",
        base_url="https://api.deepseek.com",
        api_key_env="UNUSED",
        max_retries=0,
        temperature=None,
        request_options={
            "max_tokens": 1000,
            "response_format": {"type": "json_object"},
        },
    )
    model = OpenAICompatibleToolModel(config, sdk_client=sdk)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_sources",
                "description": "Search",
                "parameters": {"type": "object"},
            },
        }
    ]

    response = asyncio.run(
        model.complete(messages=[{"role": "user", "content": "Search"}], tools=tools)
    )

    assert response.tool_calls[0].name == "search_sources"
    assert response.tool_calls[0].arguments == {"query": "evidence", "top_k": 1}
    request = completions.requests[0]
    assert request["tools"] == tools
    assert request["tool_choice"] == "auto"
    assert "response_format" not in request


def test_openai_compatible_tool_model_rejects_invalid_argument_json():
    sdk, _ = _sdk([_tool_response("not-json")])
    model = OpenAICompatibleToolModel(
        LLMConfig(model="mock", max_retries=0),
        sdk_client=sdk,
    )

    with pytest.raises(NativeToolCallingError, match="invalid JSON arguments"):
        asyncio.run(model.complete(messages=[], tools=[]))

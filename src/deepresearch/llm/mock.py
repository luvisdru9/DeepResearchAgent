"""Deterministic offline LLM client used by tests and the default configuration."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Iterable
from uuid import uuid4

from pydantic import BaseModel

from deepresearch.llm.base import LLMClient
from deepresearch.llm.types import ChatMessage, ProviderResponse


class MockReply(BaseModel):
    """One queued mock provider reply."""

    content: str
    model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = 0.0
    request_id: str | None = None
    finish_reason: str | None = "stop"
    delay_seconds: float = 0.0


MockItem = str | MockReply | Exception


class MockLLMClient(LLMClient):
    """Return queued replies without network access or credentials."""

    def __init__(
        self,
        *,
        model: str = "mock-researcher",
        responses: Iterable[MockItem] | None = None,
        default_response: str = "{}",
        **kwargs: object,
    ) -> None:
        # 离线 Mock 默认立即重试；调用方仍可显式传入非零值测试等待逻辑。
        kwargs.setdefault("retry_base_delay_seconds", 0.0)
        kwargs.setdefault("retry_max_delay_seconds", 0.0)
        kwargs.setdefault("retry_jitter_ratio", 0.0)
        super().__init__(model=model, **kwargs)
        self._responses = deque(responses or [])
        self.default_response = default_response

    async def _request(self, messages: list[ChatMessage], model: str) -> ProviderResponse:
        item: MockItem = self._responses.popleft() if self._responses else self.default_response
        if isinstance(item, Exception):
            raise item
        reply = item if isinstance(item, MockReply) else MockReply(content=item)
        if reply.delay_seconds > 0:
            await asyncio.sleep(reply.delay_seconds)

        input_tokens = reply.input_tokens
        if input_tokens is None:
            input_tokens = sum(len(message.content.split()) for message in messages)
        output_tokens = reply.output_tokens
        if output_tokens is None:
            output_tokens = len(reply.content.split())
        return ProviderResponse(
            content=reply.content,
            model=reply.model or model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=reply.cost_usd,
            request_id=reply.request_id or f"mock_{uuid4().hex[:12]}",
            finish_reason=reply.finish_reason,
        )

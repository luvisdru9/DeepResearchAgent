"""Structured inputs, outputs, replay summaries, and raw-call traces for model calls."""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from deepresearch.schema import utc_now


class ChatMessage(BaseModel):
    """One provider-neutral chat message."""

    role: Literal["system", "user", "assistant"]
    content: str


class ProviderResponse(BaseModel):
    """Normalized raw response returned by a concrete provider adapter."""

    content: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    request_id: str | None = None
    # 例如 stop / length；用于区分正常结束与输出额度耗尽。
    finish_reason: str | None = None


class LLMResponse(BaseModel):
    """Public normalized response returned to an Agent."""

    content: str
    parsed: dict | list | None = None
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_seconds: float
    cost_usd: float | None = None
    request_id: str | None = None
    finish_reason: str | None = None


class LLMCallTrace(BaseModel):
    """Replay-safe summary of one logical model call, including retries."""

    call_id: str = Field(default_factory=lambda: f"llm_{uuid4().hex[:12]}")
    agent: str
    operation: Literal["chat", "json_chat"]
    profile: str | None = None
    provider: str | None = None
    model: str
    status: Literal["DONE", "FAILED", "TIMEOUT"]
    attempt_count: int
    retry_delays_seconds: list[float] = Field(default_factory=list)
    retry_error_types: list[str] = Field(default_factory=list)
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_seconds: float
    cost_usd: float | None = None
    request_id: str | None = None
    finish_reason: str | None = None
    trace_path: str | None = None
    error: str | None = None
    started_at: datetime = Field(default_factory=utc_now)
    finished_at: datetime = Field(default_factory=utc_now)


class LLMRawTrace(BaseModel):
    """Full debugging record for one logical model call, stored outside replay."""

    call_id: str = Field(default_factory=lambda: f"llm_{uuid4().hex[:12]}")
    run_id: str
    agent: str
    operation: Literal["chat", "json_chat"]
    profile: str | None = None
    provider: str | None = None
    model: str
    messages: list[ChatMessage]
    request_options: dict
    status: Literal["DONE", "FAILED", "TIMEOUT"]
    attempt_count: int
    retry_delays_seconds: list[float] = Field(default_factory=list)
    retry_error_types: list[str] = Field(default_factory=list)
    raw_response_text: str | None = None
    parsed_response: dict | list | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    request_id: str | None = None
    finish_reason: str | None = None
    error: str | None = None
    started_at: datetime = Field(default_factory=utc_now)
    finished_at: datetime = Field(default_factory=utc_now)


class LLMTraceCollector:
    """In-memory trace sink; Pipeline snapshots only calls made during one run."""

    def __init__(self) -> None:
        self._traces: list[LLMCallTrace] = []

    def record(self, trace: LLMCallTrace) -> None:
        """Append one completed logical call."""

        self._traces.append(trace)

    def mark(self) -> int:
        """Return a stable offset before a Pipeline run begins."""

        return len(self._traces)

    def since(self, offset: int) -> list[LLMCallTrace]:
        """Return a copy of traces recorded since the supplied offset."""

        return list(self._traces[max(0, offset) :])

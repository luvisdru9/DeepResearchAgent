"""Shared retry, timeout, validation, and tracing behavior for LLM clients."""

from __future__ import annotations

from abc import ABC, abstractmethod
import asyncio
from collections.abc import Callable
from contextlib import asynccontextmanager
from datetime import datetime
import random
import time
from typing import Any, Literal, TypeVar, cast
from uuid import uuid4

from pydantic import BaseModel

from deepresearch.llm.errors import (
    LLMBudgetExceededError,
    LLMError,
    LLMRequestError,
    LLMResponseParseError,
    LLMResponseValidationError,
    LLMTimeoutError,
)
from deepresearch.llm.parsing import parse_json_model
from deepresearch.llm.trace_store import LLMTraceStore
from deepresearch.llm.types import (
    ChatMessage,
    LLMCallTrace,
    LLMRawTrace,
    LLMResponse,
    LLMTraceCollector,
    ProviderResponse,
)
from deepresearch.schema import utc_now

T = TypeVar("T", bound=BaseModel)


class LLMCallBudget:
    """Atomically reserve a bounded number of logical calls across concurrent Agents."""

    def __init__(self, max_calls: int, *, used_calls: int = 0) -> None:
        self.max_calls = max(0, max_calls)
        self.used_calls = max(0, used_calls)
        self._lock = asyncio.Lock()
        self._on_reserve: Callable[[int], None] | None = None

    def set_reservation_callback(self, callback: Callable[[int], None] | None) -> None:
        """Observe durable quota reservations before the provider request starts."""

        self._on_reserve = callback

    async def reserve(self) -> None:
        """Reserve one logical call before any provider request or retry is started."""

        async with self._lock:
            if self.used_calls >= self.max_calls:
                raise LLMBudgetExceededError(
                    f"Run-level LLM call budget exhausted ({self.used_calls}/{self.max_calls})."
                )
            self.used_calls += 1
            used_calls = self.used_calls
        if self._on_reserve is not None:
            self._on_reserve(used_calls)

    @asynccontextmanager
    async def reserve_for_later(self, reserved_calls: int):
        """Temporarily keep capacity for synthesis/review while research runs concurrently."""

        if reserved_calls <= 0:
            yield
            return
        async with self._lock:
            original_max_calls = self.max_calls
            # 不把临时上限降到已消费次数以下；这样当前阶段只会被拒绝，不会倒扣。
            self.max_calls = max(self.used_calls, self.max_calls - reserved_calls)
        try:
            yield
        finally:
            async with self._lock:
                self.max_calls = original_max_calls


class LLMClient(ABC):
    """Provider-neutral async client used by future Agent implementations."""

    def __init__(
        self,
        *,
        model: str,
        profile_name: str | None = None,
        provider_name: str | None = None,
        timeout_seconds: float = 60.0,
        max_retries: int = 2,
        retry_base_delay_seconds: float = 0.5,
        retry_max_delay_seconds: float = 8.0,
        retry_jitter_ratio: float = 0.2,
        trace_collector: LLMTraceCollector | None = None,
        raw_trace_store: LLMTraceStore | None = None,
        capture_raw_io: bool = False,
    ) -> None:
        self.model = model
        self.profile_name = profile_name
        self.provider_name = provider_name
        self.timeout_seconds = max(0.001, timeout_seconds)
        self.max_retries = max(0, max_retries)
        self.retry_base_delay_seconds = max(0.0, retry_base_delay_seconds)
        self.retry_max_delay_seconds = max(0.0, retry_max_delay_seconds)
        self.retry_jitter_ratio = min(1.0, max(0.0, retry_jitter_ratio))
        # 可替换函数让单元测试验证退避而不产生真实等待。
        self._sleep = asyncio.sleep
        self._random_uniform = random.uniform
        self.trace_collector = trace_collector or LLMTraceCollector()
        self.raw_trace_store = raw_trace_store
        self.capture_raw_io = capture_raw_io
        self.active_run_id: str | None = None
        self.active_run_created_at: datetime | None = None
        self.call_budget: LLMCallBudget | None = None

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        agent: str,
        model: str | None = None,
    ) -> LLMResponse:
        """Return a normalized text response."""

        response, _ = await self._execute(messages, agent=agent, operation="chat", model=model)
        return response

    async def json_chat(
        self,
        messages: list[ChatMessage],
        response_model: type[T],
        *,
        agent: str,
        model: str | None = None,
    ) -> T:
        """Return a Pydantic-validated structured response."""

        parser: Callable[[str], BaseModel] = lambda content: parse_json_model(content, response_model)
        _, parsed = await self._execute(
            messages,
            agent=agent,
            operation="json_chat",
            model=model,
            parser=parser,
        )
        return cast(T, parsed)

    async def _execute(
        self,
        messages: list[ChatMessage],
        *,
        agent: str,
        operation: Literal["chat", "json_chat"],
        model: str | None,
        parser: Callable[[str], BaseModel] | None = None,
    ) -> tuple[LLMResponse, BaseModel | None]:
        """Execute one logical call with retries and emit exactly one trace."""

        if self.call_budget is not None:
            await self.call_budget.reserve()
        selected_model = model or self.model
        call_id = f"llm_{uuid4().hex[:12]}"
        started_at = utc_now()
        started = time.perf_counter()
        attempts = 0
        final_error: Exception | None = None
        last_provider_response: ProviderResponse | None = None
        retry_delays: list[float] = []
        retry_error_types: list[str] = []

        for attempt in range(1, self.max_retries + 2):
            attempts = attempt
            try:
                provider_response = await asyncio.wait_for(
                    self._request(messages, selected_model),
                    timeout=self.timeout_seconds,
                )
                last_provider_response = provider_response
                parsed_model = parser(provider_response.content) if parser else None
                elapsed = time.perf_counter() - started
                response = LLMResponse(
                    content=provider_response.content,
                    parsed=parsed_model.model_dump(mode="json") if parsed_model else None,
                    model=provider_response.model,
                    input_tokens=provider_response.input_tokens,
                    output_tokens=provider_response.output_tokens,
                    latency_seconds=elapsed,
                    cost_usd=provider_response.cost_usd,
                    request_id=provider_response.request_id,
                    finish_reason=provider_response.finish_reason,
                )
                trace_path = self._persist_raw_trace(
                    call_id=call_id,
                    messages=messages,
                    agent=agent,
                    operation=operation,
                    model=response.model,
                    attempts=attempts,
                    status="DONE",
                    started_at=started_at,
                    response=response,
                    error=None,
                    retry_delays=retry_delays,
                    retry_error_types=retry_error_types,
                )
                self._record_trace(
                    call_id=call_id,
                    agent=agent,
                    operation=operation,
                    response=response,
                    attempts=attempts,
                    status="DONE",
                    started_at=started_at,
                    trace_path=trace_path,
                    retry_delays=retry_delays,
                    retry_error_types=retry_error_types,
                )
                return response, parsed_model
            except asyncio.TimeoutError:
                final_error = LLMTimeoutError(
                    f"LLM call exceeded timeout of {self.timeout_seconds:.3f}s."
                )
            except Exception as exc:  # noqa: BLE001 - provider and validation errors share retry policy.
                final_error = exc
            if final_error is not None and not self._is_retryable(final_error):
                break
            if attempt > self.max_retries:
                break
            retry_error_types.append(type(final_error).__name__)
            delay = self._retry_delay_seconds(final_error, retry_number=attempt)
            retry_delays.append(delay)
            if delay > 0:
                await self._sleep(delay)

        elapsed = time.perf_counter() - started
        status: Literal["FAILED", "TIMEOUT"] = (
            "TIMEOUT" if isinstance(final_error, LLMTimeoutError) else "FAILED"
        )
        error_text = self._format_error(final_error)
        trace_path = self._persist_raw_trace(
            call_id=call_id,
            messages=messages,
            agent=agent,
            operation=operation,
            model=last_provider_response.model if last_provider_response else selected_model,
            attempts=attempts,
            status=status,
            started_at=started_at,
            response=(
                LLMResponse(
                    content=last_provider_response.content,
                    parsed=None,
                    model=last_provider_response.model,
                    input_tokens=last_provider_response.input_tokens,
                    output_tokens=last_provider_response.output_tokens,
                    latency_seconds=elapsed,
                    cost_usd=last_provider_response.cost_usd,
                    request_id=last_provider_response.request_id,
                    finish_reason=last_provider_response.finish_reason,
                )
                if last_provider_response is not None
                else None
            ),
            error=error_text,
            retry_delays=retry_delays,
            retry_error_types=retry_error_types,
        )
        self.trace_collector.record(
            LLMCallTrace(
                call_id=call_id,
                agent=agent,
                operation=operation,
                profile=self.profile_name,
                provider=self.provider_name,
                model=selected_model,
                status=status,
                attempt_count=attempts,
                retry_delays_seconds=retry_delays,
                retry_error_types=retry_error_types,
                input_tokens=last_provider_response.input_tokens if last_provider_response else None,
                output_tokens=last_provider_response.output_tokens if last_provider_response else None,
                latency_seconds=elapsed,
                cost_usd=last_provider_response.cost_usd if last_provider_response else None,
                request_id=last_provider_response.request_id if last_provider_response else None,
                finish_reason=last_provider_response.finish_reason
                if last_provider_response
                else None,
                trace_path=trace_path,
                error=error_text,
                started_at=started_at,
                finished_at=utc_now(),
            )
        )
        if isinstance(final_error, LLMError):
            raise final_error
        raise LLMRequestError(
            f"LLM request failed after {attempts} attempts: {error_text}"
        ) from final_error

    def _record_trace(
        self,
        *,
        call_id: str,
        agent: str,
        operation: Literal["chat", "json_chat"],
        response: LLMResponse,
        attempts: int,
        status: Literal["DONE", "FAILED", "TIMEOUT"],
        started_at: datetime,
        trace_path: str | None,
        retry_delays: list[float],
        retry_error_types: list[str],
    ) -> None:
        """Convert a successful normalized response into a replay trace."""

        self.trace_collector.record(
            LLMCallTrace(
                call_id=call_id,
                agent=agent,
                operation=operation,
                profile=self.profile_name,
                provider=self.provider_name,
                model=response.model,
                status=status,
                attempt_count=attempts,
                retry_delays_seconds=retry_delays,
                retry_error_types=retry_error_types,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                latency_seconds=response.latency_seconds,
                cost_usd=response.cost_usd,
                request_id=response.request_id,
                finish_reason=response.finish_reason,
                trace_path=trace_path,
                started_at=started_at,
                finished_at=utc_now(),
            )
        )

    def configure_raw_trace(
        self,
        *,
        enabled: bool,
        store: LLMTraceStore | None = None,
    ) -> None:
        """Enable or disable full LLM I/O persistence for future calls."""

        self.capture_raw_io = enabled
        self.raw_trace_store = store

    def bind_run(self, run_id: str, *, created_at: datetime | None = None) -> None:
        """Associate subsequent traces with one pipeline run."""

        self.active_run_id = run_id
        self.active_run_created_at = created_at

    def configure_call_budget(self, budget: LLMCallBudget | None) -> None:
        """Share one per-run quota across clients used by concurrent Agents."""

        self.call_budget = budget

    def unbind_run(self) -> None:
        """Clear the current pipeline run association."""

        self.active_run_id = None
        self.active_run_created_at = None
        self.call_budget = None

    def _persist_raw_trace(
        self,
        *,
        call_id: str,
        messages: list[ChatMessage],
        agent: str,
        operation: Literal["chat", "json_chat"],
        model: str,
        attempts: int,
        status: Literal["DONE", "FAILED", "TIMEOUT"],
        started_at: datetime,
        response: LLMResponse | None,
        error: str | None,
        retry_delays: list[float],
        retry_error_types: list[str],
    ) -> str | None:
        """Persist full request/response details when raw tracing is enabled."""

        if not self.capture_raw_io or self.raw_trace_store is None or self.active_run_id is None:
            return None
        trace = LLMRawTrace(
            call_id=call_id,
            run_id=self.active_run_id,
            agent=agent,
            operation=operation,
            profile=self.profile_name,
            provider=self.provider_name,
            model=model,
            messages=messages,
            request_options=self._trace_request_options(),
            status=status,
            attempt_count=attempts,
            retry_delays_seconds=retry_delays,
            retry_error_types=retry_error_types,
            raw_response_text=response.content if response is not None else None,
            parsed_response=response.parsed if response is not None else None,
            input_tokens=response.input_tokens if response is not None else None,
            output_tokens=response.output_tokens if response is not None else None,
            cost_usd=response.cost_usd if response is not None else None,
            request_id=response.request_id if response is not None else None,
            finish_reason=response.finish_reason if response is not None else None,
            error=error,
            started_at=started_at,
            finished_at=utc_now(),
        )
        self.raw_trace_store.write(trace, run_created_at=self.active_run_created_at)
        return self.raw_trace_store.relative_path_for(
            trace,
            run_created_at=self.active_run_created_at,
        )

    def _trace_request_options(self) -> dict[str, Any]:
        """Expose provider-specific request options for raw debugging traces."""

        return {}

    @abstractmethod
    async def _request(self, messages: list[ChatMessage], model: str) -> ProviderResponse:
        """Perform one provider attempt and return a normalized raw response."""

    def _is_retryable(self, error: Exception) -> bool:
        """Return whether a failed attempt should consume the remaining retry budget."""

        return True

    def _retry_delay_seconds(self, error: Exception, *, retry_number: int) -> float:
        """Return bounded exponential delay; local response parsing retries immediately."""

        if isinstance(
            error,
            (LLMResponseParseError, LLMResponseValidationError, LLMBudgetExceededError),
        ):
            return 0.0
        base_delay = min(
            self.retry_max_delay_seconds,
            self.retry_base_delay_seconds * (2 ** max(0, retry_number - 1)),
        )
        if base_delay <= 0:
            return 0.0
        jitter = self._random_uniform(
            -self.retry_jitter_ratio,
            self.retry_jitter_ratio,
        )
        return min(
            self.retry_max_delay_seconds,
            max(0.0, base_delay * (1.0 + jitter)),
        )

    def _format_error(self, error: Exception | None) -> str:
        """Return replay-safe error text; provider adapters may redact credentials."""

        if error is None:
            return "Unknown LLM error."
        return f"{type(error).__name__}: {error}"

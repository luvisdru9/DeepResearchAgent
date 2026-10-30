"""Async adapter for DeepSeek and other OpenAI-compatible Chat Completions APIs."""

from __future__ import annotations

import os
from typing import Any

from deepresearch.config import LLMConfig
from deepresearch.llm.base import LLMClient
from deepresearch.llm.errors import LLMRequestError
from deepresearch.llm.types import ChatMessage, ProviderResponse


class OpenAICompatibleClient(LLMClient):
    """Map an OpenAI-compatible SDK response into the project's common protocol."""

    def __init__(
        self,
        config: LLMConfig,
        *,
        profile_name: str | None = None,
        sdk_client: Any | None = None,
    ) -> None:
        super().__init__(
            model=config.model,
            profile_name=profile_name,
            provider_name=config.provider,
            timeout_seconds=config.timeout_seconds,
            max_retries=config.max_retries,
            retry_base_delay_seconds=config.retry_base_delay_seconds,
            retry_max_delay_seconds=config.retry_max_delay_seconds,
            retry_jitter_ratio=config.retry_jitter_ratio,
        )
        self.temperature = config.temperature
        self.request_options = dict(config.request_options)
        self.base_url = config.base_url
        self.api_key_env = config.api_key_env
        self._client_options: dict[str, Any] | None = None

        if sdk_client is not None:
            self._client = sdk_client
            return

        api_key = os.environ.get(config.api_key_env)
        if not api_key:
            raise ValueError(
                f"Environment variable '{config.api_key_env}' is required for "
                f"LLM provider '{config.provider}'."
            )
        if config.provider == "openai_compatible" and not config.base_url:
            raise ValueError("base_url is required for provider 'openai_compatible'.")

        # 同步 CLI/实验 runner 会创建多个短生命周期事件循环。保存构造参数并在每次请求内
        # 使用 async context manager，可确保连接在所属 loop 关闭前释放。
        self._client = None
        self._client_options = {
            "api_key": api_key,
            "timeout": config.timeout_seconds,
            "max_retries": 0,
        }
        if config.base_url:
            self._client_options["base_url"] = config.base_url

    async def _request(self, messages: list[ChatMessage], model: str) -> ProviderResponse:
        """Perform one non-streaming Chat Completions request."""

        request: dict[str, Any] = {
            "model": model,
            "messages": [message.model_dump(mode="json") for message in messages],
            "stream": False,
            **self.request_options,
        }
        # 部分推理模型不接受 temperature；仅在配置不为 None 时传递。
        if self.temperature is not None:
            request["temperature"] = self.temperature

        if self._client is not None:
            response = await self._client.chat.completions.create(**request)
        else:
            from openai import AsyncOpenAI

            async with AsyncOpenAI(**(self._client_options or {})) as client:
                response = await client.chat.completions.create(**request)
        if not response.choices:
            raise LLMRequestError("OpenAI-compatible response contains no choices.")

        choice = response.choices[0]
        content = choice.message.content or ""
        usage = getattr(response, "usage", None)
        return ProviderResponse(
            content=content,
            model=getattr(response, "model", None) or model,
            input_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
            output_tokens=getattr(usage, "completion_tokens", None) if usage else None,
            # 供应商响应不提供可靠价格，成本留空，后续由独立价格表计算。
            cost_usd=None,
            request_id=getattr(response, "id", None),
            finish_reason=getattr(choice, "finish_reason", None),
        )

    def _is_retryable(self, error: Exception) -> bool:
        """Retry transient HTTP failures, but fail fast on invalid auth or requests."""

        status_code = getattr(error, "status_code", None)
        if status_code is None:
            return True
        if status_code in {400, 401, 403, 404, 422}:
            return False
        return status_code in {408, 409, 429} or status_code >= 500

    def _format_error(self, error: Exception | None) -> str:
        """Redact the configured credential if an SDK error unexpectedly includes it."""

        text = super()._format_error(error)
        secret = os.environ.get(self.api_key_env)
        return text.replace(secret, "[REDACTED]") if secret else text

    def _trace_request_options(self) -> dict[str, Any]:
        """Persist the exact request options used for one logical call."""

        options = dict(self.request_options)
        options["stream"] = False
        if self.temperature is not None:
            options["temperature"] = self.temperature
        if self.base_url:
            options["base_url"] = self.base_url
        return options

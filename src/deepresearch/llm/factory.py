"""Configuration-driven LLM client construction."""

from deepresearch.config import LLMConfig
from deepresearch.llm.base import LLMClient
from deepresearch.llm.mock import MockLLMClient
from deepresearch.llm.openai_compatible import OpenAICompatibleClient


def create_llm_client(config: LLMConfig, profile_name: str | None = None) -> LLMClient:
    """Resolve one named profile and create its provider adapter."""

    resolved = config.resolve_profile(profile_name)
    resolved_profile_name = profile_name or config.active_profile
    if resolved.provider == "mock":
        return MockLLMClient(
            model=resolved.model,
            profile_name=resolved_profile_name,
            provider_name=resolved.provider,
            timeout_seconds=resolved.timeout_seconds,
            max_retries=resolved.max_retries,
        )
    if resolved.provider in {"openai_compatible", "openai"}:
        return OpenAICompatibleClient(
            resolved,
            profile_name=resolved_profile_name,
        )
    raise ValueError(
        f"Unsupported LLM provider '{resolved.provider}'. "
        "Supported providers: mock, openai, openai_compatible."
    )

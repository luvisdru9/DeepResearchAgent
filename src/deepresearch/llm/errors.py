"""Provider-neutral exceptions exposed to Agents."""


class LLMError(Exception):
    """Base exception for all model client failures."""


class LLMRequestError(LLMError):
    """The provider call failed after exhausting retries."""


class LLMTimeoutError(LLMError):
    """The provider did not complete within the configured timeout."""


class LLMBudgetExceededError(LLMError):
    """A run-level logical call quota was exhausted before contacting the provider."""


class LLMResponseParseError(LLMError):
    """The provider returned content that could not be parsed as JSON."""


class LLMResponseValidationError(LLMError):
    """Parsed JSON did not satisfy the requested Pydantic schema."""

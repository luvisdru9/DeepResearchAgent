"""Registry for tool contracts and local handlers."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from deepresearch.tools.errors import ToolAlreadyRegisteredError, ToolNotFoundError
from deepresearch.tools.models import ToolSpec


ToolHandler = Callable[[BaseModel], Awaitable[Any]]
ToolResultSummarizer = Callable[[Any], dict[str, Any]]


def _default_summary(result: Any) -> dict[str, Any]:
    """Summarize an unknown result without serializing its potentially sensitive body."""

    return {"result_type": type(result).__name__}


@dataclass(frozen=True, slots=True)
class RegisteredTool:
    """Internal definition combining the public contract with local-only implementation."""

    spec: ToolSpec
    input_model: type[BaseModel]
    handler: ToolHandler
    summarize_result: ToolResultSummarizer = _default_summary


class ToolRegistry:
    """Fail-closed name lookup; unknown and duplicate tools are never silently accepted."""

    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    def register(self, tool: RegisteredTool) -> None:
        if tool.spec.name in self._tools:
            raise ToolAlreadyRegisteredError(
                f"Tool '{tool.spec.name}' is already registered."
            )
        self._tools[tool.spec.name] = tool

    def get(self, name: str) -> RegisteredTool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise ToolNotFoundError(f"Unknown tool '{name}'.") from exc

    def specs_for_agent(self, agent: str) -> list[ToolSpec]:
        """Expose only tools that declare the requesting role in their allowlist."""

        return [
            tool.spec
            for _, tool in sorted(self._tools.items())
            if agent in tool.spec.allowed_agents
        ]

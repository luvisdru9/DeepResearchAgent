"""Read-only SearchProvider adapter for the controlled tool layer."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from deepresearch.retrieval import SearchProvider
from deepresearch.schema import SourceDoc
from deepresearch.tools.models import ToolSpec
from deepresearch.tools.registry import RegisteredTool, ToolRegistry


class SearchToolInput(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)


class SearchToolResult(BaseModel):
    docs: list[SourceDoc]
    diagnostics: dict[str, Any] = Field(default_factory=dict)


def register_search_tool(
    registry: ToolRegistry,
    provider: SearchProvider,
    *,
    name: str = "search_sources",
    timeout_seconds: float = 30.0,
) -> ToolSpec:
    """Register one idempotent read-only search action backed by an existing provider."""

    spec = ToolSpec(
        name=name,
        description="Search configured research sources and return ranked SourceDoc records.",
        input_schema=SearchToolInput.model_json_schema(),
        risk_level="read_only",
        allowed_agents=["retriever", "researcher"],
        timeout_seconds=timeout_seconds,
        idempotent=True,
    )

    async def handler(payload: BaseModel) -> SearchToolResult:
        request = SearchToolInput.model_validate(payload)
        docs, diagnostics = await provider.search_with_diagnostics_async(
            request.query,
            top_k=request.top_k,
        )
        return SearchToolResult(docs=docs, diagnostics=diagnostics)

    def summarize(result: Any) -> dict[str, Any]:
        parsed = SearchToolResult.model_validate(result)
        return {
            "returned_docs_count": len(parsed.docs),
            "source_ids": [doc.id for doc in parsed.docs],
            "provider": parsed.diagnostics.get("provider"),
        }

    registry.register(
        RegisteredTool(
            spec=spec,
            input_model=SearchToolInput,
            handler=handler,
            summarize_result=summarize,
        )
    )
    return spec

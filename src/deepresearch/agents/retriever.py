"""
RetrieverAgent
是检索 Agent 的包装层，调用底层 search provider。
当前只接了本地检索(local.py)，后续可以替换成 Web Search、向量库或混合检索。
"""

from __future__ import annotations

from deepresearch.retrieval.base import SearchProvider
from deepresearch.schema import SourceDoc, SubTask


class RetrieverAgent:
    """Thin wrapper around a search provider.

    这里保持很薄，是为了让 pipeline 依赖 Agent 接口，而不是依赖具体的本地检索实现。
    """

    def __init__(self, provider: SearchProvider, top_k: int = 5) -> None:
        self.provider = provider
        self.top_k = top_k

    def retrieve(self, subtask: SubTask) -> list[SourceDoc]:
        """Retrieve candidate source documents for one subtask question."""

        return self.provider.search(subtask.question, top_k=self.top_k)

    def retrieve_with_diagnostics(self, subtask: SubTask) -> tuple[list[SourceDoc], dict[str, object]]:
        """Retrieve documents and expose provider ranking details for replay diagnostics."""

        return self.provider.search_with_diagnostics(subtask.question, top_k=self.top_k)

    async def retrieve_async(self, subtask: SubTask) -> list[SourceDoc]:
        """Async extension point used by concurrent DAG branches.

        provider 统一提供异步入口；原生异步 provider 可以覆盖默认兼容实现。
        """

        return await self.provider.search_async(subtask.question, top_k=self.top_k)

    async def retrieve_with_diagnostics_async(
        self,
        subtask: SubTask,
    ) -> tuple[list[SourceDoc], dict[str, object]]:
        """Async retrieval with ranking diagnostics for DAG execution."""

        return await self.provider.search_with_diagnostics_async(
            subtask.question,
            top_k=subtask.retrieval_top_k or self.top_k,
        )

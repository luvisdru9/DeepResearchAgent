"""检索与文档加载的稳定接口，隔离 Agent 和具体搜索服务。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from deepresearch.schema import SourceDoc


class SearchProvider(ABC):
    """所有本地或远程搜索 provider 必须实现的最小接口。"""

    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> list[SourceDoc]:
        """按相关性返回候选来源。"""

    def search_with_diagnostics(
        self,
        query: str,
        top_k: int = 5,
    ) -> tuple[list[SourceDoc], dict[str, object]]:
        """默认诊断实现；provider 可覆盖以暴露更完整的排序信息。"""

        docs = self.search(query, top_k=top_k)
        return docs, {
            "provider": self.__class__.__name__,
            "candidate_docs_count": len(docs),
            "returned_docs_count": len(docs),
            "ranking": [
                {
                    "rank": index,
                    "source_id": doc.id,
                    "source_title": doc.title,
                    "score": doc.score,
                    "matched_terms": [],
                    "returned": True,
                }
                for index, doc in enumerate(docs, start=1)
            ],
        }

    async def search_async(self, query: str, top_k: int = 5) -> list[SourceDoc]:
        """统一异步扩展点；原生异步 provider 可覆盖此方法。"""

        return self.search(query, top_k)

    async def search_with_diagnostics_async(
        self,
        query: str,
        top_k: int = 5,
    ) -> tuple[list[SourceDoc], dict[str, object]]:
        """默认兼容同步 provider；原生异步 provider 可覆盖此方法。"""

        return self.search_with_diagnostics(query, top_k)


@dataclass(slots=True)
class LoadedDocument:
    """DocumentLoader 的 provider-neutral 输出。"""

    content: str
    title: str | None = None
    content_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class DocumentLoader(ABC):
    """从 URL 加载并清洗正文的扩展点。"""

    @abstractmethod
    def load(self, url: str) -> LoadedDocument:
        """加载一个 URL；失败时抛出描述明确且不含密钥的异常。"""

    async def load_async(self, url: str) -> LoadedDocument:
        """统一异步扩展点；原生异步 loader 可覆盖此方法。"""

        return self.load(url)

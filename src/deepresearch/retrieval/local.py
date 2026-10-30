"""
本地 Markdown 语料检索器。会读取 data/local_corpus/*.md，提取标题和正文，用简单关键词交集打分返回 top-k 文档。
"""

# 局部是简单交集top-k版本
from __future__ import annotations

from pathlib import Path

from deepresearch.retrieval.base import SearchProvider
from deepresearch.schema import SourceDoc
from deepresearch.text import tokenize_mixed_text


class LocalSearchProvider(SearchProvider):
    """Simple keyword search over local Markdown files."""

    def __init__(self, corpus_dir: str | Path) -> None:
        self.corpus_dir = Path(corpus_dir)
        # 语料在初始化时加载一次，避免每个 subtask 重复读磁盘。
        self._docs = self._load_docs()

    def search(self, query: str, top_k: int = 5) -> list[SourceDoc]:
        """Return top-k documents by query/document token overlap."""

        docs, _ = self.search_with_diagnostics(query, top_k=top_k)
        return docs

    def search_with_diagnostics(
        self,
        query: str,
        top_k: int = 5,
    ) -> tuple[list[SourceDoc], dict[str, object]]:
        """Return top-k documents plus replay-safe ranking diagnostics."""

        terms = self._tokenize(query)
        scored: list[tuple[int, list[str], SourceDoc]] = []
        all_ranked: list[tuple[int, list[str], SourceDoc]] = []
        for doc in self._docs:
            content_terms = self._tokenize(f"{doc.title} {doc.content}")
            matched_terms = sorted(terms.intersection(content_terms))
            score = len(matched_terms)
            all_ranked.append((score, matched_terms, doc))
            if score:
                scored.append((score, matched_terms, doc))
        scored.sort(key=lambda item: item[0], reverse=True)
        all_ranked.sort(key=lambda item: item[0], reverse=True)
        if not scored:
            # 没有关键词命中时仍返回部分语料，让下游能生成“证据不足”的可复盘结果。
            selected = [(0, [], doc) for doc in self._docs[:top_k]]
        else:
            selected = scored[:top_k]
        # 把本地匹配分数附在返回对象上，使 hybrid reranker 不必反向解析 diagnostics。
        docs = [
            doc.model_copy(
                update={
                    "score": float(score),
                    "metadata": {**doc.metadata, "provider": "local"},
                }
            )
            for score, _, doc in selected
        ]

        diagnostics = {
            "provider": "local",
            "query_terms": sorted(terms),
            "candidate_docs_count": len(self._docs),
            "returned_docs_count": len(docs),
            "ranking": [
                {
                    "rank": rank,
                    "source_id": doc.id,
                    "source_title": doc.title,
                    "score": score,
                    "matched_terms": matched_terms,
                    "returned": any(doc.id == selected_doc.id for _, _, selected_doc in selected),
                }
                for rank, (score, matched_terms, doc) in enumerate(all_ranked, start=1)
            ],
        }
        return docs, diagnostics

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """Tokenize mixed Chinese/English text without adding a heavyweight dependency.

        Python regex treats a full Chinese phrase as one token, which made local retrieval
        scores collapse to zero for natural Chinese questions. We keep Latin tokens and add
        short CJK n-grams so terms such as“引用质量”“幻觉率”“评测” can match local docs.
        """

        return tokenize_mixed_text(text)

    def _load_docs(self) -> list[SourceDoc]:
        """Load Markdown files into SourceDoc records."""

        docs: list[SourceDoc] = []
        if not self.corpus_dir.exists():
            return docs
        for index, path in enumerate(sorted(self.corpus_dir.glob("*.md")), start=1):
            raw_content = path.read_text(encoding="utf-8")
            title = self._extract_title(raw_content, fallback=path.stem)
            docs.append(
                SourceDoc(
                    id=f"local_{index}",
                    title=title,
                    url=f"local://{path.name}",
                    content=self._strip_markdown_title(raw_content),
                    source_type="local",
                )
            )
        return docs

    @staticmethod
    def _extract_title(content: str, fallback: str) -> str:
        """Use the first H1 heading as the source title."""

        for line in content.splitlines():
            if line.startswith("# "):
                return line[2:].strip()
        return fallback

    @staticmethod
    def _strip_markdown_title(content: str) -> str:
        """Remove H1 title lines from document body content."""

        return "\n".join(line for line in content.splitlines() if not line.startswith("# ")).strip()

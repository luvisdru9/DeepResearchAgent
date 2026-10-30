"""Hybrid retrieval with partial-failure tolerance, deduplication, and reranking."""

from __future__ import annotations

import asyncio
import hashlib
import re

from deepresearch.retrieval.base import SearchProvider
from deepresearch.retrieval.rerank import LightweightReranker
from deepresearch.retrieval.url_utils import canonicalize_url
from deepresearch.schema import SourceDoc


class HybridSearchProvider(SearchProvider):
    """Merge multiple providers while preserving provenance and per-provider diagnostics."""

    def __init__(
        self,
        providers: dict[str, SearchProvider],
        *,
        candidate_multiplier: int = 2,
        reranker: LightweightReranker | None = None,
    ) -> None:
        if not providers:
            raise ValueError("Hybrid retrieval requires at least one provider.")
        self.providers = providers
        self.candidate_multiplier = max(1, candidate_multiplier)
        self.reranker = reranker

    def add_provider(self, name: str, provider: SearchProvider) -> None:
        """Add an optional runtime provider, such as the run-scoped Evidence Store."""

        if name in self.providers:
            raise ValueError(f"Hybrid provider '{name}' is already configured.")
        self.providers[name] = provider

    def search(self, query: str, top_k: int = 5) -> list[SourceDoc]:
        docs, _ = self.search_with_diagnostics(query, top_k=top_k)
        return docs

    def search_with_diagnostics(
        self, query: str, top_k: int = 5
    ) -> tuple[list[SourceDoc], dict[str, object]]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        successes: list[tuple[str, list[SourceDoc], dict[str, object]]] = []
        errors: dict[str, str] = {}
        candidate_k = top_k * self.candidate_multiplier
        for name, provider in self.providers.items():
            try:
                docs, diagnostics = provider.search_with_diagnostics(query, top_k=candidate_k)
                successes.append((name, docs, diagnostics))
            except Exception as exc:
                errors[name] = f"{type(exc).__name__}: {exc}"
        return self._merge(query, top_k, successes, errors)

    async def search_with_diagnostics_async(
        self, query: str, top_k: int = 5
    ) -> tuple[list[SourceDoc], dict[str, object]]:
        """Search all backends concurrently; one backend failure only degrades that query."""

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        candidate_k = top_k * self.candidate_multiplier

        async def call(name: str, provider: SearchProvider):
            try:
                docs, diagnostics = await provider.search_with_diagnostics_async(
                    query, top_k=candidate_k
                )
                return name, docs, diagnostics, None
            except Exception as exc:
                return name, [], {}, f"{type(exc).__name__}: {exc}"

        results = await asyncio.gather(
            *(call(name, provider) for name, provider in self.providers.items())
        )
        successes = [(name, docs, diagnostics) for name, docs, diagnostics, error in results if error is None]
        errors = {name: error for name, _, _, error in results if error is not None}
        return self._merge(query, top_k, successes, errors)

    def _merge(
        self,
        query: str,
        top_k: int,
        successes: list[tuple[str, list[SourceDoc], dict[str, object]]],
        errors: dict[str, str],
    ) -> tuple[list[SourceDoc], dict[str, object]]:
        if not successes:
            details = "; ".join(f"{name}: {error}" for name, error in errors.items())
            raise RuntimeError(f"All hybrid retrieval providers failed. {details}")

        candidates: list[SourceDoc] = []
        by_url: dict[str, int] = {}
        by_content: dict[str, int] = {}
        duplicate_url_count = 0
        duplicate_content_count = 0
        provider_diagnostics: dict[str, object] = {}
        for provider_name, docs, diagnostics in successes:
            provider_diagnostics[provider_name] = diagnostics
            for provider_rank, doc in enumerate(docs, start=1):
                url_key = self._url_key(doc.url)
                content_key = self._content_key(doc.content)
                duplicate_index = by_url.get(url_key) if url_key else None
                if duplicate_index is not None:
                    duplicate_url_count += 1
                elif content_key:
                    duplicate_index = by_content.get(content_key)
                    if duplicate_index is not None:
                        duplicate_content_count += 1
                if duplicate_index is not None:
                    candidates[duplicate_index] = self._merge_duplicate(
                        candidates[duplicate_index], provider_name, provider_rank, doc
                    )
                    continue
                metadata = {
                    **doc.metadata,
                    "primary_provider": provider_name,
                    "providers": [provider_name],
                    "provider_ranks": {provider_name: provider_rank},
                    "provider_scores": {provider_name: doc.score},
                    # 每个 provider 的原始元数据独立保留，避免去重后丢失历史 run/evidence 血缘。
                    "provider_metadata": {provider_name: dict(doc.metadata)},
                }
                candidates.append(doc.model_copy(update={"metadata": metadata}))
                index = len(candidates) - 1
                if url_key:
                    by_url[url_key] = index
                if content_key:
                    by_content[content_key] = index

        if self.reranker is not None:
            ranked, ranking = self.reranker.rerank(query, candidates)
        else:
            ranked = candidates
            ranking = [
                {"original_rank": index, "selected_rank": index, "source_id": doc.id}
                for index, doc in enumerate(candidates, start=1)
            ]
        selected = ranked[:top_k]
        selected_ids = {doc.id for doc in selected}
        for item in ranking:
            item["returned"] = item.get("source_id") in selected_ids
        diagnostics: dict[str, object] = {
            "provider": "hybrid",
            "providers_requested": list(self.providers),
            "providers_succeeded": [name for name, _, _ in successes],
            "provider_errors": errors,
            "raw_candidate_count": sum(len(docs) for _, docs, _ in successes),
            "candidate_docs_count": len(candidates),
            "returned_docs_count": len(selected),
            "duplicate_url_count": duplicate_url_count,
            "duplicate_content_count": duplicate_content_count,
            "rerank_enabled": self.reranker is not None,
            "ranking": ranking,
            "provider_diagnostics": provider_diagnostics,
        }
        return selected, diagnostics

    @staticmethod
    def _url_key(url: str) -> str | None:
        try:
            return canonicalize_url(url)
        except ValueError:
            return url.strip().lower() or None

    @staticmethod
    def _content_key(content: str) -> str | None:
        normalized = re.sub(r"\s+", " ", content).strip().lower()
        if not normalized:
            return None
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    @staticmethod
    def _merge_duplicate(
        existing: SourceDoc,
        provider_name: str,
        provider_rank: int,
        duplicate: SourceDoc,
    ) -> SourceDoc:
        providers = list(existing.metadata.get("providers", []))
        if provider_name not in providers:
            providers.append(provider_name)
        metadata = {
            **existing.metadata,
            "providers": providers,
            "provider_ranks": {
                **existing.metadata.get("provider_ranks", {}),
                provider_name: provider_rank,
            },
            "provider_scores": {
                **existing.metadata.get("provider_scores", {}),
                provider_name: duplicate.score,
            },
            "provider_metadata": {
                **existing.metadata.get("provider_metadata", {}),
                provider_name: dict(duplicate.metadata),
            },
        }
        # 优先保留正文更完整的副本，但不改变稳定 source id 和引用 URL。
        content = duplicate.content if len(duplicate.content) > len(existing.content) else existing.content
        return existing.model_copy(update={"content": content, "metadata": metadata})

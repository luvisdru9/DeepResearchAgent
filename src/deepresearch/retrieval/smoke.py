"""Retrieval-only smoke checks that avoid LLM cost during provider validation."""

from __future__ import annotations

from typing import Any

from deepresearch.config import RetrievalConfig
from deepresearch.retrieval.base import SearchProvider
from deepresearch.retrieval.factory import create_search_provider


async def run_retrieval_smoke(
    config: RetrievalConfig,
    query: str,
    *,
    provider: SearchProvider | None = None,
) -> dict[str, Any]:
    """Run one search and return replay-safe metadata without dumping document bodies."""

    active_provider = provider or create_search_provider(config)
    docs, diagnostics = await active_provider.search_with_diagnostics_async(
        query,
        top_k=config.top_k,
    )
    return {
        "provider": config.provider,
        "query": query,
        "top_k": config.top_k,
        "documents": [
            {
                "id": doc.id,
                "title": doc.title,
                "url": doc.url,
                "source_type": doc.source_type,
                "score": doc.score,
                "published_at": doc.published_at,
                "content_chars": len(doc.content),
                "metadata": doc.metadata,
            }
            for doc in docs
        ],
        "diagnostics": diagnostics,
    }

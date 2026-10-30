"""根据统一配置创建检索 provider。"""

from __future__ import annotations

from deepresearch.config import RetrievalConfig
from deepresearch.retrieval.base import SearchProvider
from deepresearch.retrieval.local import LocalSearchProvider


def create_search_provider(config: RetrievalConfig) -> SearchProvider:
    """创建配置指定的 provider；未知类型应尽早失败。"""

    if config.provider == "local":
        return LocalSearchProvider(config.corpus_dir)
    if config.provider == "frozen":
        from deepresearch.retrieval.frozen import FrozenSearchProvider

        if config.frozen_manifest_path is None:
            raise ValueError(
                "retrieval.frozen_manifest_path is required when provider is 'frozen'."
            )
        return FrozenSearchProvider(config.frozen_manifest_path)
    if config.provider == "tavily":
        # 延迟导入，保证离线模式不加载在线 provider 的实现依赖。
        from deepresearch.retrieval.web import TavilySearchProvider

        return TavilySearchProvider.from_config(config)
    if config.provider == "hybrid":
        from deepresearch.retrieval.hybrid import HybridSearchProvider
        from deepresearch.retrieval.rerank import LightweightReranker

        provider_names = list(dict.fromkeys(config.hybrid_providers))
        if "hybrid" in provider_names:
            raise ValueError("retrieval.hybrid_providers cannot contain 'hybrid'.")
        providers = {
            name: create_search_provider(config.model_copy(update={"provider": name}))
            for name in provider_names
        }
        reranker = (
            LightweightReranker(
                provider_weight=config.rerank_provider_weight,
                lexical_weight=config.rerank_lexical_weight,
                freshness_weight=config.rerank_freshness_weight,
                credibility_weight=config.rerank_credibility_weight,
                primary_source_bonus=config.rerank_primary_source_bonus,
                entity_anchor_penalty=config.rerank_entity_anchor_penalty,
            )
            if config.rerank_enabled
            else None
        )
        return HybridSearchProvider(
            providers,
            candidate_multiplier=config.hybrid_candidate_multiplier,
            reranker=reranker,
        )
    raise ValueError(
        f"Unsupported retrieval provider '{config.provider}'. "
        "Supported providers: local, frozen, tavily, hybrid."
    )

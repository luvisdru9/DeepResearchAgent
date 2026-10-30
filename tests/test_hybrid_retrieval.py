from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from deepresearch.retrieval.base import SearchProvider
from deepresearch.retrieval.hybrid import HybridSearchProvider
from deepresearch.retrieval.rerank import LightweightReranker
from deepresearch.schema import SourceDoc


class StaticProvider(SearchProvider):
    def __init__(self, docs: list[SourceDoc], *, error: Exception | None = None) -> None:
        self.docs = docs
        self.error = error

    def search(self, query: str, top_k: int = 5) -> list[SourceDoc]:
        if self.error:
            raise self.error
        return self.docs[:top_k]


def _doc(identifier: str, url: str, content: str, score: float | None) -> SourceDoc:
    return SourceDoc(id=identifier, title=identifier, url=url, content=content, score=score)


def test_hybrid_deduplicates_url_and_content_while_preserving_provenance():
    duplicate = _doc("a2", "https://example.com/a#part", "same body extended", 0.7)
    duplicate.metadata["originating_run_id"] = "prior_run"
    provider = HybridSearchProvider(
        {
            "first": StaticProvider(
                [
                    _doc("a", "https://example.com/a?utm_source=x", "same body", 0.9),
                    _doc("b", "https://example.com/b", "second body", 0.8),
                ]
            ),
            "second": StaticProvider(
                [
                    duplicate,
                    _doc("c", "https://other.example/c", "second body", 0.6),
                ]
            ),
        },
        reranker=LightweightReranker(freshness_weight=0),
    )

    docs, diagnostics = provider.search_with_diagnostics("same second", top_k=5)

    assert len(docs) == 2
    by_id = {doc.id: doc for doc in docs}
    assert by_id["a"].metadata["providers"] == ["first", "second"]
    assert by_id["a"].metadata["provider_metadata"]["second"]["originating_run_id"] == "prior_run"
    assert diagnostics["duplicate_url_count"] == 1
    assert diagnostics["duplicate_content_count"] == 1
    assert diagnostics["raw_candidate_count"] == 4


def test_hybrid_returns_partial_results_when_one_provider_fails():
    provider = HybridSearchProvider(
        {
            "good": StaticProvider([_doc("good", "https://good.example", "answer", 1.0)]),
            "bad": StaticProvider([], error=RuntimeError("unavailable")),
        }
    )

    docs, diagnostics = provider.search_with_diagnostics("answer", top_k=1)

    assert [doc.id for doc in docs] == ["good"]
    assert diagnostics["providers_succeeded"] == ["good"]
    assert "RuntimeError" in diagnostics["provider_errors"]["bad"]


def test_reranker_can_promote_lexically_relevant_result():
    reranker = LightweightReranker(provider_weight=0.1, lexical_weight=0.9, freshness_weight=0)
    docs = [
        _doc("high-provider", "https://a.example", "unrelated text", 1.0),
        _doc("relevant", "https://b.example", "citation quality hallucination", 0.1),
    ]
    docs[0].metadata["provider"] = "same"
    docs[1].metadata["provider"] = "same"

    ranked, diagnostics = reranker.rerank("citation quality", docs)

    assert ranked[0].id == "relevant"
    assert diagnostics[0]["combined_score"] > diagnostics[1]["combined_score"]


def test_reranker_can_prefer_primary_paper_over_secondary_index():
    reranker = LightweightReranker(
        provider_weight=0,
        lexical_weight=0,
        freshness_weight=0,
        credibility_weight=1,
    )
    docs = [
        _doc(
            "index",
            "https://www.semanticscholar.org/paper/example",
            "citation benchmark " * 50,
            1.0,
        ),
        _doc(
            "paper",
            "https://arxiv.org/abs/2305.14627",
            "citation benchmark " * 50,
            1.0,
        ),
    ]
    for doc in docs:
        doc.source_type = "web"
        doc.metadata["provider"] = "same"

    ranked, diagnostics = reranker.rerank("citation benchmark", docs)

    assert ranked[0].id == "paper"
    assert diagnostics[0]["credibility_tier"] == "high"
    assert "primary_research_repository" in diagnostics[0]["credibility_signals"]
    assert diagnostics[1]["credibility_tier"] == "medium"


def test_academia_dot_edu_is_not_treated_as_an_institutional_domain():
    doc = _doc(
        "sharing-platform",
        "https://www.academia.edu/123/example",
        "citation benchmark " * 50,
        1.0,
    )
    doc.source_type = "web"

    _, diagnostics = LightweightReranker().rerank("citation benchmark", [doc])

    assert "institutional_domain" not in diagnostics[0]["credibility_signals"]
    assert "secondary_reference_platform" in diagnostics[0]["credibility_signals"]
    assert diagnostics[0]["credibility_tier"] == "medium"


def test_reranker_preserves_zero_provider_scores():
    docs = [
        _doc("zero_a", "local://a", "unrelated", 0.0),
        _doc("zero_b", "local://b", "unrelated", 0.0),
    ]
    for doc in docs:
        doc.metadata["provider"] = "local"

    _, diagnostics = LightweightReranker().rerank("target", docs)

    assert [item["provider_score_normalized"] for item in diagnostics] == [0.0, 0.0]


def test_reranker_penalizes_results_missing_named_entity_in_title_and_lead():
    reranker = LightweightReranker(
        provider_weight=1,
        lexical_weight=0,
        freshness_weight=0,
        credibility_weight=0,
        primary_source_bonus=0,
        entity_anchor_penalty=0.4,
    )
    docs = [
        SourceDoc(
            id="unrelated",
            title="VC-Bench: A benchmark for another task",
            url="https://arxiv.org/abs/1",
            content="An unrelated evaluation benchmark.",
            score=1.0,
            metadata={"provider": "tavily"},
        ),
        SourceDoc(
            id="target",
            title="ALCE: Automatic LLM Citation Evaluation",
            url="https://arxiv.org/abs/2",
            content="ALCE evaluates citation quality.",
            score=0.8,
            metadata={"provider": "tavily"},
        ),
    ]

    ranked, diagnostics = reranker.rerank(
        "ALCE original paper official benchmark dataset evaluation metrics",
        docs,
    )

    assert [doc.id for doc in ranked] == ["target", "unrelated"]
    details = {item["source_id"]: item for item in diagnostics}
    assert details["target"]["matched_entity_anchors"] == ["ALCE"]
    assert details["target"]["entity_anchor_penalty"] == 0.0
    assert details["unrelated"]["entity_anchor_penalty"] == 0.4


def test_explicit_original_paper_query_prioritizes_matching_primary_source():
    reranker = LightweightReranker(
        provider_weight=1,
        lexical_weight=0,
        freshness_weight=0,
        credibility_weight=0,
        primary_source_bonus=0,
        entity_anchor_penalty=0,
    )
    docs = [
        SourceDoc(
            id="secondary",
            title="RAGTruth benchmark overview",
            url="https://example.com/ragtruth-overview",
            content="RAGTruth is discussed in this overview.",
            score=1.0,
            metadata={"provider": "tavily"},
        ),
        SourceDoc(
            id="primary",
            title="RAGTruth: A Hallucination Corpus",
            url="https://openreview.net/forum?id=ragtruth",
            content="We present RAGTruth for hallucination analysis.",
            score=0.1,
            metadata={"provider": "tavily"},
        ),
    ]

    ranked, diagnostics = reranker.rerank(
        "RAGTruth original paper official evaluation metrics",
        docs,
    )

    assert [doc.id for doc in ranked] == ["primary", "secondary"]
    details = {item["source_id"]: item for item in diagnostics}
    assert details["primary"]["primary_intent_priority"] is True
    assert details["secondary"]["primary_intent_priority"] is False


def test_reranker_accepts_tavily_rfc2822_published_date():
    year = datetime.now(timezone.utc).year
    doc = _doc("recent", "https://recent.example", "content", 1.0)
    doc.published_at = f"Tue, 11 Mar {year} 17:00:00 GMT"

    _, diagnostics = LightweightReranker().rerank("content", [doc])

    assert diagnostics[0]["freshness_score"] > 0


def test_hybrid_async_searches_providers_concurrently():
    active = 0
    peak = 0

    class AsyncProvider(StaticProvider):
        async def search_with_diagnostics_async(self, query: str, top_k: int = 5):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.01)
            active -= 1
            docs = self.search(query, top_k)
            return docs, {"provider": docs[0].id}

    provider = HybridSearchProvider(
        {
            "a": AsyncProvider([_doc("a", "https://a.example", "a", 1.0)]),
            "b": AsyncProvider([_doc("b", "https://b.example", "b", 1.0)]),
        }
    )

    docs, _ = asyncio.run(provider.search_with_diagnostics_async("query", top_k=2))

    assert {doc.id for doc in docs} == {"a", "b"}
    assert peak == 2

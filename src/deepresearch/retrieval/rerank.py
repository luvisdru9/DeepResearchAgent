"""Provider-neutral, deterministic reranking for merged retrieval candidates."""

from __future__ import annotations

from datetime import datetime, timezone

from deepresearch.entities import extract_named_research_entities
from deepresearch.quality import assess_source_credibility
from deepresearch.retrieval.url_utils import parse_published_datetime
from deepresearch.schema import SourceDoc
from deepresearch.text import tokenize_mixed_text


class LightweightReranker:
    """Combine provider score, lexical relevance, and freshness without another LLM call."""

    def __init__(
        self,
        *,
        provider_weight: float = 0.55,
        lexical_weight: float = 0.35,
        freshness_weight: float = 0.10,
        credibility_weight: float = 0.15,
        primary_source_bonus: float = 0.10,
        entity_anchor_penalty: float = 0.25,
    ) -> None:
        total = provider_weight + lexical_weight + freshness_weight + credibility_weight
        if total <= 0:
            raise ValueError("At least one reranking weight must be positive.")
        # 允许 YAML 使用任意同尺度权重，内部归一化后保持分数在 0..1。
        self.provider_weight = provider_weight / total
        self.lexical_weight = lexical_weight / total
        self.freshness_weight = freshness_weight / total
        self.credibility_weight = credibility_weight / total
        self.primary_source_bonus = primary_source_bonus
        self.entity_anchor_penalty = entity_anchor_penalty

    def rerank(self, query: str, docs: list[SourceDoc]) -> tuple[list[SourceDoc], list[dict[str, object]]]:
        """Return copied documents in reranked order plus explainable score components."""

        if not docs:
            return [], []
        query_terms = tokenize_mixed_text(query)
        entity_anchors = self._query_entity_anchors(query)
        primary_intent = self._query_requests_primary_source(query)
        normalized_provider_scores = self._normalize_provider_scores(docs)
        scored: list[tuple[int, float, int, SourceDoc, dict[str, object]]] = []
        for original_rank, raw_doc in enumerate(docs, start=1):
            # 在截断 top-k 之前计算可信度，使第一手来源偏好真正参与候选选择。
            doc = assess_source_credibility(raw_doc)
            document_terms = tokenize_mixed_text(f"{doc.title} {doc.content}")
            lexical_score = (
                len(query_terms.intersection(document_terms)) / len(query_terms)
                if query_terms
                else 0.0
            )
            freshness_score = self._freshness_score(doc.published_at)
            credibility_score = doc.credibility.score if doc.credibility is not None else 0.5
            credibility_signals = (
                list(doc.credibility.signals) if doc.credibility is not None else []
            )
            primary_source_bonus = (
                self.primary_source_bonus
                if "primary_research_repository" in credibility_signals
                else 0.0
            )
            anchor_text = f"{doc.title} {doc.content[:800]}".casefold()
            matched_anchors = [
                anchor for anchor in entity_anchors if anchor.casefold() in anchor_text
            ]
            is_primary_source = "primary_research_repository" in credibility_signals
            # 只在用户/补研查询明确要求第一手材料、且候选命中目标实体时硬优先。
            # 普通检索仍完全按综合分排序，避免权威域名压过真正相关的二手材料。
            primary_intent_priority = int(
                primary_intent and is_primary_source and bool(matched_anchors)
            )
            # 只有查询中存在明确专名时才降权，避免普通英文问句被误判。
            anchor_penalty = (
                self.entity_anchor_penalty
                if entity_anchors and not matched_anchors
                else 0.0
            )
            provider_score = normalized_provider_scores[original_rank - 1]
            combined_score = max(
                0.0,
                min(
                    1.0,
                    self.provider_weight * provider_score
                    + self.lexical_weight * lexical_score
                    + self.freshness_weight * freshness_score
                    + self.credibility_weight * credibility_score
                    + primary_source_bonus
                    - anchor_penalty,
                ),
            )
            score_details: dict[str, object] = {
                "original_rank": original_rank,
                "source_id": doc.id,
                "source_title": doc.title,
                "providers": doc.metadata.get("providers", [doc.metadata.get("provider", "unknown")]),
                "provider_score_normalized": round(provider_score, 6),
                "lexical_score": round(lexical_score, 6),
                "freshness_score": round(freshness_score, 6),
                "credibility_score": round(credibility_score, 6),
                "credibility_tier": (
                    doc.credibility.tier if doc.credibility is not None else "unknown"
                ),
                "credibility_signals": credibility_signals,
                "primary_source_bonus": round(primary_source_bonus, 6),
                "is_primary_source": is_primary_source,
                "primary_intent_query": primary_intent,
                "primary_intent_priority": bool(primary_intent_priority),
                "entity_anchors": entity_anchors,
                "matched_entity_anchors": matched_anchors,
                "entity_anchor_penalty": round(anchor_penalty, 6),
                "combined_score": round(combined_score, 6),
            }
            metadata = {
                **doc.metadata,
                "rerank": score_details,
                "original_provider_score": doc.score,
            }
            scored.append(
                (
                    primary_intent_priority,
                    combined_score,
                    original_rank,
                    doc.model_copy(update={"score": combined_score, "metadata": metadata}),
                    score_details,
                )
            )
        scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
        diagnostics: list[dict[str, object]] = []
        reranked: list[SourceDoc] = []
        for selected_rank, (_, _, _, doc, details) in enumerate(scored, start=1):
            details["selected_rank"] = selected_rank
            reranked.append(doc)
            diagnostics.append(details)
        return reranked, diagnostics

    @staticmethod
    def _query_entity_anchors(query: str) -> list[str]:
        """Extract acronym/CamelCase identifiers expected near a relevant result's lead."""

        return extract_named_research_entities(query)

    @staticmethod
    def _query_requests_primary_source(query: str) -> bool:
        """Recognize explicit first-party retrieval intent used by focused follow-up tasks."""

        normalized = query.casefold()
        return any(
            phrase in normalized
            for phrase in (
                "original paper",
                "primary source",
                "official paper",
                "official source",
                "原始论文",
                "原论文",
                "一手来源",
                "官方论文",
                "官方来源",
            )
        )

    @staticmethod
    def _normalize_provider_scores(docs: list[SourceDoc]) -> list[float]:
        """Normalize independently per provider so incomparable backend scales do not dominate."""

        grouped: dict[str, list[tuple[int, float | None]]] = {}
        for index, doc in enumerate(docs):
            provider = str(doc.metadata.get("primary_provider") or doc.metadata.get("provider") or "unknown")
            grouped.setdefault(provider, []).append((index, doc.score))
        normalized = [0.0] * len(docs)
        for items in grouped.values():
            numeric = [score for _, score in items if score is not None]
            maximum = max(numeric) if numeric else 0.0
            minimum = min(numeric) if numeric else 0.0
            for provider_rank, (index, score) in enumerate(items, start=1):
                if score is not None and minimum >= 0 and maximum > 0:
                    # 检索分数通常已经是非负相关度；按最大值缩放可保留绝对 0，
                    # 避免 min-max 把候选中的最低正分强行压成 0。
                    normalized[index] = score / maximum
                elif score is not None and maximum > minimum:
                    normalized[index] = (score - minimum) / (maximum - minimum)
                elif score is not None:
                    normalized[index] = 0.0
                else:
                    # 无 provider 分数时使用倒数排名，仍保留原检索器的顺序信息。
                    normalized[index] = 1.0 / provider_rank
        return normalized

    @staticmethod
    def _freshness_score(published_at: str | None) -> float:
        """Map recent dates to a small bonus; unknown or unparsable dates remain neutral."""

        if not published_at:
            return 0.0
        published = parse_published_datetime(published_at)
        if published is None:
            return 0.0
        age_days = max(0.0, (datetime.now(timezone.utc) - published).total_seconds() / 86400)
        if age_days <= 365:
            return 1.0
        if age_days <= 365 * 3:
            return 0.5
        return 0.0

"""Deterministic evidence selection that preserves original citation objects."""

from __future__ import annotations

from pydantic import BaseModel, Field

from deepresearch.schema import Evidence


class CompressionResult(BaseModel):
    """Selected evidence and diagnostics for replay and cost analysis."""

    evidence: list[Evidence] = Field(default_factory=list)
    dropped_evidence_ids: list[str] = Field(default_factory=list)
    input_count: int = 0
    output_count: int = 0
    input_chars: int = 0
    output_chars: int = 0
    truncated: bool = False
    selected_credibility_counts: dict[str, int] = Field(default_factory=dict)
    dropped_low_credibility_ids: list[str] = Field(default_factory=list)


class EvidenceCompressor:
    """Select a compact, source-diverse evidence subset without rewriting quotes."""

    _METHOD_PRIORITY = {
        "llm": 4,
        # 数值恢复仍是逐字原文证据，优先级高于通用规则 fallback。
        "rule_numeric_recovery": 4,
        "rule_gap_recovery": 3,
        "rule_fallback": 2,
        "rule": 1,
    }

    def __init__(self, *, max_evidence: int = 12, max_chars: int = 12_000) -> None:
        if max_evidence < 1:
            raise ValueError("max_evidence must be at least 1.")
        if max_chars < 1:
            raise ValueError("max_chars must be positive.")
        self.max_evidence = max_evidence
        self.max_chars = max_chars

    def compress(
        self,
        evidence: list[Evidence],
        *,
        priority_evidence_ids: set[str] | None = None,
    ) -> CompressionResult:
        """Prefer required coverage, confidence and source diversity.

        ``priority_evidence_ids`` lets the orchestrator retain a minimal set of Evidence that is
        the only support for a named subject/dimension. Evidence objects and quote pointers remain
        untouched; this only changes deterministic selection order.
        """

        unique: list[tuple[int, Evidence]] = []
        seen: set[tuple[str, str]] = set()
        for index, item in enumerate(evidence):
            fingerprint = (item.source_id, item.quote.strip())
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            unique.append((index, item))

        priority_ids = priority_evidence_ids or set()
        ranked = sorted(
            unique,
            key=lambda pair: (
                0 if pair[1].id in priority_ids else 1,
                -self._quality_score(pair[1]),
                -self._METHOD_PRIORITY.get(pair[1].extraction_method, 0),
                pair[0],
            ),
        )
        # 先在覆盖必需证据内部保持来源多样性，再处理普通候选；否则一个普通来源可能
        # 把同源但唯一覆盖某个维度的证据挤出有限上下文。
        priority_ranked = [pair for pair in ranked if pair[1].id in priority_ids]
        normal_ranked = [pair for pair in ranked if pair[1].id not in priority_ids]
        priority_diverse, priority_remaining = self._source_diverse_order(priority_ranked)
        normal_diverse, normal_remaining = self._source_diverse_order(
            normal_ranked,
            initial_seen_sources={item.source_id for _, item in priority_ranked},
        )
        ordered = [
            *priority_diverse,
            *priority_remaining,
            *normal_diverse,
            *normal_remaining,
        ]

        selected: list[Evidence] = []
        output_chars = 0
        for _, item in ordered:
            item_chars = len(item.claim) + len(item.quote)
            if len(selected) >= self.max_evidence:
                break
            # 至少保留一条证据；后续条目不得突破字符预算。
            if selected and output_chars + item_chars > self.max_chars:
                continue
            selected.append(item)
            output_chars += item_chars

        selected_ids = {item.id for item in selected}
        dropped = [item.id for _, item in unique if item.id not in selected_ids]
        credibility_counts = {
            tier: sum(1 for item in selected if item.source_credibility_tier == tier)
            for tier in ("high", "medium", "low")
        }
        input_chars = sum(len(item.claim) + len(item.quote) for _, item in unique)
        return CompressionResult(
            evidence=selected,
            dropped_evidence_ids=dropped,
            input_count=len(unique),
            output_count=len(selected),
            input_chars=input_chars,
            output_chars=output_chars,
            truncated=len(selected) < len(unique),
            selected_credibility_counts=credibility_counts,
            dropped_low_credibility_ids=[
                item.id
                for _, item in unique
                if item.id not in selected_ids and item.source_credibility_tier == "low"
            ],
        )

    @staticmethod
    def _source_diverse_order(
        ranked: list[tuple[int, Evidence]],
        *,
        initial_seen_sources: set[str] | None = None,
    ) -> tuple[list[tuple[int, Evidence]], list[tuple[int, Evidence]]]:
        """Split a ranked group into first-per-source and remaining candidates."""

        diverse: list[tuple[int, Evidence]] = []
        remaining: list[tuple[int, Evidence]] = []
        seen_sources = set(initial_seen_sources or set())
        for pair in ranked:
            if pair[1].source_id not in seen_sources:
                diverse.append(pair)
                seen_sources.add(pair[1].source_id)
            else:
                remaining.append(pair)
        return diverse, remaining

    @staticmethod
    def _quality_score(evidence: Evidence) -> float:
        """Blend extraction confidence with source quality without treating either as truth."""

        credibility = (
            evidence.source_credibility_score
            if evidence.source_credibility_score is not None
            else 0.5
        )
        return 0.65 * evidence.confidence + 0.35 * credibility

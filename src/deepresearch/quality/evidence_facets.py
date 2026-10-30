"""Deterministic evidence obligations for evaluation-methodology questions.

These facets do not decide whether a claim is true.  They only prevent chunk
ranking and Reader early-stop from treating several generic findings as enough
when the subtask explicitly asks for a metric definition or error taxonomy.
"""

from __future__ import annotations

from collections.abc import Iterable


# Query triggers and source-language aliases deliberately cover only concepts
# that have stable operational meanings in this project's evaluation domain.
# Keeping the table small avoids turning it into a benchmark-answer lookup.
_EVIDENCE_FACETS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "citation_presence_scope": (
        (
            "引用覆盖",
            "覆盖率",
            "引用完整",
            "引用质量与幻觉率",
            "citation coverage",
            "citation completeness",
            "citation quality and hallucination",
            "citation recall",
        ),
        (
            "each statement cites",
            "every statement cites",
            "require citations",
            "requires citations",
            "citation coverage",
            "citation completeness",
            "citation recall",
            "需要引用的主张",
            "需要引用的陈述",
            "引用覆盖率",
            "引用完整性",
        ),
    ),
    "citation_semantic_support": (
        (
            "引用准确",
            "引用正确",
            "归因一致",
            "引用充分",
            "citation accuracy",
            "citation correctness",
            "citation quality",
            "attribution",
        ),
        (
            "well supported by the cited",
            "no irrelevant passages are cited",
            "faithfully follows cited passages",
            "natural language inference",
            "entailment",
            "引用是否真正支持",
            "引用蕴含",
            "无关引用",
        ),
    ),
    "hallucination_analysis_unit": (
        (
            "幻觉率",
            "幻觉测量",
            "幻觉评测",
            "幻觉检测",
            "hallucination rate",
            "hallucination measurement",
            "hallucination evaluation",
            "hallucination detection",
        ),
        (
            "word-level",
            "span-level",
            "claim-level",
            "atomic claim",
            "claims in llm responses into triples",
            "breaks down claims",
            "原子主张",
            "主张拆解",
            "词级",
            "片段级",
        ),
    ),
    "hallucination_support_relation": (
        (
            "幻觉率",
            "幻觉测量",
            "幻觉评测",
            "幻觉检测",
            "hallucination rate",
            "hallucination measurement",
            "hallucination evaluation",
            "hallucination detection",
        ),
        (
            "unsupported",
            "unfounded",
            "contradictory",
            "contradict the",
            "evident conflict",
            "subtle conflict",
            "evident introduction",
            "subtle introduction",
            "hallucination taxonomy",
            "support relation",
            "supporting passage",
            "无支持",
            "不受支持",
            "支撑关系",
            "支撑段落",
            "矛盾",
            "冲突",
            "幻觉分类",
        ),
    ),
}


def required_evidence_facets(query: str) -> tuple[str, ...]:
    """Return semantic evidence facets explicitly requested by ``query``."""

    lowered = query.casefold()
    return tuple(
        name
        for name, (triggers, _) in _EVIDENCE_FACETS.items()
        if any(trigger.casefold() in lowered for trigger in triggers)
    )


def covered_evidence_facets(
    required: Iterable[str],
    texts: Iterable[str],
) -> tuple[str, ...]:
    """Return required facets for which an explicit source phrase is present."""

    combined = "\n".join(texts).casefold()
    return tuple(
        name
        for name in required
        if name in _EVIDENCE_FACETS
        and any(
            alias.casefold() in combined
            for alias in _EVIDENCE_FACETS[name][1]
        )
    )


def chunk_facet_match_count(query: str, text: str) -> int:
    """Count requested facets explicitly represented in one source chunk."""

    required = required_evidence_facets(query)
    return len(covered_evidence_facets(required, [text]))

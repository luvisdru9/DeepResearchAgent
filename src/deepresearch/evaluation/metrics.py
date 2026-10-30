from __future__ import annotations

from urllib.parse import urlparse

from deepresearch.evaluation.benchmark import AtomicClaim
from deepresearch.evaluation.claims import (
    assess_claim_citations,
    assess_gold_claim_coverage,
    weighted_gold_claim_recall,
)
from deepresearch.quality import validate_report_citations
from deepresearch.schema import EvalResult, ResearchReport


def score_report(
    sample_id: str,
    report: ResearchReport,
    expected_claims: list[str],
    latency_seconds: float = 0.0,
    *,
    gold_claims: list[AtomicClaim] | None = None,
) -> EvalResult:
    """Score one generated report with deterministic MVP metrics.

    These metrics are intentionally simple so the smoke benchmark can run offline and remain reproducible.
    """

    text = report.markdown.lower()
    expected = [claim.lower() for claim in expected_claims]
    # 当前 fact_accuracy 用标准要点字符串包含关系近似，后续可替换成 LLM-as-Judge。
    matched = [claim for claim in expected if claim and claim in text]
    legacy_fact_accuracy = len(matched) / len(expected) if expected else 0.0
    gold_assessments = (
        assess_gold_claim_coverage(report, gold_claims) if gold_claims is not None else []
    )
    fact_accuracy = (
        weighted_gold_claim_recall(gold_assessments)
        if gold_claims is not None
        else legacy_fact_accuracy
    )
    validation = validate_report_citations(report, require_citations=True)
    assessments = assess_claim_citations(report, require_citations=True)
    required_count = len(assessments)
    cited = [item for item in assessments if item.citation_numbers]
    support_weight = sum(
        1.0 if item.status == "supported" else 0.5 if item.status == "partially_supported" else 0.0
        for item in assessments
    )
    citation_existence = len(cited) / required_count if required_count else 0.0
    citation_correctness = support_weight / len(cited) if cited else 0.0
    citation_completeness = support_weight / required_count if required_count else 0.0
    # 兼容旧消费者，但明确固定别名：coverage=existence，precision=correctness。
    citation_coverage = citation_existence
    citation_precision = citation_correctness
    answer_completeness = fact_accuracy
    status_counts = {
        status: sum(item.status == status for item in assessments)
        for status in (
            "supported",
            "partially_supported",
            "unsupported",
            "contradicted",
            "miscited",
            "uncited",
        )
    }
    hallucination_weight = (
        status_counts["unsupported"]
        + status_counts["contradicted"]
        + status_counts["miscited"]
        + status_counts["uncited"]
        + 0.5 * status_counts["partially_supported"]
    )
    hallucination_rate = hallucination_weight / required_count if required_count else 1.0
    # 本地语料的 URL 没有真实域名，因此用 netloc 或原始 source_url 统一计数。
    domains = {urlparse(item.source_url).netloc or item.source_url for item in report.evidence}

    return EvalResult(
        sample_id=sample_id,
        run_id=report.run_id,
        fact_accuracy=fact_accuracy,
        weighted_claim_recall=(fact_accuracy if gold_claims is not None else None),
        citation_coverage=citation_coverage,
        citation_precision=citation_precision,
        citation_existence=citation_existence,
        citation_correctness=citation_correctness,
        citation_completeness=citation_completeness,
        hallucination_rate=hallucination_rate,
        answer_completeness=answer_completeness,
        latency_seconds=latency_seconds,
        metadata={
            "source_diversity": len(domains),
            "evaluator": "rule_claim_support",
            "metric_protocol": "claim_citation_v2",
            "support_method": "lexical_overlap_v2",
            "citation_integrity_valid": validation.valid,
            "atomic_claim_count": required_count,
            "cited_claim_count": len(cited),
            "validly_cited_claim_count": sum(bool(item.evidence_ids) for item in assessments),
            "claim_status_counts": status_counts,
            "claim_assessments": [item.model_dump(mode="json") for item in assessments],
            "gold_claim_method": (
                "lexical_gold_claim_v1" if gold_claims is not None else "legacy_exact_string"
            ),
            "gold_claim_assessments": [
                item.model_dump(mode="json") for item in gold_assessments
            ],
            "weighted_gold_claim_recall": fact_accuracy if gold_claims is not None else None,
            "legacy_fact_accuracy": legacy_fact_accuracy,
            "metric_definitions": {
                "citation_existence": "claims with a citation marker / evidence-requiring claims",
                "citation_correctness": "lexical support weight / claims with a citation marker",
                "citation_completeness": "lexical support weight / evidence-requiring claims",
                "hallucination_rate": "evidence-grounding error weight / evidence-requiring claims",
            },
        },
    )

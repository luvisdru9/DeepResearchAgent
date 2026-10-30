"""Deterministic report-claim extraction and citation/Evidence binding.

The splitter is deliberately conservative: it separates sentence-like clauses but
does not pretend to perform semantic proposition decomposition.  Every decision is
stored on the extracted claim so later LLM or human judges can audit the mapping.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from deepresearch.quality import validate_report_citations
from deepresearch.schema import ResearchReport
from deepresearch.text import tokenize_mixed_text

if TYPE_CHECKING:
    from deepresearch.evaluation.benchmark import AtomicClaim

_CITATION_PATTERN = re.compile(r"\[(\d+)\]")
_CLAUSE_BOUNDARY = re.compile(r"(?<=[。！？!?；;])\s*|(?<=\.)\s+")


class AtomicReportClaim(BaseModel):
    """One stable report claim with an explicit citation-to-Evidence mapping."""

    id: str
    finding_index: int = Field(ge=1)
    clause_index: int = Field(ge=1)
    text: str
    citation_numbers: list[int] = Field(default_factory=list)
    valid_citation_numbers: list[int] = Field(default_factory=list)
    invalid_citation_numbers: list[int] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    citation_origin: Literal["clause", "finding_inherited", "none"] = "none"
    requires_citation: bool = True


class ClaimCitationAssessment(BaseModel):
    """Deterministic support signal for one atomic report claim.

    ``support_method`` is explicit because lexical overlap is a reproducible
    engineering signal, not a semantic entailment verdict.
    """

    claim_id: str
    finding_index: int = Field(ge=1)
    claim: str
    status: Literal[
        "supported",
        "partially_supported",
        "unsupported",
        "contradicted",
        "miscited",
        "uncited",
    ]
    citation_numbers: list[int] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    invalid_citation_numbers: list[int] = Field(default_factory=list)
    support_score: float = Field(ge=0.0, le=1.0)
    support_method: Literal["lexical_overlap_v2", "llm_judge", "human"]
    reason: str


class GoldClaimCoverage(BaseModel):
    """Provisional coverage of one authored benchmark claim by report claims."""

    gold_claim_id: str
    gold_claim: str
    required: bool
    weight: float = Field(gt=0.0)
    status: Literal["covered", "partially_covered", "not_covered"]
    score: float = Field(ge=0.0, le=1.0)
    report_claim_ids: list[str] = Field(default_factory=list)
    lexical_overlap: float = Field(ge=0.0, le=1.0)
    method: Literal["lexical_gold_claim_v1"] = "lexical_gold_claim_v1"


def extract_atomic_claims(
    report: ResearchReport,
    *,
    require_citations: bool = True,
) -> list[AtomicReportClaim]:
    """Split key findings conservatively and retain auditable Evidence bindings.

    A citation at the end of a finding is inherited by uncited clauses in that
    finding.  This mirrors common report-writing style, while ``citation_origin``
    makes the inference visible instead of silently treating it as local citation.
    """

    validation = validate_report_citations(report, require_citations=require_citations)
    claims: list[AtomicReportClaim] = []
    duplicate_counts: dict[str, int] = {}

    for finding_index, finding in enumerate(report.key_findings, start=1):
        finding_numbers = _ordered_unique(
            int(value) for value in _CITATION_PATTERN.findall(finding)
        )
        clauses = _split_finding(finding)
        for clause_index, clause in enumerate(clauses, start=1):
            local_numbers = _ordered_unique(
                int(value) for value in _CITATION_PATTERN.findall(clause)
            )
            if local_numbers:
                citation_numbers = local_numbers
                citation_origin: Literal["clause", "finding_inherited", "none"] = "clause"
            elif finding_numbers:
                citation_numbers = finding_numbers
                citation_origin = "finding_inherited"
            else:
                citation_numbers = []
                citation_origin = "none"

            text = _normalize_claim_text(clause)
            if not text:
                continue
            base_id = f"claim_{hashlib.sha256(text.encode('utf-8')).hexdigest()[:12]}"
            duplicate_counts[base_id] = duplicate_counts.get(base_id, 0) + 1
            occurrence = duplicate_counts[base_id]
            claim_id = base_id if occurrence == 1 else f"{base_id}_{occurrence}"
            valid_numbers = [
                number for number in citation_numbers if number in validation.citation_map
            ]
            invalid_numbers = [
                number for number in citation_numbers if number not in validation.citation_map
            ]
            claims.append(
                AtomicReportClaim(
                    id=claim_id,
                    finding_index=finding_index,
                    clause_index=clause_index,
                    text=text,
                    citation_numbers=citation_numbers,
                    valid_citation_numbers=valid_numbers,
                    invalid_citation_numbers=invalid_numbers,
                    evidence_ids=[validation.citation_map[number] for number in valid_numbers],
                    citation_origin=citation_origin,
                    requires_citation=require_citations,
                )
            )
    return claims


def assess_claim_citations(
    report: ResearchReport,
    *,
    require_citations: bool = True,
    supported_threshold: float = 0.25,
    partial_threshold: float = 0.10,
) -> list[ClaimCitationAssessment]:
    """Return transparent provisional support labels for atomic report claims."""

    evidence_by_id = {item.id: item for item in report.evidence}
    assessments: list[ClaimCitationAssessment] = []
    for claim in extract_atomic_claims(report, require_citations=require_citations):
        support_scores: list[float] = []
        claim_terms = tokenize_mixed_text(claim.text)
        for evidence_id in claim.evidence_ids:
            evidence = evidence_by_id[evidence_id]
            evidence_terms = tokenize_mixed_text(f"{evidence.claim} {evidence.quote}")
            score = (
                len(claim_terms.intersection(evidence_terms)) / len(claim_terms)
                if claim_terms
                else 0.0
            )
            support_scores.append(score)
        best_score = max(support_scores, default=0.0)

        if not claim.citation_numbers:
            status = "uncited"
            reason = "The claim has no citation marker."
        elif not claim.evidence_ids:
            status = "miscited"
            reason = "Citation markers do not resolve to any report Evidence item."
        elif best_score >= supported_threshold:
            status = "supported"
            reason = "At least one cited Evidence item meets the lexical support threshold."
        elif best_score >= partial_threshold:
            status = "partially_supported"
            reason = "Cited Evidence has limited lexical overlap with the claim."
        else:
            status = "unsupported"
            reason = "Valid citations exist, but none meets the lexical support threshold."

        assessments.append(
            ClaimCitationAssessment(
                claim_id=claim.id,
                finding_index=claim.finding_index,
                claim=claim.text,
                status=status,
                citation_numbers=claim.citation_numbers,
                evidence_ids=claim.evidence_ids,
                invalid_citation_numbers=claim.invalid_citation_numbers,
                support_score=best_score,
                support_method="lexical_overlap_v2",
                reason=reason,
            )
        )
    return assessments


def assess_gold_claim_coverage(
    report: ResearchReport,
    gold_claims: list[AtomicClaim],
    *,
    covered_threshold: float = 0.45,
    partial_threshold: float = 0.20,
) -> list[GoldClaimCoverage]:
    """Match authored claims to report claims without reducing gold data to strings.

    This deterministic matcher is intentionally provisional.  It provides stable
    engineering regression metrics; B3 human calibration determines whether its
    thresholds are acceptable for semantic use.
    """

    report_claims = extract_atomic_claims(report, require_citations=False)
    assessments: list[GoldClaimCoverage] = []
    for gold in gold_claims:
        variants = [gold.text, *gold.accepted_variants]
        best_overlap = 0.0
        best_report_ids: list[str] = []
        for report_claim in report_claims:
            overlap = max(
                _directional_term_overlap(variant, report_claim.text)
                for variant in variants
            )
            if overlap > best_overlap:
                best_overlap = overlap
                best_report_ids = [report_claim.id]
            elif overlap == best_overlap and overlap > 0.0:
                best_report_ids.append(report_claim.id)
        if best_overlap >= covered_threshold:
            status = "covered"
            score = 1.0
        elif best_overlap >= partial_threshold:
            status = "partially_covered"
            score = 0.5
        else:
            status = "not_covered"
            score = 0.0
            best_report_ids = []
        assessments.append(
            GoldClaimCoverage(
                gold_claim_id=gold.id,
                gold_claim=gold.text,
                required=gold.required,
                weight=gold.weight,
                status=status,
                score=score,
                report_claim_ids=list(dict.fromkeys(best_report_ids)),
                lexical_overlap=best_overlap,
            )
        )
    return assessments


def weighted_gold_claim_recall(assessments: list[GoldClaimCoverage]) -> float:
    """Compute weighted recall over required gold claims (or all if none are required)."""

    active = [item for item in assessments if item.required] or assessments
    total_weight = sum(item.weight for item in active)
    if total_weight == 0.0:
        return 0.0
    return sum(item.weight * item.score for item in active) / total_weight


def _split_finding(finding: str) -> list[str]:
    """Return sentence-like clauses while keeping citation-only tails attached."""

    raw_parts = [part.strip() for part in _CLAUSE_BOUNDARY.split(finding) if part.strip()]
    parts: list[str] = []
    for part in raw_parts:
        # ``结论。 [1]`` may split into a citation-only tail; it belongs to the
        # preceding clause and must not become a zero-content claim.
        if not _normalize_claim_text(part) and parts:
            parts[-1] = f"{parts[-1]} {part}"
        else:
            parts.append(part)
    return parts


def _normalize_claim_text(text: str) -> str:
    without_citations = _CITATION_PATTERN.sub("", text)
    return re.sub(r"\s+", " ", without_citations).strip(" \t\r\n-*•")


def _ordered_unique(values: Iterable[int]) -> list[int]:
    return list(dict.fromkeys(values))


def _directional_term_overlap(expected: str, observed: str) -> float:
    """Measure how much of an authored formulation is present in one report claim."""

    expected_terms = tokenize_mixed_text(expected)
    if not expected_terms:
        return 0.0
    observed_terms = tokenize_mixed_text(observed)
    return len(expected_terms.intersection(observed_terms)) / len(expected_terms)

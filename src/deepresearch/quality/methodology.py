"""Deterministic obligations for citation/hallucination evaluation reports.

The checks are deliberately narrow. They verify that a user-facing key finding
contains an operational definition; they do not attempt to decide whether an
open-domain methodological recommendation is universally correct.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from typing import Literal

from pydantic import BaseModel, Field

from deepresearch.quality.evidence_facets import (
    covered_evidence_facets,
    required_evidence_facets,
)
from deepresearch.schema import Evidence

MethodologyObligationType = Literal[
    "citation_coverage_definition",
    "citation_correctness_definition",
    "hallucination_error_accounting",
    "evidence_grounded_blue_revision",
    "post_revision_verification",
]
MethodologyObligationStatus = Literal[
    "satisfied",
    "missing",
    "evidence_insufficient",
]

_ATOMIC_UNIT = re.compile(
    r"(?:原子主张|原子化主张|需(?:要)?证据(?:支持)?的(?:关键)?主张|"
    r"evidence[- ]requiring\s+(?:atomic\s+)?claims?|atomic\s+claims?)",
    re.IGNORECASE,
)
_CITATION_PRESENCE = re.compile(
    r"(?:有(?:效)?引用|带有引用|附有引用|引用标记|with\s+(?:a\s+)?valid\s+citation)",
    re.IGNORECASE,
)
_RATE_DENOMINATOR = re.compile(
    r"(?:比例|占比|比率|覆盖率|分子|分母|总数|proportion|rate|numerator|denominator)",
    re.IGNORECASE,
)
_SEMANTIC_SUPPORT = re.compile(
    r"(?:语义(?:上)?(?:支持|蕴含|一致)|真正(?:蕴含|支持)|证据(?:蕴含|支持)|"
    r"semantic\s+(?:support|entailment)|entail(?:s|ment)?|supports?\s+the\s+claim)",
    re.IGNORECASE,
)
_NOT_MARKER_ONLY = re.compile(
    r"(?:不能只|不(?:能|应)?仅|不等于|而非仅|不是只|"
    r"not\s+(?:merely|only)|rather\s+than\s+(?:a\s+)?(?:marker|link))",
    re.IGNORECASE,
)
_ERROR_CATEGORY_PATTERNS: dict[str, re.Pattern[str]] = {
    "unsupported": re.compile(r"(?:无支持|不受支持|unsupported|unfounded)", re.IGNORECASE),
    "contradicted": re.compile(r"(?:矛盾|冲突|contradict(?:ed|ory|ion)?)", re.IGNORECASE),
    "miscited": re.compile(r"(?:错引|误引|引用错配|miscited|mis-cited)", re.IGNORECASE),
    "uncited": re.compile(r"(?:无引用|未引用|uncited)", re.IGNORECASE),
}
_ABSENCE_CLAIM = re.compile(
    r"(?:来源|论文|研究|材料|证据|文档|报告|数据).{0,20}"
    r"(?:没有(?:提供|包含|报告|说明|附带)|未(?:直接)?(?:提供|包含|附|报告|说明|定义|给出)|"
    r"缺少|缺乏|不包含|不提供|"
    r"does\s+not\s+(?:provide|report|include|describe|state)|"
    r"did\s+not\s+(?:provide|report|include|describe|state)|lacks?|"
    r"without\s+(?:data|experiments?|definitions?|validation)|"
    r"no\s+(?:data|experiments?|definitions?|validation))",
    re.IGNORECASE,
)
_DIRECT_ABSENCE = re.compile(
    r"(?:没有(?:提供|包含|报告|说明|附带)|未(?:直接)?(?:提供|包含|附|报告|说明|定义|给出)|"
    r"缺少|缺乏|不包含|不提供|"
    r"does\s+not\s+(?:provide|report|include|describe|state)|"
    r"did\s+not\s+(?:provide|report|include|describe|state)|lacks?|"
    r"without\s+(?:data|experiments?|definitions?|validation)|"
    r"no\s+(?:data|experiments?|definitions?|validation))",
    re.IGNORECASE,
)
_REVERSED_ABSENCE = re.compile(
    r"(?:没有|未|并未|未能).{0,28}"
    r"(?:来源|论文|研究|材料|证据|文档|报告|数据).{0,16}"
    r"(?:中)?(?:提供|包含|报告|说明|附带|定义|给出|验证)",
    re.IGNORECASE,
)
# “尚缺实证数据验证”省略了“来源/论文”等主语，但仍然是在把当前
# 检索范围内未见材料上升为来源事实。该窄规则只覆盖实证/实验验证缺失，
# 不把一般的“仍需进一步研究”建议误判为缺失性断言。
_UNATTRIBUTED_EMPIRICAL_ABSENCE = re.compile(
    r"(?:尚缺|缺少|缺乏|没有|未有).{0,16}"
    r"(?:实证|实验)(?:数据|结果|证据|研究|验证)",
    re.IGNORECASE,
)
_BLUE_REVISION = re.compile(
    r"(?=.*(?:Blue(?:\s+Agent)?|蓝方))(?=.*(?:证据|Evidence))"
    r"(?=.*(?:修订|修改|改写|revise|revision)).+",
    re.IGNORECASE | re.DOTALL,
)
_POST_REVISION_VERIFICATION = re.compile(
    r"(?=.*Verifier)(?=.*(?:修订后|修订结果|after\s+(?:the\s+)?revision))"
    r"(?=.*(?:核验|复核|检查|verify|verification))"
    r"(?=.*(?:不确定|未解决|uncertaint|unresolved)).+",
    re.IGNORECASE | re.DOTALL,
)


class MethodologyObligationDiagnostic(BaseModel):
    """One required operational definition and its report/evidence linkage."""

    obligation_type: MethodologyObligationType
    status: MethodologyObligationStatus
    matched_finding_indexes: list[int] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    reason: str


class SourceAbsenceDiagnostic(BaseModel):
    """Whether a key finding's source-absence assertion is directly evidenced."""

    finding_index: int
    text: str
    status: Literal["not_applicable", "directly_supported", "unverified"]
    evidence_ids: list[str] = Field(default_factory=list)
    reason: str


FindingInput = tuple[str, list[str]]


def _citation_methodology_obligations_apply(query: str) -> bool:
    """Recognize citation/hallucination evaluation questions."""

    required = set(required_evidence_facets(query))
    complete_trigger = {
        "citation_presence_scope",
        "citation_semantic_support",
        "hallucination_analysis_unit",
        "hallucination_support_relation",
    }.issubset(required)
    lowered = query.casefold()
    paired_topic = (
        any(term in lowered for term in ("引用质量", "引用评测", "citation quality"))
        and any(term in lowered for term in ("幻觉", "hallucination"))
    )
    return complete_trigger or paired_topic


def _red_blue_process_obligations_apply(query: str) -> bool:
    """Recognize questions that explicitly ask how Red-Blue verification works."""

    lowered = query.casefold()
    red_blue = any(term in lowered for term in ("red-blue", "red blue", "红蓝", "对抗核查"))
    process = any(
        term in lowered
        for term in ("核查", "验证", "改进", "报告", "verification", "verify", "report")
    )
    return red_blue and process


def methodology_obligations_apply(query: str) -> bool:
    """Recognize report families with deterministic content obligations."""

    return _citation_methodology_obligations_apply(
        query
    ) or _red_blue_process_obligations_apply(query)


def assess_methodology_obligations(
    query: str,
    findings: Iterable[FindingInput],
    evidence: list[Evidence],
) -> list[MethodologyObligationDiagnostic]:
    """Check applicable operational definitions only in cited key findings."""

    if not methodology_obligations_apply(query):
        return []
    material = list(findings)
    evidence_by_id = {item.id: item for item in evidence}
    citation_obligations = _citation_methodology_obligations_apply(query)
    process_obligations = _red_blue_process_obligations_apply(query)
    required_facets = tuple(
        dict.fromkeys(
            [
                *required_evidence_facets(query),
                *(
                    (
                        "citation_presence_scope",
                        "citation_semantic_support",
                        "hallucination_analysis_unit",
                        "hallucination_support_relation",
                    )
                    if citation_obligations
                    else ()
                ),
            ]
        )
    )
    covered_facets = set(
        covered_evidence_facets(
            required_facets,
            (f"{item.claim}\n{item.quote}" for item in evidence),
        )
    )
    specifications: list[
        tuple[
            MethodologyObligationType,
            tuple[str, ...],
            Callable[[str], bool],
            Callable[[str], bool] | None,
        ]
    ] = []
    if citation_obligations:
        specifications.extend(
            [
                (
                    "citation_coverage_definition",
                    ("citation_presence_scope",),
                    lambda text: bool(
                        _ATOMIC_UNIT.search(text)
                        and _CITATION_PRESENCE.search(text)
                        and _RATE_DENOMINATOR.search(text)
                    ),
                    None,
                ),
                (
                    "citation_correctness_definition",
                    ("citation_semantic_support",),
                    lambda text: bool(
                        _SEMANTIC_SUPPORT.search(text) and _NOT_MARKER_ONLY.search(text)
                    ),
                    None,
                ),
                (
                    "hallucination_error_accounting",
                    ("hallucination_analysis_unit", "hallucination_support_relation"),
                    lambda text: bool(
                        _ATOMIC_UNIT.search(text)
                        and all(
                            pattern.search(text)
                            for pattern in _ERROR_CATEGORY_PATTERNS.values()
                        )
                    ),
                    None,
                ),
            ]
        )
    if process_obligations:
        specifications.extend(
            [
                (
                    "evidence_grounded_blue_revision",
                    (),
                    lambda text: bool(_BLUE_REVISION.search(text)),
                    lambda text: bool(_BLUE_REVISION.search(text)),
                ),
                (
                    "post_revision_verification",
                    (),
                    lambda text: bool(_POST_REVISION_VERIFICATION.search(text)),
                    lambda text: bool(_POST_REVISION_VERIFICATION.search(text)),
                ),
            ]
        )
    diagnostics: list[MethodologyObligationDiagnostic] = []
    for obligation_type, needed_facets, matcher, evidence_matcher in specifications:
        matched = [
            index
            for index, (text, evidence_ids) in enumerate(material, start=1)
            if evidence_ids
            and matcher(text)
            and (
                evidence_matcher is None
                or any(
                    evidence_id in evidence_by_id
                    and evidence_matcher(
                        f"{evidence_by_id[evidence_id].claim}\n"
                        f"{evidence_by_id[evidence_id].quote}"
                    )
                    for evidence_id in evidence_ids
                )
            )
        ]
        matched_evidence_ids = list(
            dict.fromkeys(
                evidence_id
                for index in matched
                for evidence_id in material[index - 1][1]
                if evidence_id in evidence_by_id
            )
        )
        if matched:
            status: MethodologyObligationStatus = "satisfied"
            reason = "A cited key finding states the complete operational obligation."
        elif (
            all(facet in covered_facets for facet in needed_facets)
            if evidence_matcher is None
            else any(
                evidence_matcher(f"{item.claim}\n{item.quote}") for item in evidence
            )
        ):
            status = "missing"
            reason = "Supporting Evidence facets exist, but no cited key finding states the definition."
        else:
            status = "evidence_insufficient"
            reason = "The available Evidence does not cover every premise needed for this definition."
        diagnostics.append(
            MethodologyObligationDiagnostic(
                obligation_type=obligation_type,
                status=status,
                matched_finding_indexes=matched,
                evidence_ids=matched_evidence_ids,
                reason=reason,
            )
        )
    return diagnostics


def assess_source_absence_claims(
    findings: Iterable[FindingInput],
    evidence: list[Evidence],
) -> list[SourceAbsenceDiagnostic]:
    """Reject absence claims inferred only from the current retrieval window."""

    evidence_by_id = {item.id: item for item in evidence}
    diagnostics: list[SourceAbsenceDiagnostic] = []
    for index, (text, evidence_ids) in enumerate(findings, start=1):
        if not (
            _ABSENCE_CLAIM.search(text)
            or _REVERSED_ABSENCE.search(text)
            or _UNATTRIBUTED_EMPIRICAL_ABSENCE.search(text)
        ):
            diagnostics.append(
                SourceAbsenceDiagnostic(
                    finding_index=index,
                    text=text,
                    status="not_applicable",
                    evidence_ids=evidence_ids,
                    reason="No source-absence assertion detected.",
                )
            )
            continue
        directly_supported = [
            evidence_id
            for evidence_id in evidence_ids
            if evidence_id in evidence_by_id
            and (
                _DIRECT_ABSENCE.search(
                    f"{evidence_by_id[evidence_id].claim}\n"
                    f"{evidence_by_id[evidence_id].quote}"
                )
                or _REVERSED_ABSENCE.search(
                    f"{evidence_by_id[evidence_id].claim}\n"
                    f"{evidence_by_id[evidence_id].quote}"
                )
                or _UNATTRIBUTED_EMPIRICAL_ABSENCE.search(
                    f"{evidence_by_id[evidence_id].claim}\n"
                    f"{evidence_by_id[evidence_id].quote}"
                )
            )
        ]
        status = "directly_supported" if directly_supported else "unverified"
        diagnostics.append(
            SourceAbsenceDiagnostic(
                finding_index=index,
                text=text,
                status=status,
                evidence_ids=directly_supported,
                reason=(
                    "Mapped Evidence explicitly states the absence."
                    if directly_supported
                    else "Absence appears inferred from retrieval scope rather than asserted by Evidence."
                ),
            )
        )
    return diagnostics


def report_finding_inputs(
    findings: list[str],
    evidence: list[Evidence],
) -> list[FindingInput]:
    """Map rendered citation numbers back to Evidence IDs for deterministic gates."""

    mapped: list[FindingInput] = []
    for finding in findings:
        citation_numbers = [int(value) for value in re.findall(r"\[(\d+)\]", finding)]
        evidence_ids = [
            evidence[number - 1].id
            for number in citation_numbers
            if 1 <= number <= len(evidence)
        ]
        mapped.append((finding, list(dict.fromkeys(evidence_ids))))
    return mapped

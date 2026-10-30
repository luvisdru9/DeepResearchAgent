"""Deterministic substance checks for user-facing report findings.

These checks intentionally recognize only high-precision failure shapes.  They
complement, rather than replace, semantic RedTeam/Verifier review.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel

from deepresearch.schema import Evidence
from deepresearch.text import tokenize_mixed_text

FindingSubstanceStatus = Literal[
    "substantive",
    "bibliographic_only",
    "truncated_fragment",
    "non_propositional",
]

_CITATION_PATTERN = re.compile(r"\s*(?:\[\d+\]\s*)+$")
_TRUNCATED_END = re.compile(
    r"(?:\.{3,}|…+)(?:\s*\((?:truncated|截断)\))?\s*$|\((?:truncated|截断)\)\s*$",
    re.IGNORECASE,
)
_HEADING_ONLY = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:abstract|摘要|introduction|引言|keywords?|关键词)\s*[:：-]?\s*$",
    re.IGNORECASE,
)
_HEADING_PREFIX = re.compile(
    r"^\s*(?:abstract|摘要|keywords?|关键词)\s*[:：]",
    re.IGNORECASE,
)
_BIBLIOGRAPHIC_PHRASES = re.compile(
    r"(?:论文(?:题为|名为|发表于)|作者(?:是|为)|发表于|出版于|"
    r"paper\s+(?:titled|authored\s+by|published\s+in)|"
    r"authored\s+by|published\s+(?:in|at))",
    re.IGNORECASE,
)
_AUTHOR_LIST_PREFIX = re.compile(
    r"^\s*(?:authors?|作者)\s*[:：]\s*[^。！？.!?]+$",
    re.IGNORECASE,
)
_RESULT_TERMS = re.compile(
    r"(?:发现|结果|表明|显示|证明|提高|降低|优于|弱于|相关|导致|支持|反驳|"
    r"find(?:s|ings)?|found|result(?:s)?|show(?:s|ed)?|demonstrat(?:e|es|ed)|"
    r"improv(?:e|es|ed)|reduc(?:e|es|ed)|outperform(?:s|ed)?|correlat(?:e|es|ed))",
    re.IGNORECASE,
)
_BIBLIOMETRIC_QUERY_TERMS = re.compile(
    r"(?:文献计量|参考文献列表|文献清单|论文列表|引用次数|被引次数|"
    r"bibliometr(?:ic|y)|bibliography|citation\s+count|list\s+(?:the\s+)?papers)",
    re.IGNORECASE,
)


class FindingSubstanceDiagnostic(BaseModel):
    """One auditable deterministic judgment about a key finding."""

    finding_index: int
    text: str
    status: FindingSubstanceStatus
    reason: str
    allowed_by_query: bool = False


def assess_finding_substance(
    query: str,
    text: str,
    *,
    finding_index: int,
    evidence: list[Evidence] | None = None,
) -> FindingSubstanceDiagnostic:
    """Classify one finding without attempting open-ended semantic judgment."""

    normalized = _normalize_finding(text)
    bibliographic_allowed = bool(_BIBLIOMETRIC_QUERY_TERMS.search(query))
    if _TRUNCATED_END.search(normalized):
        return FindingSubstanceDiagnostic(
            finding_index=finding_index,
            text=text,
            status="truncated_fragment",
            reason="Finding ends with an explicit ellipsis/truncation marker.",
        )
    if not normalized or _HEADING_ONLY.fullmatch(normalized) or _HEADING_PREFIX.match(normalized):
        return FindingSubstanceDiagnostic(
            finding_index=finding_index,
            text=text,
            status="non_propositional",
            reason="Finding is empty, a section heading, or an abstract/keyword header.",
        )

    matches_source_title = _matches_source_title(normalized, evidence or [])
    metadata_only = (
        bool(_BIBLIOGRAPHIC_PHRASES.search(normalized))
        or bool(_AUTHOR_LIST_PREFIX.fullmatch(normalized))
    ) and not bool(_RESULT_TERMS.search(normalized))
    if matches_source_title or metadata_only:
        if bibliographic_allowed:
            return FindingSubstanceDiagnostic(
                finding_index=finding_index,
                text=text,
                status="substantive",
                reason="Bibliographic identity is explicitly requested by the research query.",
                allowed_by_query=True,
            )
        return FindingSubstanceDiagnostic(
            finding_index=finding_index,
            text=text,
            status="bibliographic_only",
            reason=(
                "Finding provides source identity/metadata but no supported research result; "
                "the query does not request bibliometrics."
            ),
        )

    return FindingSubstanceDiagnostic(
        finding_index=finding_index,
        text=text,
        status="substantive",
        reason="No deterministic non-substantive pattern was detected.",
    )


def assess_report_findings(
    query: str,
    findings: list[str],
    evidence: list[Evidence] | None = None,
) -> list[FindingSubstanceDiagnostic]:
    """Classify all findings in stable report order."""

    return [
        assess_finding_substance(
            query,
            finding,
            finding_index=index,
            evidence=evidence,
        )
        for index, finding in enumerate(findings, start=1)
    ]


def _normalize_finding(text: str) -> str:
    without_citations = _CITATION_PATTERN.sub("", text.strip())
    return re.sub(r"\s+", " ", without_citations).strip(" -*`\t\r\n")


def _matches_source_title(text: str, evidence: list[Evidence]) -> bool:
    """Recognize a title restatement conservatively using token containment."""

    finding_terms = tokenize_mixed_text(text)
    if len(finding_terms) < 3:
        return False
    for item in evidence:
        title_terms = tokenize_mixed_text(item.source_title)
        if len(title_terms) < 3:
            continue
        overlap = len(finding_terms.intersection(title_terms))
        # A title-only finding is almost entirely explained by the source title
        # and contains no explicit result predicate.
        if overlap / len(finding_terms) >= 0.85 and not _RESULT_TERMS.search(text):
            return True
    return False

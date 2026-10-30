"""本地引用完整性校验，避免把确定性格式判断交给 LLM。"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from deepresearch.schema import ResearchReport

_CITATION_PATTERN = re.compile(r"\[(\d+)\]")
# 证据条目必须从行首开始；证据原文内部的编号列表会被渲染为缩进行，不能误算成新条目。
_TABLE_ENTRY_PATTERN = re.compile(r"^(\d+)\.\s", re.MULTILINE)


class CitationValidation(BaseModel):
    """Replay-safe result of deterministic report citation validation."""

    valid: bool
    evidence_count: int
    evidence_table_entry_count: int
    cited_numbers: list[int] = Field(default_factory=list)
    invalid_numbers: list[int] = Field(default_factory=list)
    uncited_finding_indexes: list[int] = Field(default_factory=list)
    citation_map: dict[int, str] = Field(default_factory=dict)
    issues: list[str] = Field(default_factory=list)


def validate_report_citations(
    report: ResearchReport,
    *,
    require_citations: bool = True,
) -> CitationValidation:
    """Validate local citation numbering and evidence-table shape without semantic guessing."""

    evidence_count = len(report.evidence)
    citation_map = {
        index: evidence.id
        for index, evidence in enumerate(report.evidence, start=1)
    }
    cited_numbers: list[int] = []
    invalid_numbers: set[int] = set()
    uncited_finding_indexes: list[int] = []

    for index, finding in enumerate(report.key_findings, start=1):
        numbers = [int(value) for value in _CITATION_PATTERN.findall(finding)]
        cited_numbers.extend(numbers)
        invalid_numbers.update(
            number for number in numbers if number < 1 or number > evidence_count
        )
        if require_citations and evidence_count and not numbers:
            uncited_finding_indexes.append(index)

    evidence_section = _markdown_section(report.markdown, "证据表")
    table_numbers = [int(value) for value in _TABLE_ENTRY_PATTERN.findall(evidence_section)]
    issues: list[str] = []
    if invalid_numbers:
        issues.append(
            "Citation numbers outside the evidence table: "
            + ", ".join(str(number) for number in sorted(invalid_numbers))
        )
    if uncited_finding_indexes:
        issues.append(
            "Key findings without citations: "
            + ", ".join(str(index) for index in uncited_finding_indexes)
        )
    expected_table_numbers = list(range(1, evidence_count + 1))
    if table_numbers != expected_table_numbers:
        issues.append(
            "Evidence table numbering does not match report.evidence order: "
            f"expected {expected_table_numbers}, got {table_numbers}."
        )

    return CitationValidation(
        valid=not issues,
        evidence_count=evidence_count,
        evidence_table_entry_count=len(table_numbers),
        cited_numbers=list(dict.fromkeys(cited_numbers)),
        invalid_numbers=sorted(invalid_numbers),
        uncited_finding_indexes=uncited_finding_indexes,
        citation_map=citation_map,
        issues=issues,
    )


def _markdown_section(markdown: str, heading: str) -> str:
    """Extract one level-two Markdown section without parsing unrelated numbered lists."""

    marker = f"## {heading}"
    lines = markdown.splitlines()
    try:
        start = lines.index(marker) + 1
    except ValueError:
        return ""
    end = len(lines)
    for index in range(start, len(lines)):
        if lines[index].startswith("## "):
            end = index
            break
    return "\n".join(lines[start:end])

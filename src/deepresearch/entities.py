"""Deterministic named-object extraction shared by planning and retrieval diagnostics."""

from __future__ import annotations

import re


# 这些缩写通常描述技术类别而不是比较题中的具体研究对象，不能占用专属检索分支。
_GENERIC_TECHNICAL_TOKENS = {
    "AI",
    "API",
    "CPU",
    "DAG",
    "GPU",
    "HTML",
    "HTTP",
    "HTTPS",
    "JSON",
    "LLM",
    "NLP",
    "PDF",
    "QA",
    "RAG",
    "SQL",
    "URI",
    "URL",
    "XML",
}


def extract_named_research_entities(query: str) -> list[str]:
    """Extract acronym, CamelCase, and ``<name> Bench`` research objects in query order."""

    positioned: list[tuple[int, str]] = []
    compound_spans: list[tuple[int, int]] = []
    compound_pattern = re.compile(
        r"(?<![A-Za-z0-9])([A-Za-z][A-Za-z0-9._-]{2,30}\s+"
        r"(?:Bench|Benchmark|Gym|Eval))(?![A-Za-z0-9])"
    )
    for match in compound_pattern.finditer(query):
        candidate = " ".join(match.group(1).split())
        base = candidate.split()[0]
        if _looks_named(base):
            positioned.append((match.start(), candidate))
            compound_spans.append(match.span())

    token_pattern = re.compile(r"(?<![A-Za-z0-9])([A-Za-z][A-Za-z0-9._-]{2,30})")
    for match in token_pattern.finditer(query):
        if any(start <= match.start() < end for start, end in compound_spans):
            continue
        candidate = match.group(1).strip("._-")
        if not _looks_named(candidate) or candidate.upper() in _GENERIC_TECHNICAL_TOKENS:
            continue
        positioned.append((match.start(), candidate))

    entities: list[str] = []
    for _, candidate in sorted(positioned, key=lambda item: item[0]):
        if candidate.casefold() not in {item.casefold() for item in entities}:
            entities.append(candidate)
    return entities


def _looks_named(candidate: str) -> bool:
    """Return whether one Latin token carries an acronym/CamelCase identity signal."""

    uppercase_count = sum(character.isupper() for character in candidate)
    return (
        len(candidate) >= 3
        and (
            candidate.isupper()
            or uppercase_count >= 2
            or any(character.isdigit() for character in candidate)
        )
    )

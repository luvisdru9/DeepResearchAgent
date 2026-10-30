"""Deterministic risk signals for instructions embedded in retrieved documents.

This scanner is deliberately conservative: it records heuristic signals for audit and
never claims that a document is a confirmed prompt-injection attack.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Literal

from pydantic import BaseModel, Field

from deepresearch.config import ContentSafetyConfig
from deepresearch.schema import SourceDoc


ContentRiskLevel = Literal["safe", "suspicious", "high"]


class ContentSafetyAssessment(BaseModel):
    """Replay-safe source assessment; matched source text is intentionally omitted."""

    risk_level: ContentRiskLevel
    signal_categories: list[str] = Field(default_factory=list)
    signal_count: int = Field(ge=0)
    scanned_fields: list[str] = Field(default_factory=list)
    method: Literal["rule"] = "rule"


# Patterns require instruction-like wording where practical, reducing false positives for
# ordinary papers that merely discuss prompt injection, credentials, or tool use.
_CATEGORY_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "instruction_override": (
        re.compile(r"\b(?:ignore|disregard|override)\s+(?:all\s+)?(?:previous|prior|above|system)\s+(?:instructions?|prompts?|messages?)\b", re.I),
        re.compile(r"\bforget\s+(?:all\s+)?(?:previous|prior)\s+(?:instructions?|prompts?)\b", re.I),
        re.compile(r"(?:忽略|无视|覆盖|忘记).{0,12}(?:之前|先前|以上|系统).{0,8}(?:指令|提示|消息)", re.I),
    ),
    "role_impersonation": (
        re.compile(r"\b(?:you\s+are\s+now|act\s+as|pretend\s+to\s+be)\s+(?:the\s+)?(?:system|developer|administrator|root|assistant)\b", re.I),
        re.compile(r"\b(?:system|developer)\s+message\s*:", re.I),
        re.compile(r"(?:你现在是|扮演|假装成为).{0,10}(?:系统|开发者|管理员|root|助手)", re.I),
        re.compile(r"(?:系统|开发者)消息\s*[：:]", re.I),
    ),
    "credential_solicitation": (
        re.compile(r"\b(?:reveal|print|show|send|upload|return|expose)\b.{0,32}\b(?:api[-_ ]?keys?|passwords?|tokens?|credentials?|environment variables?|secrets?)\b", re.I),
        re.compile(r"\b(?:api[-_ ]?keys?|passwords?|tokens?|credentials?|environment variables?|secrets?)\b.{0,24}\b(?:reveal|print|show|send|upload|return|expose)\b", re.I),
        re.compile(r"(?:显示|输出|发送|上传|返回|泄露|提供).{0,24}(?:API[ _-]?密钥|密钥|密码|令牌|凭据|环境变量|秘密)", re.I),
    ),
    "tool_execution": (
        re.compile(r"\b(?:run|execute)\s+(?:this\s+)?(?:shell\s+)?(?:command|script|code)\b", re.I),
        re.compile(r"\b(?:call|invoke|use)\s+(?:the\s+)?(?:tool|function)\b.{0,28}\b(?:now|immediately|with)\b", re.I),
        re.compile(r"(?:运行|执行).{0,10}(?:命令|脚本|代码)", re.I),
        re.compile(r"(?:调用|使用).{0,10}(?:工具|函数).{0,16}(?:立即|现在|参数)", re.I),
    ),
}


def assess_content_safety(
    doc: SourceDoc,
    *,
    scan_titles: bool = True,
) -> SourceDoc:
    """Attach deterministic risk metadata without changing source content or provenance."""

    fields = {"content": doc.content}
    if scan_titles:
        fields["title"] = doc.title

    categories = sorted(
        category
        for category, patterns in _CATEGORY_PATTERNS.items()
        if any(pattern.search(value) for value in fields.values() for pattern in patterns)
    )
    risk_level: ContentRiskLevel
    if len(categories) >= 2:
        risk_level = "high"
    elif categories:
        risk_level = "suspicious"
    else:
        risk_level = "safe"

    assessment = ContentSafetyAssessment(
        risk_level=risk_level,
        signal_categories=categories,
        signal_count=len(categories),
        scanned_fields=list(fields),
    )
    metadata = dict(doc.metadata)
    metadata["content_safety"] = assessment.model_dump(mode="json")
    return doc.model_copy(update={"metadata": metadata})


def apply_content_safety(
    docs: Iterable[SourceDoc],
    config: ContentSafetyConfig,
) -> tuple[list[SourceDoc], dict[str, object]]:
    """Assess sources, optionally exclude high-risk ones, and return Replay-safe summary."""

    source_docs = list(docs)
    if config.enabled:
        assessed = [
            assess_content_safety(doc, scan_titles=config.scan_titles) for doc in source_docs
        ]
    else:
        assessed = source_docs

    excluded = (
        [doc for doc in assessed if _risk_level(doc) == "high"]
        if config.enabled and config.mode == "exclude_high_risk"
        else []
    )
    excluded_ids = {doc.id for doc in excluded}
    retained = [doc for doc in assessed if doc.id not in excluded_ids]
    levels = {
        level: sum(1 for doc in assessed if _risk_level(doc) == level)
        for level in ("safe", "suspicious", "high")
    }
    categories = sorted(
        {
            str(category)
            for doc in assessed
            for category in _assessment(doc).get("signal_categories", [])
        }
    )
    summary: dict[str, object] = {
        "enabled": config.enabled,
        "mode": config.mode,
        "scan_titles": config.scan_titles,
        "risk_level_counts": levels,
        "signal_categories": categories,
        "retained_source_ids": [doc.id for doc in retained],
        "excluded_source_ids": [doc.id for doc in excluded],
    }
    return retained, summary


def _assessment(doc: SourceDoc) -> dict[str, object]:
    value = doc.metadata.get("content_safety", {})
    return value if isinstance(value, dict) else {}


def _risk_level(doc: SourceDoc) -> str:
    return str(_assessment(doc).get("risk_level", "safe"))

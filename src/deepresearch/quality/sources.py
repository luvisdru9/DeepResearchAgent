"""Deterministic, explainable source-credibility heuristics."""

from __future__ import annotations

from urllib.parse import urlsplit

from deepresearch.schema import SourceCredibility, SourceDoc


_LOW_AUTHORITY_HOSTS = {
    "blogspot.com",
    "csdn.net",
    "medium.com",
    "substack.com",
    "zhihu.com",
}

# 百科和论文索引适合发现线索，但通常不是所述事实或论文内容的第一手来源。
# 单独保留该信号，避免 HTTPS、日期和长正文叠加后把聚合页误判为高权威来源。
_SECONDARY_REFERENCE_HOSTS = {
    "academia.edu",
    "baike.baidu.com",
    "semanticscholar.org",
    "wikipedia.org",
}

# 这些站点托管论文原文或正式出版记录；该信号只表示“更接近第一手研究材料”，
# 不表示论文中的每个结论天然正确。
_PRIMARY_RESEARCH_HOSTS = {
    "aclanthology.org",
    "arxiv.org",
    "doi.org",
    "jmlr.org",
    "openreview.net",
    "proceedings.mlr.press",
    "proceedings.neurips.cc",
}


def assess_source_credibility(doc: SourceDoc) -> SourceDoc:
    """Attach a conservative source profile without claiming that its content is true."""

    score = 0.50
    signals: list[str] = []
    parsed = urlsplit(doc.url)
    host = (parsed.hostname or "").lower()

    if doc.source_type == "local":
        score += 0.10
        signals.append("local_curated_corpus")
    elif doc.source_type == "memory":
        score -= 0.05
        signals.append("derived_from_prior_evidence")
    elif doc.source_type == "web":
        signals.append("web_source")

    if parsed.scheme == "https":
        score += 0.02
        signals.append("https")
    if _is_authority_host(host):
        score += 0.20
        signals.append("institutional_domain")
    elif any(host == item or host.endswith(f".{item}") for item in _PRIMARY_RESEARCH_HOSTS):
        score += 0.25
        signals.append("primary_research_repository")
    elif any(host == item or host.endswith(f".{item}") for item in _LOW_AUTHORITY_HOSTS):
        score -= 0.20
        signals.append("user_generated_platform")
    elif any(
        host == item or host.endswith(f".{item}")
        for item in _SECONDARY_REFERENCE_HOSTS
    ):
        score -= 0.10
        signals.append("secondary_reference_platform")

    if doc.published_at:
        score += 0.05
        signals.append("publication_date_available")

    providers = doc.metadata.get("providers")
    if isinstance(providers, list) and len(set(map(str, providers))) > 1:
        score += 0.05
        signals.append("multiple_retrieval_providers")

    content_origin = doc.metadata.get("content_origin")
    if content_origin in {"tavily_raw_content", "document_loader"}:
        score += 0.03
        signals.append("full_content_available")
    elif content_origin == "tavily_snippet":
        score -= 0.10
        signals.append("snippet_only")

    if len(doc.content.strip()) >= 500:
        score += 0.02
        signals.append("substantial_content")
    elif len(doc.content.strip()) < 120:
        score -= 0.10
        signals.append("short_content")

    # 有实质内容的百科/论文索引仍可作为二手线索，不应与短内容农场同归为 low。
    if "secondary_reference_platform" in signals and "substantial_content" in signals:
        score = max(score, 0.50)

    score = round(min(0.95, max(0.10, score)), 2)
    tier = "high" if score >= 0.75 else "medium" if score >= 0.50 else "low"
    credibility = SourceCredibility(score=score, tier=tier, signals=signals)
    return doc.model_copy(update={"credibility": credibility})


def _is_authority_host(host: str) -> bool:
    """Recognize institutional suffixes without maintaining a vendor allow-list."""

    if not host:
        return False
    # academia.edu 等商业内容平台使用 .edu 结尾，但并非大学或研究机构官网。
    if any(
        host == item or host.endswith(f".{item}")
        for item in _SECONDARY_REFERENCE_HOSTS | _LOW_AUTHORITY_HOSTS
    ):
        return False
    labels = host.split(".")
    return host.endswith((".gov", ".edu")) or any(
        label in {"gov", "edu", "ac"} for label in labels[-3:-1]
    )

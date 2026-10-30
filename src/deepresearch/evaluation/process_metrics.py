"""Deterministic research-process and resource metrics derived from Replay 2.0."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

_PRIMARY_SOURCE_TYPES = {
    "primary",
    "research_paper",
    "official_document",
    "dataset",
}


def score_research_process(
    replay: dict[str, Any],
    *,
    required_source_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Compute auditable B2 trajectory metrics from a saved Replay payload."""

    docs = _unique_records(_flatten_mapping_lists(replay.get("docs_by_task")), key="id")
    evidence = _unique_records(replay.get("evidence", []), key="id")
    retrieved_source_ids = {str(item.get("id")) for item in docs if item.get("id")}
    required_ids = set(required_source_ids or [])
    required_source_recall = (
        len(required_ids.intersection(retrieved_source_ids)) / len(required_ids)
        if required_ids
        else None
    )
    domains = {
        _source_domain(str(item.get("url", "")))
        for item in docs
        if item.get("url")
    }
    primary_count = sum(
        str(item.get("source_type", "")).lower() in _PRIMARY_SOURCE_TYPES
        or bool((item.get("metadata") or {}).get("is_primary"))
        for item in docs
    )
    retry_count = sum(
        max(0, len(trace.get("attempt_traces", [])) - 1)
        for trace in replay.get("node_traces", [])
        if isinstance(trace, dict)
    )
    failed_stages = [
        str(trace.get("subtask_id"))
        for trace in replay.get("node_traces", [])
        if isinstance(trace, dict)
        and str(trace.get("status")) not in {"DONE", "VERIFIED"}
    ]
    for component in ("planner", "synthesizer", "redteam"):
        trace = replay.get(component)
        if isinstance(trace, dict) and trace.get("error"):
            failed_stages.append(component)

    state = replay.get("research_state") or {}
    budget = state.get("budget") or {}
    replay_metrics = replay.get("metrics") or {}
    document_count = len(docs)
    evidence_count = len(evidence)
    fallback_count, fallback_breakdown = _fallback_events(replay)
    empty_result_rule_assist_count = _empty_result_rule_assists(replay)
    return {
        "protocol": "replay_process_v1",
        "required_source_recall": required_source_recall,
        "required_source_count": len(required_ids),
        "retrieved_required_source_count": len(required_ids.intersection(retrieved_source_ids)),
        "retrieved_source_ids": sorted(retrieved_source_ids),
        "independent_domain_count": len(domains),
        "independent_domains": sorted(domains),
        "primary_source_ratio": primary_count / document_count if document_count else 0.0,
        "primary_source_count": primary_count,
        "document_count": document_count,
        "evidence_count": evidence_count,
        "evidence_yield": evidence_count / document_count if document_count else 0.0,
        "fallback_count": fallback_count,
        "fallback_breakdown": fallback_breakdown,
        # 合法空 JSON 后的规则补充不是请求/解析失败，单列而不计入 fallback_count。
        "empty_result_rule_assist_count": empty_result_rule_assist_count,
        "retry_count": retry_count,
        "failed_stages": failed_stages,
        "stop_reason": state.get("stop_reason"),
        "budget_exhausted": bool(budget.get("exhausted", False)),
        "budget_exhausted_limit": budget.get("exhausted_limit"),
        "resources": {
            "latency_seconds": float(
                replay_metrics.get("latency_seconds", budget.get("elapsed_seconds", 0.0)) or 0.0
            ),
            "llm_calls": int(budget.get("llm_calls", 0) or 0),
            "llm_input_tokens": int(
                replay_metrics.get("llm_input_tokens", budget.get("llm_input_tokens", 0)) or 0
            ),
            "llm_output_tokens": int(
                replay_metrics.get("llm_output_tokens", budget.get("llm_output_tokens", 0)) or 0
            ),
            "cost_usd": float(
                replay_metrics.get("cost_usd", budget.get("cost_usd", 0.0)) or 0.0
            ),
        },
    }


def _flatten_mapping_lists(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, dict):
        return []
    return [
        item
        for records in value.values()
        if isinstance(records, list)
        for item in records
        if isinstance(item, dict)
    ]


def _unique_records(value: Any, *, key: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    unique: dict[str, dict[str, Any]] = {}
    for item in value:
        if isinstance(item, dict) and item.get(key) is not None:
            unique.setdefault(str(item[key]), item)
    return list(unique.values())


def _source_domain(url: str) -> str:
    parsed = urlparse(url)
    return parsed.netloc.lower() or parsed.scheme.lower() or url.lower()


def _fallback_events(replay: dict[str, Any]) -> tuple[int, dict[str, int]]:
    """Count component and per-document fallbacks without recursively double-counting."""

    breakdown = {"planner": 0, "synthesizer": 0, "redteam": 0, "reader_document": 0}
    for component in ("planner", "synthesizer", "redteam"):
        trace = replay.get(component)
        breakdown[component] = int(
            isinstance(trace, dict) and bool(trace.get("fallback_used"))
        )

    for result in (replay.get("node_results") or {}).values():
        if not isinstance(result, dict):
            continue
        reader = (result.get("diagnostics") or {}).get("reader")
        if not isinstance(reader, dict):
            continue
        doc_results = reader.get("doc_results")
        if isinstance(doc_results, list):
            breakdown["reader_document"] += sum(
                isinstance(item, dict) and bool(item.get("fallback_used"))
                for item in doc_results
            )
        elif reader.get("fallback_used"):
            breakdown["reader_document"] += 1
    return sum(breakdown.values()), breakdown


def _empty_result_rule_assists(replay: dict[str, Any]) -> int:
    """Count Reader assists after valid empty outputs without calling them LLM failures."""

    count = 0
    for result in (replay.get("node_results") or {}).values():
        if not isinstance(result, dict):
            continue
        reader = (result.get("diagnostics") or {}).get("reader")
        if isinstance(reader, dict) and reader.get("empty_result_rule_assist_used"):
            count += 1
    return count

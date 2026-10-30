"""Helpers for readable artifact filenames while preserving stable run/call ids."""

from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import Any


def format_compact_utc_timestamp(value: Any) -> str:
    """Convert a datetime-like value into a filename-safe UTC timestamp."""

    timestamp = _coerce_datetime(value)
    if timestamp is None:
        return "unknown_time"
    return timestamp.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def build_run_artifact_stem(run_id: str, created_at: Any) -> str:
    """Build a readable replay stem that still preserves the canonical run id."""

    return f"{format_compact_utc_timestamp(created_at)}_{run_id}"


def build_replay_filename(run_id: str, created_at: Any) -> str:
    """Build the on-disk replay filename for one run."""

    return f"{build_run_artifact_stem(run_id, created_at)}_replay.json"


def build_llm_trace_filename(
    *,
    started_at: Any,
    agent: str,
    operation: str,
    call_id: str,
) -> str:
    """Build a readable raw LLM trace filename for one logical call."""

    safe_agent = sanitize_filename_component(agent)
    safe_operation = sanitize_filename_component(operation)
    return f"{format_compact_utc_timestamp(started_at)}_{safe_agent}_{safe_operation}_{call_id}.json"


def build_retrieval_comparison_filename(created_at: Any) -> str:
    """Build a readable filename for one retrieval-provider comparison run."""

    return f"{format_compact_utc_timestamp(created_at)}_retrieval_comparison.json"


def build_compression_comparison_filename(created_at: Any) -> str:
    """Build a readable filename for one context-compression ablation run."""

    return f"{format_compact_utc_timestamp(created_at)}_compression_comparison.json"


def build_memory_seed_filename(created_at: Any) -> str:
    """Build the frozen SQLite-memory filename used by one comparison experiment."""

    return f"{format_compact_utc_timestamp(created_at)}_memory_seed.sqlite3"


def build_system_comparison_filename(created_at: Any, *, markdown: bool = False) -> str:
    """Build JSON or Markdown filenames for one full strategy-matrix experiment."""

    suffix = "md" if markdown else "json"
    return f"{format_compact_utc_timestamp(created_at)}_system_comparison.{suffix}"


def build_benchmark_evaluation_filename(created_at: Any) -> str:
    """Build the durable artifact name for one benchmark evaluation run."""

    return f"{format_compact_utc_timestamp(created_at)}_benchmark_evaluation.json"


def sanitize_filename_component(value: str) -> str:
    """Collapse arbitrary text into a stable filename-safe component."""

    normalized = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    return normalized or "unknown"


def _coerce_datetime(value: Any) -> datetime | None:
    """Parse the datetime shapes used in replay and trace payloads."""

    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        text = value.strip()
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        try:
            return datetime.fromisoformat(text)
        except ValueError:
            return None
    return None

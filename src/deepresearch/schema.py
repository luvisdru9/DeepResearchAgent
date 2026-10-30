"""
定义全项目的数据结构，是 MVP 的“公共协议层”。
包括 ResearchTask、SubTask、SourceDoc、Evidence、ResearchReport、EvalResult 等。
后续扩展 LLM、DAG、记忆、评测时都会围绕这些 schema 走。
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    """Return timezone-aware UTC time for replay-safe timestamps."""

    return datetime.now(timezone.utc)

# DAG节点状态
class NodeStatus(StrEnum):
    """Lifecycle states used by DAG nodes and future verification steps."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    DONE = "DONE"
    FAILED = "FAILED"
    RETRYING = "RETRYING"
    SKIPPED = "SKIPPED"
    VERIFIED = "VERIFIED"
    REJECTED = "REJECTED"
    TIMEOUT = "TIMEOUT"


class ResearchRunStatus(StrEnum):
    """Lifecycle of the whole research run, independent from individual DAG nodes."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    DONE = "DONE"
    FAILED = "FAILED"
    INTERRUPTED = "INTERRUPTED"


class ResearchRoundStatus(StrEnum):
    """Outcome of one research round; PARTIAL keeps recoverable failures visible."""

    RUNNING = "RUNNING"
    DONE = "DONE"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"


class ResearchStopReason(StrEnum):
    """Why the controller stopped, so completion is not confused with exhaustion."""

    SINGLE_PASS_COMPLETE = "single_pass_complete"
    GOAL_SATISFIED = "goal_satisfied"
    FOLLOW_UP_COMPLETE = "follow_up_complete"
    # DAG/补研可能执行完毕，但最终 Verifier 仍发现实质性问题；不能误记为目标已满足。
    QUALITY_UNRESOLVED = "quality_unresolved"
    MAX_ROUNDS_REACHED = "max_rounds_reached"
    BUDGET_EXHAUSTED = "budget_exhausted"
    NO_PROGRESS = "no_progress"
    INTERRUPTED = "interrupted"
    ERROR = "error"


class EvidenceGapStatus(StrEnum):
    """Lifecycle for an unresolved research question discovered during execution."""

    OPEN = "OPEN"
    RESOLVED = "RESOLVED"
    DEFERRED = "DEFERRED"


class ResearchConflictStatus(StrEnum):
    """Whether a detected contradiction still needs adjudication."""

    OPEN = "OPEN"
    RESOLVED = "RESOLVED"

#用户的一次研究请求
class ResearchTask(BaseModel):
    """Top-level user request and run metadata."""

    query: str
    language: str = "zh"
    domain: str | None = None
    time_range: str | None = None
    report_length: str = "medium"
    require_citations: bool = True
    max_budget_usd: float | None = None
    max_llm_calls: int | None = Field(default=None, ge=0)
    max_total_tokens: int | None = Field(default=None, ge=0)
    max_runtime_seconds: float | None = Field(default=None, gt=0)
    run_id: str = Field(default_factory=lambda: uuid4().hex[:12])
    created_at: datetime = Field(default_factory=utc_now)

#planner拆出来的子任务
class SubTask(BaseModel):
    """A planned research node that can depend on earlier nodes."""

    id: str
    question: str
    depends_on: list[str] = Field(default_factory=list)
    # research 节点检索新资料；synthesis 节点只加工依赖节点已经产生的研究结果。
    task_type: Literal["research", "synthesis"] = "research"
    # 可选的短搜索查询与面向 Reader 的完整研究问题解耦；旧 Replay 缺失该字段仍可读取。
    retrieval_query: str | None = None
    # 补研轮可临时扩大候选池，过滤旧来源后仍只保留全局 top_k。
    retrieval_top_k: int | None = Field(default=None, ge=1)
    status: NodeStatus = NodeStatus.PENDING


class PlannerTrace(BaseModel):
    """Explain which planning strategy produced the final DAG."""

    requested_provider: str
    used_provider: str
    fallback_used: bool = False
    reasoning_summary: str | None = None
    error: str | None = None


class SynthesizerTrace(BaseModel):
    """Explain which synthesis strategy produced the final report."""

    requested_provider: str
    used_provider: str
    fallback_used: bool = False
    error: str | None = None
    # 确定性内容门禁的逐 finding 结果进入 Replay；用户报告不展示内部诊断。
    substance_diagnostics: list[dict[str, Any]] = Field(default_factory=list)
    methodology_diagnostics: list[dict[str, Any]] = Field(default_factory=list)
    source_absence_diagnostics: list[dict[str, Any]] = Field(default_factory=list)


class RedTeamTrace(BaseModel):
    """Explain which RedTeam / verifier strategy reviewed the final draft."""

    requested_provider: str
    used_provider: str
    fallback_used: bool = False
    summary: str | None = None
    overall_status: str | None = None
    error: str | None = None


class NodeAttemptTrace(BaseModel):
    """Execution details for one concrete attempt of a DAG node."""

    attempt: int
    status: NodeStatus
    started_at: datetime
    finished_at: datetime
    latency_seconds: float
    error: str | None = None


# 每个 DAG 节点的执行记录，会写入 replay。
class NodeTrace(BaseModel):
    """Replay-friendly execution record for one DAG node."""

    subtask_id: str
    question: str
    depends_on: list[str] = Field(default_factory=list)
    status: NodeStatus
    attempts: int = 0
    # 未实际执行的 SKIPPED 节点没有 started_at，但仍记录 finished_at 以便定位状态确定时间。
    started_at: datetime | None = None
    finished_at: datetime = Field(default_factory=utc_now)
    latency_seconds: float = 0.0
    error: str | None = None
    # attempts 保留为总次数，兼容已有调用方；attempt_traces 提供每次重试的完整细节。
    attempt_traces: list[NodeAttemptTrace] = Field(default_factory=list)


class SourceCredibility(BaseModel):
    """Explainable heuristic signals about a source; not a factual truth judgment."""

    score: float = Field(ge=0.0, le=1.0)
    tier: Literal["low", "medium", "high"]
    method: Literal["rule"] = "rule"
    signals: list[str] = Field(default_factory=list)


# 检索到的文档
class SourceDoc(BaseModel):
    """A retrieved document with enough metadata to cite later."""

    id: str
    title: str
    url: str
    content: str
    source_type: str = "local"
    # Web 检索可补充发布时间、检索分数和 provider 元数据；本地语料保持向后兼容。
    published_at: str | None = None
    retrieved_at: datetime = Field(default_factory=utc_now)
    score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    credibility: SourceCredibility | None = None


class SourceChunk(BaseModel):
    """A stable span of one source document with exact character provenance."""

    id: str
    source_id: str
    source_title: str
    source_url: str
    text: str
    chunk_index: int = Field(ge=0)
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

# reader从文档中抽出的Evidence
class Evidence(BaseModel):
    """A traceable evidence item extracted from a source document."""

    id: str
    claim: str
    quote: str
    source_id: str
    source_title: str
    source_url: str
    confidence: float = 0.7
    supports_subtask: str | None = None
    # 标记证据的抽取来源，后续评测 citation precision / hallucination 时可区分质量层级。
    extraction_method: Literal[
        "rule",
        "llm",
        "rule_fallback",
        "rule_gap_recovery",
        "rule_numeric_recovery",
    ] = "rule"
    # Chunk 与字符偏移将最终引用定位到原始 SourceDoc，而不是只停留在网页级 URL。
    chunk_id: str | None = None
    quote_start_char: int | None = Field(default=None, ge=0)
    quote_end_char: int | None = Field(default=None, ge=0)
    # 来源可信度是可解释启发式，与 Reader 对单条证据的 confidence 分开。
    source_credibility_score: float | None = Field(default=None, ge=0.0, le=1.0)
    source_credibility_tier: Literal["low", "medium", "high"] | None = None


class EvidenceCoverageSubject(BaseModel):
    """Evidence coverage for one named research object across fixed comparison dimensions."""

    subject: str
    evidence_ids: list[str] = Field(default_factory=list)
    dimension_evidence_ids: dict[str, list[str]] = Field(default_factory=dict)
    missing_dimensions: list[str] = Field(default_factory=list)


class EvidenceCoverageMatrix(BaseModel):
    """Deterministic subject-by-dimension map supplied to synthesis and verification."""

    dimensions: list[str] = Field(
        default_factory=lambda: ["goal", "data", "metrics", "scope"]
    )
    subjects: list[EvidenceCoverageSubject] = Field(default_factory=list)
    complete: bool = False


class EvidenceGap(BaseModel):
    """A concrete missing-evidence question that a later round may investigate."""

    id: str = Field(default_factory=lambda: f"gap_{uuid4().hex[:12]}")
    question: str
    # 对象级补研可使用更聚焦的搜索词，question 继续保留完整研究意图。
    retrieval_query: str | None = None
    # 对象化补研使用显式解析要求，避免任意新 Evidence 误关闭缺口。
    target_entities: list[str] = Field(default_factory=list)
    resolution_terms: list[str] = Field(default_factory=list)
    reason: str
    source: Literal["planner", "retriever", "reader", "redteam", "verifier", "system"]
    priority: Literal["low", "medium", "high"] = "medium"
    status: EvidenceGapStatus = EvidenceGapStatus.OPEN
    related_subtask_ids: list[str] = Field(default_factory=list)
    related_evidence_ids: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)
    resolved_at: datetime | None = None


class ResearchConflict(BaseModel):
    """A verifier-confirmed contradiction, kept separate from generic review advice."""

    id: str = Field(default_factory=lambda: f"conflict_{uuid4().hex[:12]}")
    claim: str
    description: str
    source: Literal["redteam", "verifier", "system"]
    status: ResearchConflictStatus = ResearchConflictStatus.OPEN
    evidence_ids: list[str] = Field(default_factory=list)
    resolution_reason: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    resolved_at: datetime | None = None


class RejectedEvidence(BaseModel):
    """A candidate discarded before entering the trusted Evidence collection."""

    id: str = Field(default_factory=lambda: f"rejected_{uuid4().hex[:12]}")
    stage: Literal["reader", "citation_validation", "verifier"]
    reason: str
    subtask_id: str | None = None
    source_id: str | None = None
    candidate_claim: str | None = None
    candidate_quote: str | None = None
    created_at: datetime = Field(default_factory=utc_now)


class ResearchBudgetUsage(BaseModel):
    """Observed resource usage and configured ceilings for future stop decisions."""

    max_budget_usd: float | None = None
    max_llm_calls: int | None = Field(default=None, ge=0)
    max_total_tokens: int | None = Field(default=None, ge=0)
    max_runtime_seconds: float | None = Field(default=None, gt=0)
    max_rounds: int = Field(default=1, ge=1)
    elapsed_seconds: float = Field(default=0.0, ge=0.0)
    llm_calls: int = Field(default=0, ge=0)
    llm_input_tokens: int = Field(default=0, ge=0)
    llm_output_tokens: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0.0)
    documents_retrieved: int = Field(default=0, ge=0)
    evidence_count: int = Field(default=0, ge=0)
    exhausted: bool = False
    exhausted_limit: str | None = None


class ResearchRound(BaseModel):
    """One observable plan/retrieve/read pass in a future multi-round controller."""

    round_number: int = Field(ge=1)
    status: ResearchRoundStatus = ResearchRoundStatus.RUNNING
    planned_subtask_ids: list[str] = Field(default_factory=list)
    completed_subtask_ids: list[str] = Field(default_factory=list)
    failed_subtask_ids: list[str] = Field(default_factory=list)
    visited_source_ids: list[str] = Field(default_factory=list)
    visited_source_urls: list[str] = Field(default_factory=list)
    new_evidence_ids: list[str] = Field(default_factory=list)
    opened_gap_ids: list[str] = Field(default_factory=list)
    resolved_gap_ids: list[str] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=utc_now)
    finished_at: datetime | None = None


class ResearchState(BaseModel):
    """Unified, replay-safe state shared by orchestration, persistence, and resume."""

    schema_version: str = "1.0"
    run_id: str
    query: str
    status: ResearchRunStatus = ResearchRunStatus.PENDING
    current_round: int = Field(default=0, ge=0)
    rounds: list[ResearchRound] = Field(default_factory=list)
    pending_questions: list[str] = Field(default_factory=list)
    evidence_gaps: list[EvidenceGap] = Field(default_factory=list)
    confirmed_conclusions: list[str] = Field(default_factory=list)
    conflicts: list[ResearchConflict] = Field(default_factory=list)
    rejected_evidence: list[RejectedEvidence] = Field(default_factory=list)
    # ID 索引方便轻量恢复消费者使用，完整结构仍保存在上面两个字段。
    conflict_ids: list[str] = Field(default_factory=list)
    rejected_evidence_ids: list[str] = Field(default_factory=list)
    visited_source_ids: list[str] = Field(default_factory=list)
    visited_source_urls: list[str] = Field(default_factory=list)
    source_credibility: dict[str, SourceCredibility] = Field(default_factory=dict)
    evidence_ids: list[str] = Field(default_factory=list)
    budget: ResearchBudgetUsage = Field(default_factory=ResearchBudgetUsage)
    stop_reason: ResearchStopReason | None = None
    last_error: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class ResearchNodeResult(BaseModel):
    """Structured output passed from one DAG node to its downstream nodes."""

    subtask_id: str
    task_type: Literal["research", "synthesis"] = "research"
    dependency_ids: list[str] = Field(default_factory=list)
    # inherited_evidence 保留可供下游继续加工的上游内容，IDs 便于 replay 快速检查血缘。
    inherited_evidence: list[Evidence] = Field(default_factory=list)
    inherited_evidence_ids: list[str] = Field(default_factory=list)
    # docs/evidence 只保存当前节点新产生的内容，避免最终汇总时重复计数。
    docs: list[SourceDoc] = Field(default_factory=list)
    chunks: list[SourceChunk] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    findings: list[str] = Field(default_factory=list)
    # 诊断信息不参与核心数据流，只进入 replay，便于定位检索覆盖、Reader 抽取和 fallback 问题。
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class AgentMessage(BaseModel):
    """Generic message envelope reserved for future multi-agent traces."""

    agent: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)

# red agent找出的问题
class RedTeamFinding(BaseModel):
    """A quality issue raised by the RedTeam reviewer."""

    issue: str
    severity: str = "medium"
    suggested_fix: str
    # 以下字段供 LLM RedTeam 进行 verifier-style 证据核查；保持可选以兼容规则 RedTeam。
    claim: str | None = None
    status: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    reason: str | None = None
    # citation_integrity 仅表示编号/表结构问题；语义支持度必须使用 citation_support。
    category: Literal[
        "factuality",
        "citation_integrity",
        "citation_support",
        "logic",
        "coverage",
        "time_scope",
        "other",
    ] = "other"
    # Verifier 显式区分“需要新证据”和“仅删除/收窄现有表述”，避免文本修订掩盖证据缺口。
    repair_action: Literal["retrieve", "narrow", "remove", "qualify", "none"] = (
        "retrieve"
    )

# 最终报告对象
class ResearchReport(BaseModel):
    """Structured report plus its Markdown rendering."""

    run_id: str
    query: str
    title: str
    summary: str
    key_findings: list[str]
    evidence: list[Evidence]
    limitations: list[str] = Field(default_factory=list)
    redteam_findings: list[RedTeamFinding] = Field(default_factory=list)
    markdown: str
    created_at: datetime = Field(default_factory=utc_now)


class ResearchEvidenceSnapshot(BaseModel):
    """Reusable output of one completed planning/retrieval/reading pass."""

    snapshot_id: str = Field(default_factory=lambda: f"snapshot_{uuid4().hex[:12]}")
    source_run_id: str
    task: ResearchTask
    evidence: list[Evidence]
    source_replay_path: str
    created_at: datetime = Field(default_factory=utc_now)


class SnapshotSynthesisResult(BaseModel):
    """Controlled synthesis result produced from a fixed Evidence snapshot."""

    report: ResearchReport
    context_evidence: list[Evidence]
    dropped_evidence_ids: list[str] = Field(default_factory=list)
    context_input_chars: int = 0
    context_output_chars: int = 0
    truncated: bool = False
    citation_validation: dict[str, Any] = Field(default_factory=dict)
    redteam_findings: list[RedTeamFinding] = Field(default_factory=list)
    review_status: str = "skipped"
    llm_calls: list[Any] = Field(default_factory=list)
    latency_seconds: float = 0.0

# 评测结果
class EvalResult(BaseModel):
    """Metrics for one benchmark sample."""

    sample_id: str
    run_id: str
    fact_accuracy: float
    # Benchmark v2 使用带权 gold claim recall；legacy 数据仍保留 fact_accuracy。
    weighted_claim_recall: float | None = None
    citation_coverage: float
    citation_precision: float
    # B2 显式拆分引用存在、正确和完整性；旧字段继续保留供历史比较代码读取。
    citation_existence: float | None = None
    citation_correctness: float | None = None
    citation_completeness: float | None = None
    hallucination_rate: float
    answer_completeness: float
    latency_seconds: float
    cost_usd: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)

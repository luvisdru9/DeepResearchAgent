"""
配置系统。现在支持 LLM 配置、检索配置、运行配置、评测数据路径。
默认 YAML 可启用真实 OpenAI-compatible LLM profile；检索仍以本地语料 data/local_corpus 为主，运行输出默认写到 runs/。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """One provider-neutral model profile plus optional named profiles."""

    provider: str = "mock"
    model: str = "mock-researcher"
    base_url: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    timeout_seconds: float = 60.0
    max_retries: int = 2
    # 网络级瞬时失败使用有界指数退避；结构化解析重试不等待。
    retry_base_delay_seconds: float = Field(default=0.5, ge=0.0)
    retry_max_delay_seconds: float = Field(default=8.0, ge=0.0)
    retry_jitter_ratio: float = Field(default=0.2, ge=0.0, le=1.0)
    temperature: float | None = 0.0
    request_options: dict[str, Any] = Field(default_factory=dict)
    active_profile: str | None = None
    profiles: dict[str, "LLMConfig"] = Field(default_factory=dict)

    def resolve_profile(self, profile_name: str | None = None) -> "LLMConfig":
        """Resolve a named profile while preserving the original flat config format."""

        selected = profile_name or self.active_profile
        if selected is None:
            return self
        if selected not in self.profiles:
            available = ", ".join(sorted(self.profiles)) or "(none)"
            raise ValueError(f"Unknown LLM profile '{selected}'. Available profiles: {available}.")
        return self.profiles[selected]


class PlanningConfig(BaseModel):
    """Select rule or LLM planning without coupling Pipeline to one model vendor."""

    provider: str = "rule"
    # 未指定时回退到 llm.active_profile，兼容已有单模型配置。
    llm_profile: str | None = None
    fallback_to_rule: bool = True


class SynthesisConfig(BaseModel):
    """Select rule or LLM report synthesis with a deterministic fallback."""

    provider: str = "rule"
    # 每个 Agent 可选择独立 Profile，也可以显式复用同一个 Profile。
    llm_profile: str | None = None
    fallback_to_rule: bool = True


class RedTeamConfig(BaseModel):
    """Configure the shared RedTeam/Verifier review and one-pass Blue repair cycle."""

    provider: str = "rule"
    llm_profile: str | None = None
    fallback_to_rule: bool = True
    # 第一版只允许一次 Blue 修订，避免质量循环无限调用模型。
    enable_blue_revision: bool = True
    # 只有确实执行过修订时才进行第二次核验。
    verify_after_revision: bool = True
    # Verifier 若只剩措辞/逻辑边界问题，允许再进行一次不检索的文本修订。
    enable_final_text_revision: bool = True
    # 最终文本修订不会自动算作通过；必须再经 Verifier 核验。
    verify_after_final_text_revision: bool = True


class ReaderConfig(BaseModel):
    """Select rule or LLM evidence extraction while preserving local validation."""

    provider: str = "rule"
    llm_profile: str | None = None
    fallback_to_rule: bool = True
    max_docs_per_call: int = 3
    max_evidence_per_doc: int = 3
    max_chars_per_doc: int = 4000
    # 达到目标证据数后 Reader 可提前停止，避免每个子任务固定读满 top-k 文档。
    target_evidence_per_subtask: int = 3
    # 达到目标来源覆盖后也可以提前停止；0 表示不启用该条件。
    target_unique_sources_per_subtask: int = 0
    # 已经有证据后，如果连续若干文档无证据，可停止继续消耗 LLM 调用；0 表示不启用。
    max_empty_docs_before_stop: int = 0
    # LLM 正常返回但没有抽到证据时，是否用规则 Reader 对少量文档补救。
    enable_gap_recovery: bool = True
    gap_recovery_max_docs: int = 2


class RetrievalConfig(BaseModel):
    """Provider-neutral retrieval settings plus Tavily-specific options."""

    provider: str = "local"
    corpus_dir: Path = Path("data/local_corpus")
    # frozen provider 只读取一个已发布的快照 manifest；不访问网络或当前本地语料。
    frozen_manifest_path: Path | None = None
    top_k: int = 5
    # 在线搜索凭据只记录环境变量名，不允许把真实密钥写入 YAML 或 replay。
    api_key_env: str = "TAVILY_API_KEY"
    endpoint: str = "https://api.tavily.com/search"
    timeout_seconds: float = 30.0
    search_depth: str = "basic"
    topic: str = "general"
    include_raw_content: bool = True
    include_published_date: bool = True
    include_domains: list[str] = Field(default_factory=list)
    exclude_domains: list[str] = Field(default_factory=list)
    language: str | None = None
    diversify_domains: bool = True
    # 默认使用搜索 API 返回的清洗正文；开启后才直接访问结果 URL。
    fetch_full_content: bool = False
    max_chars_per_doc: int = 100_000
    # hybrid 会并发组合多个 provider；候选扩展后统一去重和轻量重排。
    hybrid_providers: list[str] = Field(default_factory=lambda: ["local", "tavily"])
    hybrid_candidate_multiplier: int = Field(default=2, ge=1, le=5)
    rerank_enabled: bool = True
    rerank_provider_weight: float = Field(default=0.55, ge=0.0)
    rerank_lexical_weight: float = Field(default=0.35, ge=0.0)
    rerank_freshness_weight: float = Field(default=0.10, ge=0.0)
    # 轻量偏好第一手/机构来源；仅作为排序分量，不能替代 Reader 和 Verifier 的事实核查。
    rerank_credibility_weight: float = Field(default=0.15, ge=0.0)
    rerank_primary_source_bonus: float = Field(default=0.10, ge=0.0, le=0.5)
    # 对 ALCE/RAGTruth 这类可识别专名，若标题和摘要开头都未命中，对候选降权。
    rerank_entity_anchor_penalty: float = Field(default=0.25, ge=0.0, le=1.0)


class ContentSafetyConfig(BaseModel):
    """Deterministic diagnostics for instructions embedded in retrieved content."""

    enabled: bool = True
    # audit 只记录；exclude_high_risk 会在 Chunker/Reader 前排除多类别命中的来源。
    mode: Literal["audit", "exclude_high_risk"] = "audit"
    scan_titles: bool = True


class ToolCallingConfig(BaseModel):
    """Default-off bounds for the isolated provider-native tool-calling pilot."""

    enabled: bool = False
    llm_profile: str | None = None
    max_steps: int = Field(default=3, ge=1, le=5)
    max_calls_per_turn: int = Field(default=1, ge=1, le=3)
    max_result_chars: int = Field(default=8000, ge=512, le=50_000)


class ChunkingConfig(BaseModel):
    """Deterministic document chunking before Reader extraction."""

    enabled: bool = True
    max_chars: int = Field(default=2500, ge=256)
    overlap_chars: int = Field(default=200, ge=0)
    max_chunks_per_doc: int = Field(default=4, ge=1, le=100)


class RuntimeConfig(BaseModel):
    """Runtime switches shared by CLI, pipeline, and evaluation."""

    run_dir: Path = Path("runs")
    # 每个正常产出的报告保存为独立目录，包含 Markdown、结构化 JSON 和 manifest。
    report_subdir: str = "reports"
    save_llm_io: bool = False
    llm_trace_subdir: str = "llm_traces"
    language: str = "zh"
    report_length: str = "medium"
    enable_redteam: bool = True
    enable_compression: bool = True
    max_subtasks: int = 4
    max_concurrency: int = 4
    max_retries: int = 1
    node_timeout_seconds: float | None = None
    # 本地 API 关闭时最多等待后台 Pipeline 到达完成状态；超时任务标记为可恢复中断。
    api_shutdown_grace_seconds: float = Field(default=10.0, ge=0.0, le=300.0)


class ResearchLoopConfig(BaseModel):
    """Limits for the future gap-driven multi-round research controller."""

    # 第一阶段只记录首轮状态；显式关闭可保证现有 Pipeline 行为与调用成本不变。
    enabled: bool = False
    max_rounds: int = Field(default=1, ge=1, le=10)
    max_follow_up_subtasks: int = Field(default=3, ge=1, le=20)
    min_new_evidence_per_round: int = Field(default=1, ge=0)
    stop_on_no_progress: bool = True
    # 仅事实支持、覆盖和时效性问题允许触发补研；逻辑/格式问题仍由 Blue 修订处理。
    allow_redteam_follow_up: bool = True
    # 首次 Verifier 仍有证据型阻断项时，最多再触发一轮补研。
    allow_verifier_follow_up: bool = True
    # 有总调用上限时，为补研后的重新合成与最终复核保留逻辑调用名额。
    reserve_llm_calls_for_finalization: int = Field(default=2, ge=0, le=10)
    prefer_new_sources: bool = True
    follow_up_candidate_multiplier: int = Field(default=3, ge=1, le=10)
    # 搜索没有返回新 URL 时是否允许复用旧来源，避免严格去重导致补研必然空结果。
    allow_repeated_sources_when_exhausted: bool = True


class MemoryConfig(BaseModel):
    """SQLite state/evidence persistence and deterministic context compression settings."""

    enabled: bool = False
    # null 时跟随 runtime.run_dir，避免测试或自定义输出目录仍写入项目 runs/。
    sqlite_path: Path | None = None
    persist_evidence: bool = True
    compression_max_evidence: int = Field(default=12, ge=1)
    compression_max_chars: int = Field(default=12_000, ge=256)
    # 历史 Evidence 通过本地 numpy 索引加入 hybrid；可在公平检索实验中关闭。
    retrieval_enabled: bool = False
    retrieval_top_k: int = Field(default=5, ge=1)
    vector_dimensions: int = Field(default=512, ge=64)
    min_similarity: float = Field(default=0.05, ge=0.0, le=1.0)
    embedding_provider: str = "hashing"
    embedding_model: str = "text-embedding-3-small"
    embedding_base_url: str | None = None
    embedding_api_key_env: str = "OPENAI_API_KEY"
    embedding_timeout_seconds: float = Field(default=30.0, gt=0)
    embedding_send_dimensions: bool = True
    embedding_cache_enabled: bool = True
    scope_by_domain: bool = True
    scope_max_age_days: int | None = Field(default=3650, ge=1)
    scope_min_query_overlap: float = Field(default=0.05, ge=0.0, le=1.0)


class EvaluationConfig(BaseModel):
    """Benchmark dataset location."""

    benchmark_path: Path = Path("data/researchbench.json")
    comparison_subdir: str = "comparisons"
    # 默认关闭历史记忆，防止后运行的 provider 从前一 variant 的结果中获益。
    retrieval_comparison_use_memory: bool = False
    provider: str = "rule"
    llm_profile: str | None = None
    # 可以配置多个 Judge profile；为空时使用 llm_profile。每个 profile 可重复采样。
    judge_profiles: list[str] = Field(default_factory=list)
    judge_repetitions: int = Field(default=1, ge=1, le=10)
    # 多模型评测先在各 profile 内形成多数，再比较模型间共识；旧配置仍可使用扁平多数。
    judge_aggregation_policy: Literal[
        "majority_with_conservative_tie",
        "hierarchical_model_consensus",
    ] = "majority_with_conservative_tie"
    judge_min_successful_repetitions_per_profile: int = Field(default=2, ge=1, le=10)
    # B3 固定 Judge 协议；修改 prompt/标签集合时必须升级版本。
    judge_protocol_version: str = "4.0"
    # 固定种子的 claim 顺序扰动用于降低位置偏差，同时保持实验可复现。
    judge_shuffle_claims: bool = True
    judge_random_seed: int = 20261013
    fallback_to_rule: bool = True
    # 以下门槛只授权工程自动草稿，不等同于经过人工校准的正式 Benchmark。
    automatic_consensus_min_resolved_rate: float = Field(default=0.85, ge=0.0, le=1.0)
    automatic_consensus_min_cross_model_agreement: float = Field(
        default=0.80, ge=0.0, le=1.0
    )
    automatic_consensus_min_cohen_kappa: float = Field(
        default=0.60, ge=-1.0, le=1.0
    )


class AppConfig(BaseModel):
    """Root application config loaded from YAML with defaults for missing sections."""

    llm: LLMConfig = Field(default_factory=LLMConfig)
    planning: PlanningConfig = Field(default_factory=PlanningConfig)
    reader: ReaderConfig = Field(default_factory=ReaderConfig)
    synthesis: SynthesisConfig = Field(default_factory=SynthesisConfig)
    redteam: RedTeamConfig = Field(default_factory=RedTeamConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    content_safety: ContentSafetyConfig = Field(default_factory=ContentSafetyConfig)
    tool_calling: ToolCallingConfig = Field(default_factory=ToolCallingConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    research_loop: ResearchLoopConfig = Field(default_factory=ResearchLoopConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)


def load_config(path: str | Path = "configs/default.yaml") -> AppConfig:
    """Load YAML config, falling back to defaults when the file is absent."""

    load_env_file()
    config_path = Path(path)
    if not config_path.exists():
        return AppConfig()
    with config_path.open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle) or {}
    return AppConfig.model_validate(raw)


def load_env_file(path: str | Path = ".env") -> None:
    """Load local KEY=VALUE secrets without overriding exported environment variables."""

    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)

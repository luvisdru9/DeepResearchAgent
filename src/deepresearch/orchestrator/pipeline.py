"""
主流程编排器。

它串起 Planner、Retriever、Reader、RedTeam、Synthesizer，并把子任务交给 DAGExecutor 记录状态和耗时。
外部保持同步 run() 接口，内部 DAG 使用 asyncio 并发调度。
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from typing import Literal
from urllib.parse import urlsplit

# 初始化创建agent
from deepresearch.agents import (
    LLMRedTeamAgent,
    LLMReaderAgent,
    LLMSynthesizerAgent,
    LLMPlannerAgent,
    PlannerAgent,
    ReaderAgent,
    RedTeamAgent,
    RetrieverAgent,
    SynthesizerAgent,
)
from deepresearch.config import AppConfig
from deepresearch.entities import extract_named_research_entities
from deepresearch.llm import (
    LLMCallBudget,
    LLMCallTrace,
    LLMClient,
    LLMTraceStore,
    create_llm_client,
)
from deepresearch.memory import (
    CompressionResult,
    EvidenceCompressor,
    SQLiteMemoryStore,
    create_embedding_provider,
)
from deepresearch.orchestrator.dag import DAGExecutionResult, DAGExecutor
from deepresearch.orchestrator.artifacts import ReportArtifactStore
from deepresearch.orchestrator.replay import ReplayStore, build_dag_replay
from deepresearch.quality import (
    CitationValidation,
    FindingSubstanceDiagnostic,
    MethodologyObligationDiagnostic,
    SourceAbsenceDiagnostic,
    apply_content_safety,
    assess_methodology_obligations,
    assess_report_findings,
    assess_source_absence_claims,
    assess_source_credibility,
    report_finding_inputs,
    validate_report_citations,
)
from deepresearch.retrieval import (
    EvidenceMemorySearchProvider,
    HybridSearchProvider,
    SearchProvider,
    TextChunker,
    create_search_provider,
)
from deepresearch.retrieval.url_utils import canonicalize_url
from deepresearch.schema import (
    EvidenceGap,
    EvidenceGapStatus,
    Evidence,
    EvidenceCoverageMatrix,
    EvidenceCoverageSubject,
    NodeStatus,
    NodeTrace,
    PlannerTrace,
    RedTeamFinding,
    RedTeamTrace,
    RejectedEvidence,
    ResearchNodeResult,
    ResearchBudgetUsage,
    ResearchRound,
    ResearchRoundStatus,
    ResearchReport,
    ResearchConflict,
    ResearchConflictStatus,
    ResearchRunStatus,
    ResearchState,
    ResearchStopReason,
    ResearchTask,
    SnapshotSynthesisResult,
    SourceDoc,
    SubTask,
    SynthesizerTrace,
    utc_now,
)


class ResearchPipeline:
    """Coordinates the local or Web-backed deep research workflow."""

    def __init__(
        self,
        config: AppConfig,
        *,
        llm_client: LLMClient | None = None,
        llm_clients: dict[str, LLMClient] | None = None,
        search_provider: SearchProvider | None = None,
    ) -> None:
        self.config = config
        # llm_client 保留为全局测试覆盖；llm_clients 支持按 Agent 精确注入。
        self._default_llm_client = llm_client
        self._injected_llm_clients = dict(llm_clients or {})
        self._profile_llm_clients: dict[str, LLMClient] = {}
        self.memory_store: SQLiteMemoryStore | None = None
        if config.memory.enabled:
            memory_path = config.memory.sqlite_path or (config.runtime.run_dir / "deepresearch.sqlite3")
            self.memory_store = SQLiteMemoryStore(memory_path)
        # 测试可注入 provider；正常运行统一由工厂根据 retrieval.provider 创建。
        provider = search_provider or create_search_provider(config.retrieval)
        if (
            isinstance(provider, HybridSearchProvider)
            and self.memory_store is not None
            and config.memory.retrieval_enabled
            and "memory" not in provider.providers
        ):
            # 历史 Evidence 是运行期资源，不放进通用 factory，避免检索层依赖完整 AppConfig。
            provider.add_provider(
                "memory",
                EvidenceMemorySearchProvider(
                    self.memory_store,
                    dimensions=config.memory.vector_dimensions,
                    min_similarity=config.memory.min_similarity,
                    max_results=config.memory.retrieval_top_k,
                    embedding_provider=create_embedding_provider(
                        config.memory.embedding_provider,
                        dimensions=config.memory.vector_dimensions,
                        model=config.memory.embedding_model,
                        base_url=config.memory.embedding_base_url,
                        api_key_env=config.memory.embedding_api_key_env,
                        timeout_seconds=config.memory.embedding_timeout_seconds,
                        send_dimensions=config.memory.embedding_send_dimensions,
                    ),
                    cache_enabled=config.memory.embedding_cache_enabled,
                    scope_by_domain=config.memory.scope_by_domain,
                    scope_max_age_days=config.memory.scope_max_age_days,
                    scope_min_query_overlap=config.memory.scope_min_query_overlap,
                ),
            )
        rule_planner = PlannerAgent(max_subtasks=config.runtime.max_subtasks)
        if config.planning.provider == "llm":
            planner_llm = self._resolve_llm_client(
                "planner",
                config.planning.llm_profile,
            )
            self.planner = LLMPlannerAgent(
                planner_llm,
                max_subtasks=config.runtime.max_subtasks,
                fallback=rule_planner,
                fallback_to_rule=config.planning.fallback_to_rule,
            )
        elif config.planning.provider == "rule":
            self.planner = rule_planner
        else:
            raise ValueError(
                f"Unsupported planning provider '{config.planning.provider}'. "
                "Supported providers: rule, llm."
            )
        self.retriever = RetrieverAgent(provider, top_k=config.retrieval.top_k)
        self.chunker = TextChunker(
            max_chars=config.chunking.max_chars,
            overlap_chars=config.chunking.overlap_chars,
            max_chunks_per_doc=config.chunking.max_chunks_per_doc,
        )
        rule_reader = ReaderAgent()
        if config.reader.provider == "llm":
            reader_llm = self._resolve_llm_client(
                "reader",
                config.reader.llm_profile,
            )
            self.reader = LLMReaderAgent(
                reader_llm,
                fallback=rule_reader,
                fallback_to_rule=config.reader.fallback_to_rule,
                max_docs_per_call=config.reader.max_docs_per_call,
                max_evidence_per_doc=config.reader.max_evidence_per_doc,
                max_chars_per_doc=config.reader.max_chars_per_doc,
                target_evidence_per_subtask=config.reader.target_evidence_per_subtask,
                target_unique_sources_per_subtask=config.reader.target_unique_sources_per_subtask,
                max_empty_docs_before_stop=config.reader.max_empty_docs_before_stop,
                enable_gap_recovery=config.reader.enable_gap_recovery,
                gap_recovery_max_docs=config.reader.gap_recovery_max_docs,
            )
        elif config.reader.provider == "rule":
            self.reader = rule_reader
        else:
            raise ValueError(
                f"Unsupported reader provider '{config.reader.provider}'. "
                "Supported providers: rule, llm."
            )
        rule_redteam = RedTeamAgent()
        if config.redteam.provider == "llm":
            redteam_llm = self._resolve_llm_client(
                "redteam",
                config.redteam.llm_profile,
            )
            self.redteam = LLMRedTeamAgent(
                redteam_llm,
                fallback=rule_redteam,
                fallback_to_rule=config.redteam.fallback_to_rule,
            )
        elif config.redteam.provider == "rule":
            self.redteam = rule_redteam
        else:
            raise ValueError(
                f"Unsupported redteam provider '{config.redteam.provider}'. "
                "Supported providers: rule, llm."
            )
        rule_synthesizer = SynthesizerAgent()
        if config.synthesis.provider == "llm":
            synthesizer_llm = self._resolve_llm_client(
                "synthesizer",
                config.synthesis.llm_profile,
            )
            self.synthesizer = LLMSynthesizerAgent(
                synthesizer_llm,
                fallback=rule_synthesizer,
                fallback_to_rule=config.synthesis.fallback_to_rule,
            )
        elif config.synthesis.provider == "rule":
            self.synthesizer = rule_synthesizer
        else:
            raise ValueError(
                f"Unsupported synthesis provider '{config.synthesis.provider}'. "
                "Supported providers: rule, llm."
            )
        self.dag = DAGExecutor(
            max_retries=config.runtime.max_retries,
            timeout_seconds=config.runtime.node_timeout_seconds,
            max_concurrency=config.runtime.max_concurrency,
        )
        self.replay = ReplayStore(config.runtime.run_dir)
        self.report_artifacts = ReportArtifactStore(
            config.runtime.run_dir,
            subdir=config.runtime.report_subdir,
        )
        self.evidence_compressor = EvidenceCompressor(
            max_evidence=config.memory.compression_max_evidence,
            max_chars=config.memory.compression_max_chars,
        )
        self.llm_trace_store = LLMTraceStore(
            config.runtime.run_dir,
            subdir=config.runtime.llm_trace_subdir,
        )
        # 同一 Profile 只创建一个客户端；不同 Agent/Profile 的 trace 在运行结束时统一汇总。
        self.llm_clients = self._unique_llm_clients()
        for client in self.llm_clients:
            client.configure_raw_trace(
                enabled=config.runtime.save_llm_io,
                store=self.llm_trace_store,
            )
        # 兼容旧代码读取 pipeline.llm；新代码应使用具体 Agent 持有的客户端。
        self.llm = self.llm_clients[0] if self.llm_clients else None

    def run(self, task: ResearchTask) -> ResearchReport:
        """Synchronous compatibility wrapper for CLI and existing callers."""

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.run_async(task))
        raise RuntimeError("ResearchPipeline.run() cannot be used inside an event loop; await run_async().")

    def resume(self, run_id: str) -> ResearchReport:
        """Resume an interrupted SQLite-backed task or return its completed report."""

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.resume_async(run_id))
        raise RuntimeError(
            "ResearchPipeline.resume() cannot be used inside an event loop; await resume_async()."
        )

    async def resume_async(self, run_id: str) -> ResearchReport:
        """Load the original task and seed completed DAG nodes from durable state."""

        if self.memory_store is None:
            raise RuntimeError("Task resume requires memory.enabled=true.")
        run = self.memory_store.get_run(run_id)
        if run is None:
            raise KeyError(f"Unknown run_id '{run_id}'.")
        if run["status"] == "DONE" and run.get("replay_path"):
            payload = self.replay.read(run_id)
            return ResearchReport.model_validate(payload["report"])
        task = ResearchTask.model_validate(json.loads(run["task_json"]))
        return await self.run_async(task, resume=True)

    def synthesize_snapshot(
        self,
        task: ResearchTask,
        evidence: list[Evidence],
        *,
        compression_enabled: bool,
        max_evidence: int | None = None,
    ) -> SnapshotSynthesisResult:
        """Synchronous entrypoint for a controlled synthesis trial over fixed Evidence."""

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.synthesize_snapshot_async(
                    task,
                    evidence,
                    compression_enabled=compression_enabled,
                    max_evidence=max_evidence,
                )
            )
        raise RuntimeError(
            "ResearchPipeline.synthesize_snapshot() cannot run inside an event loop; "
            "await synthesize_snapshot_async()."
        )

    async def synthesize_snapshot_async(
        self,
        task: ResearchTask,
        evidence: list[Evidence],
        *,
        compression_enabled: bool,
        max_evidence: int | None = None,
    ) -> SnapshotSynthesisResult:
        """Run one Synthesizer + review pass without changing the fixed upstream snapshot.

        Blue revision is intentionally excluded here: paired compression experiments need an equal
        number of model stages, otherwise repair-loop branching hides the effect of context selection.
        """

        started = time.perf_counter()
        offsets: dict[int, int] = {}
        for client in self.llm_clients:
            client.bind_run(task.run_id, created_at=task.created_at)
            offsets[id(client)] = client.trace_collector.mark()
        try:
            if compression_enabled:
                compressor = EvidenceCompressor(
                    max_evidence=max_evidence or self.config.memory.compression_max_evidence,
                    max_chars=self.config.memory.compression_max_chars,
                )
                compression = compressor.compress(
                    evidence,
                    priority_evidence_ids=self._coverage_priority_evidence_ids(
                        task.query,
                        evidence,
                    ),
                )
            else:
                total_chars = sum(len(item.claim) + len(item.quote) for item in evidence)
                compression = CompressionResult(
                    evidence=evidence,
                    input_count=len(evidence),
                    output_count=len(evidence),
                    input_chars=total_chars,
                    output_chars=total_chars,
                )
            context_evidence = compression.evidence
            evidence_coverage = self._build_evidence_coverage_matrix(
                task.query,
                context_evidence,
            )
            if isinstance(self.synthesizer, LLMSynthesizerAgent):
                report = await self.synthesizer.synthesize_async(
                    task,
                    context_evidence,
                    [],
                    coverage_matrix=evidence_coverage,
                )
            else:
                report = self.synthesizer.synthesize(task, context_evidence, [])
            citation_validation = validate_report_citations(
                report,
                require_citations=task.require_citations,
            )
            redteam_findings: list[RedTeamFinding] = []
            review_status = "skipped"
            if self.config.runtime.enable_redteam:
                if isinstance(self.redteam, LLMRedTeamAgent):
                    raw_findings = await self.redteam.review_async(
                        report.key_findings,
                        context_evidence,
                        task=task,
                        report=report,
                        mode="attack",
                        coverage_matrix=evidence_coverage,
                    )
                    trace = self.redteam.last_trace
                else:
                    raw_findings = self.redteam.review(
                        report.key_findings,
                        context_evidence,
                        task=task,
                        report=report,
                    )
                    trace = RedTeamTrace(requested_provider="rule", used_provider="rule")
                filtered, suppressed = self._filter_deterministic_citation_findings(
                    raw_findings,
                    citation_validation,
                )
                filtered = self._with_report_requirement_findings(
                    filtered,
                    task=task,
                    report=report,
                )
                redteam_findings = self._actionable_redteam_findings(
                    filtered,
                    coverage_matrix=evidence_coverage,
                )
                review_status = self._effective_review_status(
                    trace,
                    redteam_findings,
                    reviewed_findings=filtered,
                    suppressed_findings=suppressed,
                )
                report = self._with_redteam_findings(report, redteam_findings)
            calls = self._collect_llm_calls(offsets)
            return SnapshotSynthesisResult(
                report=report,
                context_evidence=context_evidence,
                dropped_evidence_ids=compression.dropped_evidence_ids,
                context_input_chars=compression.input_chars,
                context_output_chars=compression.output_chars,
                truncated=compression.truncated,
                citation_validation=citation_validation.model_dump(mode="json"),
                redteam_findings=redteam_findings,
                review_status=review_status,
                llm_calls=[call.model_dump(mode="json") for call in calls],
                latency_seconds=time.perf_counter() - started,
            )
        finally:
            for client in self.llm_clients:
                client.unbind_run()

    async def run_async(self, task: ResearchTask, *, resume: bool = False) -> ResearchReport:
        """Run one research task with dependency-aware asynchronous node execution."""

        started = time.perf_counter()
        memory_started = False
        resume_subtasks: list[SubTask] = []
        resume_results: dict[str, ResearchNodeResult] = {}
        resume_traces: dict[str, NodeTrace] = {}
        previous_state: ResearchState | None = None
        if resume:
            if self.memory_store is None:
                raise RuntimeError("Task resume requires memory.enabled=true.")
            resume_subtasks, resume_results, resume_traces = self._load_resume_nodes(task.run_id)
            previous_state = self.memory_store.load_research_state(task.run_id)
        previous_budget = previous_state.budget if previous_state is not None else ResearchBudgetUsage()
        llm_trace_offsets: dict[int, int] = {}
        llm_call_budget = (
            LLMCallBudget(task.max_llm_calls, used_calls=previous_budget.llm_calls)
            if task.max_llm_calls is not None
            else None
        )
        # ResearchState 先于 Planner 建立，确保规划失败也能留下可恢复的运行状态。
        research_state = (
            previous_state.model_copy(
                update={
                    "status": ResearchRunStatus.RUNNING,
                    "stop_reason": None,
                    "last_error": None,
                    "updated_at": utc_now(),
                }
            )
            if previous_state is not None
            else ResearchState(
                run_id=task.run_id,
                query=task.query,
                status=ResearchRunStatus.RUNNING,
                budget=ResearchBudgetUsage(
                    max_budget_usd=task.max_budget_usd,
                    max_llm_calls=task.max_llm_calls,
                    max_total_tokens=task.max_total_tokens,
                    max_runtime_seconds=task.max_runtime_seconds,
                    max_rounds=self.config.research_loop.max_rounds,
                ),
                created_at=task.created_at,
                updated_at=task.created_at,
            )
        )

        def persist_llm_reservation(used_calls: int) -> None:
            """Persist quota before provider I/O so cancellation cannot reset call usage."""

            if self.memory_store is None or not memory_started:
                return
            reserved_state = research_state.model_copy(
                update={
                    "budget": research_state.budget.model_copy(
                        update={"llm_calls": used_calls}
                    ),
                    "updated_at": utc_now(),
                }
            )
            self.memory_store.save_research_state(reserved_state)

        if llm_call_budget is not None:
            llm_call_budget.set_reservation_callback(persist_llm_reservation)
        for client in self.llm_clients:
            client.bind_run(task.run_id, created_at=task.created_at)
            client.configure_call_budget(llm_call_budget)
            llm_trace_offsets[id(client)] = client.trace_collector.mark()
        try:
            if self.memory_store is not None:
                self.memory_store.start_run(task)
                self.memory_store.save_research_state(research_state)
                memory_started = True
            self._bind_memory_scope(task)
            initial_budget_limit = self._exhausted_budget_limit(
                task,
                llm_trace_offsets,
                started,
                base_usage=research_state.budget,
            )
            if resume_subtasks:
                subtasks = resume_subtasks
            elif isinstance(self.planner, LLMPlannerAgent) and initial_budget_limit is None:
                subtasks = await self.planner.plan_async(task)
            elif isinstance(self.planner, LLMPlannerAgent):
                subtasks = PlannerAgent(max_subtasks=self.config.runtime.max_subtasks).plan(task)
            else:
                # 规则 Planner 很轻量，但统一保留异步主流程，便于后续替换复杂 Planner。
                subtasks = self.planner.plan(task)
            if not resume_subtasks:
                # 比较题的维度式 Planner 输出容易让每个检索都同时包含所有对象，最终被
                # 搜索排序中的单一对象垄断。只补 retrieval_query，不改 Reader 问题和 DAG。
                subtasks = self._with_named_entity_retrieval_queries(task.query, subtasks)
            if resume_subtasks:
                planner_trace = PlannerTrace(
                    requested_provider="resume",
                    used_provider="sqlite",
                    reasoning_summary="Reused the persisted task graph and completed node results.",
                )
            elif isinstance(self.planner, LLMPlannerAgent) and initial_budget_limit is not None:
                planner_trace = PlannerTrace(
                    requested_provider="llm",
                    used_provider="rule",
                    fallback_used=True,
                    error=f"budget_exhausted:{initial_budget_limit}",
                )
            else:
                planner_trace = (
                    self.planner.last_trace
                    if isinstance(self.planner, LLMPlannerAgent)
                    else PlannerTrace(requested_provider="rule", used_provider="rule")
                )

            if self.memory_store is not None:
                # 规划完成后立即保存 PENDING 节点，进程中断时仍可重建原任务图。
                existing_ids = {item.id for item in resume_subtasks}
                for subtask in subtasks:
                    if subtask.id not in existing_ids:
                        self.memory_store.save_node(task.run_id, subtask, None, None)

            async def execute_subtask(
                subtask: SubTask,
                dependency_results: dict[str, ResearchNodeResult],
            ) -> ResearchNodeResult:
                """Execute a node using the concrete outputs of its dependencies."""

                inherited_evidence = self._collect_dependency_evidence(dependency_results)
                inherited_findings = self._collect_dependency_findings(dependency_results)

                if subtask.task_type == "synthesis":
                    # 综合节点不重复检索，只把上游 findings 和证据加工为可继续传递的结果。
                    findings = self._synthesize_node_findings(
                        subtask,
                        inherited_findings,
                        inherited_evidence,
                    )
                    return ResearchNodeResult(
                        subtask_id=subtask.id,
                        task_type=subtask.task_type,
                        dependency_ids=list(dependency_results),
                        inherited_evidence=inherited_evidence,
                        inherited_evidence_ids=[item.id for item in inherited_evidence],
                        findings=findings,
                    )

                # 把上游结论加入检索上下文，使后续研究围绕已有成果继续推进。
                retrieval_question = self._build_retrieval_question(subtask, inherited_findings)
                retrieval_subtask = subtask.model_copy(update={"question": retrieval_question})
                docs, retrieval_diagnostics = await self.retriever.retrieve_with_diagnostics_async(
                    retrieval_subtask
                )
                entity_filter_diagnostics = {
                    "follow_up_entity_filter_applied": False,
                    "follow_up_entity_filter_target": None,
                    "entity_mismatched_docs_removed": 0,
                }
                if subtask.id.startswith("round_"):
                    docs, entity_filter_diagnostics = self._filter_follow_up_docs_by_entity(
                        retrieval_question,
                        docs,
                    )
                source_reuse_diagnostics = {
                    "new_source_count": len(docs),
                    "repeated_source_count": 0,
                    "source_reuse_mode": "initial_round",
                }
                if subtask.id.startswith("round_") and self.config.research_loop.prefer_new_sources:
                    candidate_count_before_filter = len(docs)
                    async with visited_source_lock:
                        new_docs = [
                            doc
                            for doc in docs
                            if self._source_key(doc) not in visited_source_keys
                        ]
                        repeated_count = len(docs) - len(new_docs)
                        if new_docs:
                            docs = new_docs
                            reuse_mode = "new_sources_only"
                        elif self.config.research_loop.allow_repeated_sources_when_exhausted:
                            reuse_mode = "reused_after_new_sources_exhausted"
                        else:
                            docs = []
                            reuse_mode = "repeated_sources_rejected"
                        docs = docs[: self.config.retrieval.top_k]
                        visited_source_keys.update(self._source_key(doc) for doc in docs)
                    source_reuse_diagnostics = {
                        "new_source_count": (
                            len(docs) if reuse_mode == "new_sources_only" else 0
                        ),
                        "repeated_source_count": repeated_count,
                        "source_reuse_mode": reuse_mode,
                        "candidate_source_count_before_reuse_filter": candidate_count_before_filter,
                    }
                # 可信度是来源级启发式；Reader confidence 仍表示单条证据的抽取置信度。
                docs = [assess_source_credibility(doc) for doc in docs]
                # 外部内容在进入 Chunker/Reader 前接受启发式安全诊断。默认 audit
                # 仅记录，不改变来源集合；严格模式才排除多类别命中的来源。
                docs, content_safety_diagnostics = apply_content_safety(
                    docs,
                    self.config.content_safety,
                )
                credibility_tiers = {
                    tier: sum(
                        1
                        for doc in docs
                        if doc.credibility is not None and doc.credibility.tier == tier
                    )
                    for tier in ("high", "medium", "low")
                }
                chunks = (
                    self.chunker.chunk_documents_for_query(
                        docs,
                        f"{subtask.question}\n{retrieval_question}",
                    )
                    if self.config.chunking.enabled
                    else []
                )
                reader_docs = (
                    self.chunker.to_reader_docs(docs, chunks)
                    if self.config.chunking.enabled
                    else docs
                )
                node_budget_limit = self._exhausted_budget_limit(
                    task,
                    llm_trace_offsets,
                    started,
                    base_usage=research_state.budget,
                )
                if isinstance(self.reader, LLMReaderAgent) and node_budget_limit is None:
                    evidence, reader_diagnostics = await self.reader.extract_with_diagnostics_async(
                        subtask,
                        reader_docs,
                    )
                elif isinstance(self.reader, LLMReaderAgent):
                    evidence = await ReaderAgent().extract_async(
                        subtask,
                        reader_docs,
                        extraction_method="rule_fallback",
                    )
                    reader_diagnostics = {
                        "reader_provider": "rule",
                        "requested_provider": "llm",
                        "requested_docs_count": len(reader_docs),
                        "read_docs_count": len(reader_docs),
                        "fallback_used": True,
                        "fallback_reason": f"budget_exhausted:{node_budget_limit}",
                        "evidence_count": len(evidence),
                    }
                else:
                    evidence = await self.reader.extract_async(subtask, reader_docs)
                    reader_diagnostics = {
                        "reader_provider": "rule",
                        "requested_docs_count": len(reader_docs),
                        "read_docs_count": len(reader_docs),
                        "fallback_used": False,
                        "evidence_count": len(evidence),
                    }
                return ResearchNodeResult(
                    subtask_id=subtask.id,
                    task_type=subtask.task_type,
                    dependency_ids=list(dependency_results),
                    inherited_evidence=inherited_evidence,
                    inherited_evidence_ids=[item.id for item in inherited_evidence],
                    docs=docs,
                    chunks=chunks,
                    evidence=evidence,
                    findings=[item.claim for item in evidence[:5]],
                    diagnostics={
                        "retrieval": {
                            "provider": self.config.retrieval.provider,
                            "top_k": self.config.retrieval.top_k,
                            "retrieval_question": retrieval_question,
                            **retrieval_diagnostics,
                            "retrieved_docs_count": len(docs),
                            "retrieved_source_ids": [doc.id for doc in docs],
                            "chunking_enabled": self.config.chunking.enabled,
                            "chunk_ranking_strategy": (
                                "query_section_round_robin"
                                if self.config.chunking.enabled
                                else "disabled"
                            ),
                            "chunk_count": len(chunks),
                            "ranked_chunk_ids": [chunk.id for chunk in chunks],
                            "source_credibility_tiers": credibility_tiers,
                            "content_safety": content_safety_diagnostics,
                            **entity_filter_diagnostics,
                            **source_reuse_diagnostics,
                        },
                        "reader": reader_diagnostics,
                    },
                )

            def persist_node(
                subtask: SubTask,
                trace: NodeTrace,
                result: ResearchNodeResult | None,
            ) -> None:
                if self.memory_store is not None:
                    self.memory_store.save_node(task.run_id, subtask, trace, result)

            visited_source_keys: set[str] = set()
            visited_source_lock = asyncio.Lock()
            dag_result = await self.dag.run_async(
                subtasks,
                execute_subtask,
                seed_results=resume_results,
                seed_traces=resume_traces,
                on_node_complete=persist_node,
            )
            visited_source_keys.update(
                self._source_key(doc)
                for result in dag_result.results.values()
                for doc in result.docs
            )
            (
                subtasks,
                research_rounds,
                evidence_gaps,
                research_stop_reason,
            ) = await self._run_follow_up_rounds(
                subtasks,
                dag_result,
                execute_subtask,
                research_query=task.query,
                budget_check=lambda: self._exhausted_budget_limit(
                    task,
                    llm_trace_offsets,
                    started,
                    base_usage=research_state.budget,
                ),
                on_node_complete=persist_node,
                llm_call_budget=llm_call_budget,
            )
            docs_by_task = {}
            chunks_by_task = {}
            all_evidence: list[Evidence] = []

            for subtask_id, output in dag_result.results.items():
                docs_by_task[subtask_id] = output.docs
                chunks_by_task[subtask_id] = output.chunks
                all_evidence.extend(output.evidence)
            all_evidence = self._dedupe_evidence(all_evidence)
            compression_result = self._compress_evidence_context(
                all_evidence,
                research_query=task.query,
            )
            synthesis_evidence = compression_result.evidence
            evidence_coverage = self._build_evidence_coverage_matrix(
                task.query,
                synthesis_evidence,
            )

            if self.memory_store is not None:
                trace_by_id = {trace.subtask_id: trace for trace in dag_result.traces}
                for subtask in subtasks:
                    self.memory_store.save_node(
                        task.run_id,
                        subtask,
                        trace_by_id.get(subtask.id),
                        dag_result.results.get(subtask.id),
                    )
                if self.config.memory.persist_evidence:
                    self.memory_store.save_evidence(task.run_id, all_evidence)

            leaf_findings = self._collect_leaf_findings(subtasks, dag_result.results)
            draft_findings = leaf_findings or [item.claim for item in synthesis_evidence[:5]]
            budget_limit = self._exhausted_budget_limit(
                task,
                llm_trace_offsets,
                started,
                base_usage=research_state.budget,
            )
            # LLM Synthesizer 使用异步调用；规则实现继续作为稳定的本地路径和失败回退。
            if isinstance(self.synthesizer, LLMSynthesizerAgent) and budget_limit is None:
                report = await self.synthesizer.synthesize_async(
                    task,
                    synthesis_evidence,
                    [],
                    coverage_matrix=evidence_coverage,
                )
                synthesizer_trace = self.synthesizer.last_trace
            elif budget_limit is None:
                report = self.synthesizer.synthesize(
                    task,
                    synthesis_evidence,
                    [],
                )
                synthesizer_trace = SynthesizerTrace(
                    requested_provider="rule",
                    used_provider="rule",
                )
            else:
                # 预算触顶后仍用确定性 Synthesizer 交付已有证据，避免任务只留下半成品。
                report = SynthesizerAgent().synthesize(task, synthesis_evidence, [])
                synthesizer_trace = SynthesizerTrace(
                    requested_provider=self.config.synthesis.provider,
                    used_provider="rule",
                    fallback_used=True,
                    error=f"budget_exhausted:{budget_limit}",
                )
            # 质量闭环保留初稿，并把 Red 攻击、Blue 修订和最终核验分别写入 replay。
            draft_report = report
            draft_citation_validation = validate_report_citations(
                draft_report,
                require_citations=task.require_citations,
            )
            final_citation_validation = draft_citation_validation
            raw_draft_review_findings = []
            draft_review_findings = []
            suppressed_draft_findings = []
            actionable_draft_findings = []
            advisory_draft_findings: list[RedTeamFinding] = []
            raw_verification_findings = []
            suppressed_final_findings = []
            final_findings = []
            final_advisory_findings: list[RedTeamFinding] = []
            effective_draft_status = "skipped"
            effective_final_status = "skipped"
            research_follow_up_attempted = False
            research_follow_up_succeeded = False
            pre_research_report: ResearchReport | None = None
            pre_research_review = RedTeamTrace(
                requested_provider="skipped",
                used_provider="skipped",
            )
            pre_research_findings: list[RedTeamFinding] = []
            revised_report: ResearchReport | None = None
            revision_attempted = False
            revision_succeeded = False
            follow_up_revision_handled = False
            revision_trace = SynthesizerTrace(
                requested_provider="skipped",
                used_provider="skipped",
            )
            verification_trace = RedTeamTrace(
                requested_provider="skipped",
                used_provider="skipped",
            )
            final_text_revision_attempted = False
            final_text_revision_succeeded = False
            final_text_revised_report: ResearchReport | None = None
            final_text_revision_trace = SynthesizerTrace(
                requested_provider="skipped",
                used_provider="skipped",
            )
            final_text_verification_trace = RedTeamTrace(
                requested_provider="skipped",
                used_provider="skipped",
            )
            raw_final_text_verification_findings: list[RedTeamFinding] = []
            suppressed_final_text_findings: list[RedTeamFinding] = []
            # 终局修订把“可以只改文字解决”和“仍需新证据”的问题分开保存；
            # deferred 项不会因为另一项成功收窄而被静默抹去。
            terminal_repairable_findings: list[RedTeamFinding] = []
            terminal_deferred_findings: list[RedTeamFinding] = []
            terminal_action_conversions: list[dict[str, str]] = []
            verifier_follow_up_attempted = False
            verifier_follow_up_succeeded = False
            pre_verifier_follow_up_verification = RedTeamTrace(
                requested_provider="skipped",
                used_provider="skipped",
            )
            pre_verifier_follow_up_findings: list[RedTeamFinding] = []
            verifier_follow_up_revision_trace = SynthesizerTrace(
                requested_provider="skipped",
                used_provider="skipped",
            )
            verifier_follow_up_verification_trace = RedTeamTrace(
                requested_provider="skipped",
                used_provider="skipped",
            )
            budget_limit = budget_limit or self._exhausted_budget_limit(
                task,
                llm_trace_offsets,
                started,
                base_usage=research_state.budget,
            )
            if self.config.runtime.enable_redteam and budget_limit is None:
                if isinstance(self.redteam, LLMRedTeamAgent):
                    raw_draft_review_findings = await self.redteam.review_async(
                        report.key_findings or draft_findings,
                        synthesis_evidence,
                        task=task,
                        report=report,
                        mode="attack",
                        coverage_matrix=evidence_coverage,
                    )
                    redteam_trace = self.redteam.last_trace
                else:
                    raw_draft_review_findings = self.redteam.review(
                        report.key_findings or draft_findings,
                        synthesis_evidence,
                        task=task,
                        report=report,
                    )
                    redteam_trace = RedTeamTrace(
                        requested_provider="rule",
                        used_provider="rule",
                    )
                # 本地校验通过时，模型提出的纯编号/表结构问题属于可确定的误报；原始意见仍写入 replay。
                draft_review_findings, suppressed_draft_findings = (
                    self._filter_deterministic_citation_findings(
                        raw_draft_review_findings,
                        draft_citation_validation,
                    )
                )
                draft_review_findings = self._with_report_requirement_findings(
                    draft_review_findings,
                    task=task,
                    report=report,
                )
                # supported 项是核验记录，不应作为待修复问题进入 Blue prompt 或最终报告。
                actionable_draft_findings = self._actionable_redteam_findings(
                    draft_review_findings,
                    coverage_matrix=evidence_coverage,
                )
                advisory_draft_findings = self._advisory_redteam_findings(
                    draft_review_findings,
                    coverage_matrix=evidence_coverage,
                )
                final_findings = actionable_draft_findings
                final_advisory_findings = advisory_draft_findings
                effective_draft_status = self._effective_review_status(
                    redteam_trace,
                    actionable_draft_findings,
                    reviewed_findings=draft_review_findings,
                    suppressed_findings=suppressed_draft_findings,
                )
                effective_final_status = effective_draft_status

                remaining_round = len(research_rounds) + 1
                redteam_follow_up = None
                if (
                    self.config.research_loop.enabled
                    and self.config.research_loop.allow_redteam_follow_up
                    and remaining_round <= self.config.research_loop.max_rounds
                    and actionable_draft_findings
                    and self._exhausted_budget_limit(
                        task,
                        llm_trace_offsets,
                        started,
                        base_usage=research_state.budget,
                    )
                    is None
                ):
                    redteam_follow_up = await self._run_redteam_follow_up_round(
                        remaining_round,
                        task.query,
                        actionable_draft_findings,
                        dag_result,
                        execute_subtask,
                        on_node_complete=persist_node,
                        llm_call_budget=llm_call_budget,
                    )
                if redteam_follow_up is not None:
                    research_follow_up_attempted = True
                    pre_research_report = draft_report
                    pre_research_review = redteam_trace
                    pre_research_findings = list(actionable_draft_findings)
                    (
                        follow_up_subtasks,
                        follow_up_result,
                        follow_up_round,
                        redteam_gaps,
                        follow_up_stop_reason,
                    ) = redteam_follow_up
                    subtasks.extend(follow_up_subtasks)
                    research_rounds.append(follow_up_round)
                    evidence_gaps.extend(redteam_gaps)
                    self._merge_dag_results(dag_result, follow_up_result)
                    research_stop_reason = follow_up_stop_reason
                    research_follow_up_succeeded = bool(follow_up_round.new_evidence_ids)

                    if research_follow_up_succeeded:
                        docs_by_task.update(
                            {
                                subtask_id: result.docs
                                for subtask_id, result in follow_up_result.results.items()
                            }
                        )
                        chunks_by_task.update(
                            {
                                subtask_id: result.chunks
                                for subtask_id, result in follow_up_result.results.items()
                            }
                        )
                        all_evidence = self._dedupe_evidence(
                            [
                                item
                                for result in dag_result.results.values()
                                for item in result.evidence
                            ]
                        )
                        compression_result = self._compress_evidence_context(
                            all_evidence,
                            research_query=task.query,
                        )
                        synthesis_evidence = compression_result.evidence
                        evidence_coverage = self._build_evidence_coverage_matrix(
                            task.query,
                            synthesis_evidence,
                        )
                        if self.memory_store is not None:
                            follow_trace_by_id = {
                                trace.subtask_id: trace for trace in follow_up_result.traces
                            }
                            for follow_subtask in follow_up_subtasks:
                                self.memory_store.save_node(
                                    task.run_id,
                                    follow_subtask,
                                    follow_trace_by_id.get(follow_subtask.id),
                                    follow_up_result.results.get(follow_subtask.id),
                                )
                            if self.config.memory.persist_evidence:
                                self.memory_store.save_evidence(task.run_id, all_evidence)

                        can_directly_repair = (
                            self.config.redteam.enable_blue_revision
                            and isinstance(self.synthesizer, LLMSynthesizerAgent)
                            and isinstance(self.redteam, LLMRedTeamAgent)
                            and redteam_trace.used_provider == "llm"
                            and self._exhausted_budget_limit(
                                task,
                                llm_trace_offsets,
                                started,
                                base_usage=research_state.budget,
                            )
                            is None
                        )
                        if can_directly_repair:
                            # 补研本身就是对 Red 意见的回应；直接 Blue 修订再 Verifier，
                            # 避免重复“无定向重新合成 -> 再攻击”浪费有限调用名额。
                            revision_attempted = True
                            follow_up_revision_handled = True
                            report = await self.synthesizer.revise_async(
                                task,
                                draft_report,
                                synthesis_evidence,
                                pre_research_findings,
                                coverage_matrix=evidence_coverage,
                            )
                            revision_trace = self.synthesizer.last_trace
                            revision_succeeded = revision_trace.used_provider == "llm"
                            revised_report = report if revision_succeeded else None
                            final_citation_validation = validate_report_citations(
                                report,
                                require_citations=task.require_citations,
                            )
                            if revision_succeeded:
                                effective_final_status = "not_verified"
                            if (
                                self.config.redteam.verify_after_revision
                                and revision_succeeded
                                and self._exhausted_budget_limit(
                                    task,
                                    llm_trace_offsets,
                                    started,
                                    base_usage=research_state.budget,
                                )
                                is None
                            ):
                                raw_verification_findings = await self.redteam.review_async(
                                    report.key_findings,
                                    synthesis_evidence,
                                    task=task,
                                    report=report,
                                    mode="verify",
                                    prior_findings=pre_research_findings,
                                    coverage_matrix=evidence_coverage,
                                )
                                verification_trace = self.redteam.last_trace
                                verification_findings, suppressed_final_findings = (
                                    self._filter_deterministic_citation_findings(
                                        raw_verification_findings,
                                        final_citation_validation,
                                    )
                                )
                                verification_findings = (
                                    self._with_report_requirement_findings(
                                        verification_findings,
                                        task=task,
                                        report=report,
                                    )
                                )
                                final_findings = self._actionable_redteam_findings(
                                    verification_findings,
                                    coverage_matrix=evidence_coverage,
                                )
                                final_advisory_findings = self._advisory_redteam_findings(
                                    verification_findings,
                                    coverage_matrix=evidence_coverage,
                                )
                                effective_final_status = self._effective_review_status(
                                    verification_trace,
                                    final_findings,
                                    reviewed_findings=verification_findings,
                                    suppressed_findings=suppressed_final_findings,
                                )
                        else:
                            # 规则 Agent 或显式关闭 Blue 时保留原有的重新合成/审查路径。
                            if isinstance(self.synthesizer, LLMSynthesizerAgent):
                                report = await self.synthesizer.synthesize_async(
                                    task,
                                    synthesis_evidence,
                                    [],
                                    coverage_matrix=evidence_coverage,
                                )
                                synthesizer_trace = self.synthesizer.last_trace
                            else:
                                report = self.synthesizer.synthesize(task, synthesis_evidence, [])
                                synthesizer_trace = SynthesizerTrace(
                                    requested_provider="rule",
                                    used_provider="rule",
                                )
                            draft_report = report
                            draft_citation_validation = validate_report_citations(
                                draft_report,
                                require_citations=task.require_citations,
                            )
                            final_citation_validation = draft_citation_validation
                            draft_findings = [item.claim for item in synthesis_evidence[:5]]
                            if isinstance(self.redteam, LLMRedTeamAgent):
                                raw_draft_review_findings = await self.redteam.review_async(
                                    report.key_findings or draft_findings,
                                    synthesis_evidence,
                                    task=task,
                                    report=report,
                                    mode="attack",
                                    coverage_matrix=evidence_coverage,
                                )
                                redteam_trace = self.redteam.last_trace
                            else:
                                raw_draft_review_findings = self.redteam.review(
                                    report.key_findings or draft_findings,
                                    synthesis_evidence,
                                    task=task,
                                    report=report,
                                )
                                redteam_trace = RedTeamTrace(
                                    requested_provider="rule",
                                    used_provider="rule",
                                )
                            draft_review_findings, suppressed_draft_findings = (
                                self._filter_deterministic_citation_findings(
                                    raw_draft_review_findings,
                                    draft_citation_validation,
                                )
                            )
                            draft_review_findings = (
                                self._with_report_requirement_findings(
                                    draft_review_findings,
                                    task=task,
                                    report=report,
                                )
                            )
                            actionable_draft_findings = self._actionable_redteam_findings(
                                draft_review_findings,
                                coverage_matrix=evidence_coverage,
                            )
                            advisory_draft_findings = self._advisory_redteam_findings(
                                draft_review_findings,
                                coverage_matrix=evidence_coverage,
                            )
                            final_findings = actionable_draft_findings
                            final_advisory_findings = advisory_draft_findings
                            effective_draft_status = self._effective_review_status(
                                redteam_trace,
                                actionable_draft_findings,
                                reviewed_findings=draft_review_findings,
                                suppressed_findings=suppressed_draft_findings,
                            )
                            effective_final_status = effective_draft_status

                should_revise = (
                    not follow_up_revision_handled
                    and self.config.redteam.enable_blue_revision
                    and isinstance(self.redteam, LLMRedTeamAgent)
                    and redteam_trace.used_provider == "llm"
                    and effective_draft_status in {"needs_revision", "fail"}
                    and isinstance(self.synthesizer, LLMSynthesizerAgent)
                    and bool(actionable_draft_findings)
                    and self._exhausted_budget_limit(
                        task,
                        llm_trace_offsets,
                        started,
                        base_usage=research_state.budget,
                    )
                    is None
                )
                if should_revise:
                    revision_attempted = True
                    report = await self.synthesizer.revise_async(
                        task,
                        draft_report,
                        synthesis_evidence,
                        actionable_draft_findings,
                        coverage_matrix=evidence_coverage,
                    )
                    revision_trace = self.synthesizer.last_trace
                    revision_succeeded = revision_trace.used_provider == "llm"
                    revised_report = report if revision_succeeded else None
                    final_citation_validation = validate_report_citations(
                        report,
                        require_citations=task.require_citations,
                    )
                    if revision_succeeded:
                        effective_final_status = "not_verified"

                    if (
                        self.config.redteam.verify_after_revision
                        and revision_succeeded
                        and self._exhausted_budget_limit(
                            task,
                            llm_trace_offsets,
                            started,
                            base_usage=research_state.budget,
                        )
                        is None
                    ):
                        raw_verification_findings = await self.redteam.review_async(
                            report.key_findings,
                            synthesis_evidence,
                            task=task,
                            report=report,
                            mode="verify",
                            prior_findings=actionable_draft_findings,
                            coverage_matrix=evidence_coverage,
                        )
                        verification_trace = self.redteam.last_trace
                        verification_findings, suppressed_final_findings = (
                            self._filter_deterministic_citation_findings(
                                raw_verification_findings,
                                final_citation_validation,
                            )
                        )
                        verification_findings = self._with_report_requirement_findings(
                            verification_findings,
                            task=task,
                            report=report,
                        )
                        final_findings = self._actionable_redteam_findings(
                            verification_findings,
                            coverage_matrix=evidence_coverage,
                        )
                        final_advisory_findings = self._advisory_redteam_findings(
                            verification_findings,
                            coverage_matrix=evidence_coverage,
                        )
                        effective_final_status = self._effective_review_status(
                            verification_trace,
                            final_findings,
                            reviewed_findings=verification_findings,
                            suppressed_findings=suppressed_final_findings,
                        )
                verifier_round_number = len(research_rounds) + 1
                verifier_research_gaps = self._redteam_evidence_gaps(
                    final_findings,
                    task.query,
                    source="verifier",
                    evidence=synthesis_evidence,
                )
                should_run_verifier_follow_up = (
                    self.config.research_loop.enabled
                    and self.config.research_loop.allow_verifier_follow_up
                    and verifier_round_number <= self.config.research_loop.max_rounds
                    and bool(verifier_research_gaps)
                    and verification_trace.used_provider == "llm"
                    and isinstance(self.synthesizer, LLMSynthesizerAgent)
                    and isinstance(self.redteam, LLMRedTeamAgent)
                    and self._exhausted_budget_limit(
                        task,
                        llm_trace_offsets,
                        started,
                        base_usage=research_state.budget,
                    )
                    is None
                )
                if should_run_verifier_follow_up:
                    verifier_follow_up_attempted = True
                    pre_verifier_follow_up_verification = verification_trace
                    pre_verifier_follow_up_findings = list(final_findings)
                    verifier_follow_up = await self._run_redteam_follow_up_round(
                        verifier_round_number,
                        task.query,
                        final_findings,
                        dag_result,
                        execute_subtask,
                        on_node_complete=persist_node,
                        llm_call_budget=llm_call_budget,
                        review_source="verifier",
                    )
                    if verifier_follow_up is not None:
                        (
                            verifier_subtasks,
                            verifier_result,
                            verifier_round,
                            verifier_gaps,
                            verifier_stop_reason,
                        ) = verifier_follow_up
                        subtasks.extend(verifier_subtasks)
                        research_rounds.append(verifier_round)
                        evidence_gaps.extend(verifier_gaps)
                        self._merge_dag_results(dag_result, verifier_result)
                        research_stop_reason = verifier_stop_reason
                        verifier_follow_up_succeeded = bool(
                            verifier_round.resolved_gap_ids
                        )

                        if verifier_follow_up_succeeded:
                            docs_by_task.update(
                                {
                                    subtask_id: result.docs
                                    for subtask_id, result in verifier_result.results.items()
                                }
                            )
                            chunks_by_task.update(
                                {
                                    subtask_id: result.chunks
                                    for subtask_id, result in verifier_result.results.items()
                                }
                            )
                            all_evidence = self._dedupe_evidence(
                                [
                                    item
                                    for result in dag_result.results.values()
                                    for item in result.evidence
                                ]
                            )
                            compression_result = self._compress_evidence_context(
                                all_evidence,
                                research_query=task.query,
                            )
                            synthesis_evidence = compression_result.evidence
                            evidence_coverage = self._build_evidence_coverage_matrix(
                                task.query,
                                synthesis_evidence,
                            )
                            if self.memory_store is not None:
                                verifier_trace_by_id = {
                                    trace.subtask_id: trace
                                    for trace in verifier_result.traces
                                }
                                for verifier_subtask in verifier_subtasks:
                                    self.memory_store.save_node(
                                        task.run_id,
                                        verifier_subtask,
                                        verifier_trace_by_id.get(verifier_subtask.id),
                                        verifier_result.results.get(verifier_subtask.id),
                                    )
                                if self.config.memory.persist_evidence:
                                    self.memory_store.save_evidence(task.run_id, all_evidence)

                            report = await self.synthesizer.revise_async(
                                task,
                                report,
                                synthesis_evidence,
                                pre_verifier_follow_up_findings,
                                coverage_matrix=evidence_coverage,
                            )
                            verifier_follow_up_revision_trace = self.synthesizer.last_trace
                            verifier_revision_succeeded = (
                                verifier_follow_up_revision_trace.used_provider == "llm"
                            )
                            final_citation_validation = validate_report_citations(
                                report,
                                require_citations=task.require_citations,
                            )
                            if verifier_revision_succeeded:
                                effective_final_status = "not_verified"
                            if (
                                verifier_revision_succeeded
                                and self._exhausted_budget_limit(
                                    task,
                                    llm_trace_offsets,
                                    started,
                                    base_usage=research_state.budget,
                                )
                                is None
                            ):
                                raw_verification_findings = await self.redteam.review_async(
                                    report.key_findings,
                                    synthesis_evidence,
                                    task=task,
                                    report=report,
                                    mode="verify",
                                    prior_findings=pre_verifier_follow_up_findings,
                                    coverage_matrix=evidence_coverage,
                                )
                                verifier_follow_up_verification_trace = self.redteam.last_trace
                                verification_trace = verifier_follow_up_verification_trace
                                (
                                    verifier_findings,
                                    suppressed_after_verifier_follow_up,
                                ) = self._filter_deterministic_citation_findings(
                                    raw_verification_findings,
                                    final_citation_validation,
                                )
                                verifier_findings = (
                                    self._with_report_requirement_findings(
                                        verifier_findings,
                                        task=task,
                                        report=report,
                                    )
                                )
                                suppressed_final_findings.extend(
                                    suppressed_after_verifier_follow_up
                                )
                                final_findings = self._actionable_redteam_findings(
                                    verifier_findings,
                                    coverage_matrix=evidence_coverage,
                                )
                                final_advisory_findings = (
                                    self._advisory_redteam_findings(
                                        verifier_findings,
                                        coverage_matrix=evidence_coverage,
                                    )
                                )
                                effective_final_status = self._effective_review_status(
                                    verification_trace,
                                    final_findings,
                                    reviewed_findings=verifier_findings,
                                    suppressed_findings=(
                                        suppressed_after_verifier_follow_up
                                    ),
                                )

                terminal_research_exhausted = (
                    not self.config.research_loop.enabled
                    or len(research_rounds) >= self.config.research_loop.max_rounds
                    or research_stop_reason
                    in {
                        ResearchStopReason.MAX_ROUNDS_REACHED,
                        ResearchStopReason.NO_PROGRESS,
                    }
                    # 显式禁用 Verifier 补研时，终局也没有可用的新证据轮次。
                    or not self.config.research_loop.allow_verifier_follow_up
                    # 第一版最多执行一轮 Verifier 补研；已经尝试后不会在此继续扩张循环。
                    or verifier_follow_up_attempted
                )
                (
                    terminal_repairable_findings,
                    terminal_deferred_findings,
                    terminal_action_conversions,
                ) = self._route_terminal_revision_findings(
                    final_findings,
                    allow_terminal_removal=terminal_research_exhausted,
                )
                text_only_findings = terminal_repairable_findings
                required_final_calls = (
                    2 if self.config.redteam.verify_after_final_text_revision else 1
                )
                should_run_final_text_revision = (
                    self.config.redteam.enable_final_text_revision
                    and revision_succeeded
                    and verification_trace.used_provider == "llm"
                    and effective_final_status in {"needs_revision", "fail"}
                    and bool(text_only_findings)
                    and isinstance(self.synthesizer, LLMSynthesizerAgent)
                    and isinstance(self.redteam, LLMRedTeamAgent)
                    and self._has_llm_call_capacity(
                        llm_call_budget,
                        required_calls=required_final_calls,
                    )
                    and self._exhausted_budget_limit(
                        task,
                        llm_trace_offsets,
                        started,
                        base_usage=research_state.budget,
                    )
                    is None
                )
                if should_run_final_text_revision:
                    final_text_revision_attempted = True
                    report = await self.synthesizer.revise_text_only_async(
                        task,
                        report,
                        report.evidence or synthesis_evidence,
                        text_only_findings,
                        coverage_matrix=evidence_coverage,
                    )
                    final_text_revision_trace = self.synthesizer.last_trace
                    final_text_revision_succeeded = (
                        final_text_revision_trace.used_provider == "llm"
                    )
                    final_text_revised_report = (
                        report if final_text_revision_succeeded else None
                    )
                    final_citation_validation = validate_report_citations(
                        report,
                        require_citations=task.require_citations,
                    )
                    if final_text_revision_succeeded:
                        # 修订本身不能将状态抬升为 pass，必须等待复核。
                        effective_final_status = "not_verified"
                    if (
                        self.config.redteam.verify_after_final_text_revision
                        and final_text_revision_succeeded
                        and self._exhausted_budget_limit(
                            task,
                            llm_trace_offsets,
                            started,
                            base_usage=research_state.budget,
                        )
                        is None
                    ):
                        raw_final_text_verification_findings = (
                            await self.redteam.review_async(
                                report.key_findings,
                                synthesis_evidence,
                                task=task,
                                report=report,
                                mode="verify",
                                # 复核应看到修订前的全部问题，而不只是送给 Blue 的安全子集。
                                prior_findings=final_findings,
                                coverage_matrix=evidence_coverage,
                            )
                        )
                        final_text_verification_trace = self.redteam.last_trace
                        (
                            final_text_verification_findings,
                            suppressed_final_text_findings,
                        ) = self._filter_deterministic_citation_findings(
                            raw_final_text_verification_findings,
                            final_citation_validation,
                        )
                        final_text_verification_findings = (
                            self._with_report_requirement_findings(
                                final_text_verification_findings,
                                task=task,
                                report=report,
                            )
                        )
                        post_repair_findings = self._actionable_redteam_findings(
                            final_text_verification_findings,
                            coverage_matrix=evidence_coverage,
                        )
                        # 没有交给文本修订的证据型问题按设计仍然未解决。即使模型复核
                        # 偶然漏报，也不能让它们从质量状态中消失。
                        final_findings = self._dedupe_redteam_findings(
                            [*terminal_deferred_findings, *post_repair_findings]
                        )
                        final_advisory_findings = self._advisory_redteam_findings(
                            final_text_verification_findings,
                            coverage_matrix=evidence_coverage,
                        )
                        effective_final_status = self._effective_review_status(
                            final_text_verification_trace,
                            final_findings,
                            reviewed_findings=final_text_verification_findings,
                            suppressed_findings=suppressed_final_text_findings,
                        )
                report = self._with_redteam_findings(
                    report,
                    [*final_findings, *final_advisory_findings],
                )
            else:
                redteam_trace = RedTeamTrace(
                    requested_provider=(
                        self.config.redteam.provider
                        if self.config.runtime.enable_redteam
                        else "disabled"
                    ),
                    used_provider="skipped" if budget_limit else "disabled",
                    summary=f"budget_exhausted:{budget_limit}" if budget_limit else None,
                )
            # 用户报告只展示研究结论、引用和自然语言限制；内部覆盖矩阵仍仅进入 Replay。
            report = self._with_public_coverage_disclosure(report, evidence_coverage)
            (
                finding_substance_diagnostics,
                substance_gate_findings,
            ) = self._finding_substance_findings(task, report)
            (
                methodology_obligation_diagnostics,
                source_absence_diagnostics,
                methodology_gate_findings,
            ) = self._methodology_quality_findings(task, report)
            if substance_gate_findings or methodology_gate_findings:
                # 该门禁位于所有模型修订之后，是规则 fallback 和异常分支的最后保险。
                # 这里只阻止错误地记为 pass；不在终局偷偷生成新的用户文本。
                final_findings = self._dedupe_redteam_findings(
                    [
                        *final_findings,
                        *substance_gate_findings,
                        *methodology_gate_findings,
                    ]
                )
                effective_final_status = "needs_revision"
                report = self._with_redteam_findings(
                    report,
                    [*final_findings, *final_advisory_findings],
                )
            elapsed = time.perf_counter() - started
            llm_calls = self._collect_llm_calls(llm_trace_offsets)
            input_tokens = sum(trace.input_tokens or 0 for trace in llm_calls)
            output_tokens = sum(trace.output_tokens or 0 for trace in llm_calls)
            cost_usd = sum(trace.cost_usd or 0.0 for trace in llm_calls)
            budget_limit = budget_limit or self._exhausted_budget_limit(
                task,
                llm_trace_offsets,
                started,
                base_usage=research_state.budget,
            )
            if budget_limit is not None:
                research_stop_reason = ResearchStopReason.BUDGET_EXHAUSTED
            elif effective_final_status in {"needs_revision", "fail", "not_verified"}:
                # 研究节点完成或补研拿到新证据，不代表报告已经通过最终质量审查。
                research_stop_reason = ResearchStopReason.QUALITY_UNRESOLVED
            completion_status = self._derive_completion_status(
                quality_status=effective_final_status,
                stop_reason=research_stop_reason,
                coverage_matrix=evidence_coverage,
                evidence_gaps=evidence_gaps,
                query=task.query,
            )
            research_state = self._complete_research_state(
                research_state,
                subtasks=subtasks,
                dag_result=dag_result,
                all_evidence=all_evidence,
                report=report,
                elapsed_seconds=elapsed,
                llm_calls=llm_calls,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost_usd,
                final_review_status=effective_final_status,
                citation_validation=final_citation_validation,
                research_rounds=research_rounds,
                evidence_gaps=evidence_gaps,
                stop_reason=research_stop_reason,
                final_findings=final_findings,
                pre_research_findings=pre_research_findings,
                research_follow_up_succeeded=(
                    research_follow_up_succeeded or verifier_follow_up_succeeded
                ),
                exhausted_budget_limit=budget_limit,
            )
            if self.memory_store is not None:
                self.memory_store.save_research_state(research_state)
            quality_diagnostics = self._build_quality_diagnostics(
                dag_result.results,
                all_evidence,
                report,
                llm_calls,
                compression_result,
            )
            # replay 是复盘和评测的主要依据，尽量保存结构化对象而不是只保存最终文本。
            report_artifact_paths = self.report_artifacts.paths_for(
                task.run_id,
                task.created_at,
            )
            replay_path = self.replay.write(
                task.run_id,
                {
                    "task": task,
                    # 统一状态独立于 DAG/报告保存，为多轮研究和任务级恢复提供稳定入口。
                    "research_state": research_state,
                    "planner": planner_trace,
                    "synthesizer": synthesizer_trace,
                    # redteam 保留为首次攻击 trace，兼容已有 Replay 消费方。
                    "redteam": redteam_trace,
                    "quality_cycle": {
                        "enabled": self.config.runtime.enable_redteam,
                        "research_follow_up_attempted": research_follow_up_attempted,
                        "research_follow_up_succeeded": research_follow_up_succeeded,
                        "verifier_follow_up_attempted": verifier_follow_up_attempted,
                        "verifier_follow_up_succeeded": verifier_follow_up_succeeded,
                        "pre_verifier_follow_up_verification": (
                            pre_verifier_follow_up_verification
                        ),
                        "pre_verifier_follow_up_findings": (
                            pre_verifier_follow_up_findings
                        ),
                        "verifier_follow_up_revision": verifier_follow_up_revision_trace,
                        "verifier_follow_up_verification": (
                            verifier_follow_up_verification_trace
                        ),
                        "pre_research_report": pre_research_report,
                        "pre_research_review": pre_research_review,
                        "pre_research_findings": pre_research_findings,
                        "draft_report": draft_report,
                        "draft_citation_validation": draft_citation_validation,
                        "draft_review": redteam_trace,
                        "raw_draft_findings": raw_draft_review_findings,
                        "draft_findings": draft_review_findings,
                        "suppressed_draft_findings": suppressed_draft_findings,
                        "actionable_draft_findings": actionable_draft_findings,
                        "advisory_draft_findings": advisory_draft_findings,
                        "effective_draft_status": effective_draft_status,
                        "revision_attempted": revision_attempted,
                        "revision_succeeded": revision_succeeded,
                        "revision": revision_trace,
                        "revised_report": revised_report,
                        "final_citation_validation": final_citation_validation,
                        "final_verification": verification_trace,
                        "raw_verification_findings": raw_verification_findings,
                        "suppressed_final_findings": suppressed_final_findings,
                        "final_text_revision_attempted": final_text_revision_attempted,
                        "final_text_revision_succeeded": final_text_revision_succeeded,
                        "final_text_revision": final_text_revision_trace,
                        "final_text_revised_report": final_text_revised_report,
                        "final_text_verification": final_text_verification_trace,
                        "raw_final_text_verification_findings": (
                            raw_final_text_verification_findings
                        ),
                        "suppressed_final_text_findings": suppressed_final_text_findings,
                        "terminal_repairable_findings": terminal_repairable_findings,
                        "terminal_deferred_findings": terminal_deferred_findings,
                        "terminal_action_conversions": terminal_action_conversions,
                        "finding_substance_diagnostics": finding_substance_diagnostics,
                        "methodology_obligation_diagnostics": (
                            methodology_obligation_diagnostics
                        ),
                        "source_absence_diagnostics": source_absence_diagnostics,
                        "final_findings": final_findings,
                        "final_advisory_findings": final_advisory_findings,
                        "effective_final_status": effective_final_status,
                        # Verifier 结论与研究完成度是两个维度；例如引用全部受支持，
                        # 仍可能因为命名对象缺少必答维度而仅达到 partial_coverage。
                        "completion_status": completion_status,
                    },
                    "subtasks": subtasks,
                    # dag 是面向人工复盘的拓扑摘要，node_traces 保留逐节点执行细节。
                    "dag": build_dag_replay(
                        subtasks,
                        dag_result.traces,
                        started_order=dag_result.started_order,
                        completed_order=dag_result.completed_order,
                    ),
                    "node_traces": dag_result.traces,
                    # node_results 显式记录每个节点自产出和从依赖继承的数据血缘。
                    "node_results": self._build_node_results_replay(dag_result.results),
                    "docs_by_task": docs_by_task,
                    "chunks_by_task": chunks_by_task,
                    "evidence": all_evidence,
                    # synthesis_context 明确记录本次送入写作/核验阶段的压缩证据集合。
                    "synthesis_context": compression_result,
                    # 覆盖矩阵显式记录每个对象的目标/数据/指标/范围是否有证据。
                    "evidence_coverage": evidence_coverage,
                    "report": report,
                    # 报告目录是面向阅读和外部服务的稳定产物；Replay 继续保存完整研究轨迹。
                    "report_artifacts": report_artifact_paths.as_dict(),
                    "llm_calls": llm_calls,
                    "diagnostics": quality_diagnostics,
                    "metrics": {
                        "latency_seconds": elapsed,
                        "llm_input_tokens": input_tokens,
                        "llm_output_tokens": output_tokens,
                        "cost_usd": cost_usd,
                    },
                },
            )
            self.report_artifacts.write(
                report,
                created_at=task.created_at,
                replay_path=replay_path,
                quality_status=effective_final_status,
                stop_reason=research_state.stop_reason.value,
                completion_status=completion_status,
                coverage_required=self._coverage_is_required(
                    evidence_coverage,
                    query=task.query,
                ),
                coverage_complete=evidence_coverage.complete,
                missing_dimension_count=sum(
                    len(row.missing_dimensions) for row in evidence_coverage.subjects
                ),
                open_gap_count=sum(
                    gap.status == EvidenceGapStatus.OPEN for gap in evidence_gaps
                ),
            )
            if self.memory_store is not None:
                self.memory_store.finish_run(task.run_id, replay_path=replay_path)
            return report
        except Exception as exc:
            if self.memory_store is not None and memory_started:
                try:
                    failed_calls = self._collect_llm_calls(llm_trace_offsets)
                    failed_input_tokens = sum(call.input_tokens or 0 for call in failed_calls)
                    failed_output_tokens = sum(call.output_tokens or 0 for call in failed_calls)
                    failed_cost = sum(call.cost_usd or 0.0 for call in failed_calls)
                    failed_state = research_state.model_copy(
                        update={
                            "status": ResearchRunStatus.FAILED,
                            "stop_reason": ResearchStopReason.ERROR,
                            "last_error": self._safe_error(exc),
                            "budget": research_state.budget.model_copy(
                                update={
                                    "elapsed_seconds": (
                                        research_state.budget.elapsed_seconds
                                        + time.perf_counter()
                                        - started
                                    ),
                                    "llm_calls": (
                                        research_state.budget.llm_calls + len(failed_calls)
                                    ),
                                    "llm_input_tokens": (
                                        research_state.budget.llm_input_tokens
                                        + failed_input_tokens
                                    ),
                                    "llm_output_tokens": (
                                        research_state.budget.llm_output_tokens
                                        + failed_output_tokens
                                    ),
                                    "cost_usd": research_state.budget.cost_usd + failed_cost,
                                }
                            ),
                            "updated_at": utc_now(),
                        }
                    )
                    self.memory_store.save_research_state(failed_state)
                    self.memory_store.fail_run(task.run_id, self._safe_error(exc))
                except Exception:
                    # 状态存储失败不能覆盖真正导致研究任务失败的异常。
                    pass
            raise
        finally:
            for client in self.llm_clients:
                client.unbind_run()

    @staticmethod
    def _complete_research_state(
        state: ResearchState,
        *,
        subtasks: list[SubTask],
        dag_result,
        all_evidence: list[Evidence],
        report: ResearchReport,
        elapsed_seconds: float,
        llm_calls: list[LLMCallTrace],
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        final_review_status: str,
        citation_validation: CitationValidation,
        research_rounds: list[ResearchRound],
        evidence_gaps: list[EvidenceGap],
        stop_reason: ResearchStopReason,
        final_findings: list[RedTeamFinding],
        pre_research_findings: list[RedTeamFinding],
        research_follow_up_succeeded: bool,
        exhausted_budget_limit: str | None,
    ) -> ResearchState:
        """Summarize the current single pass into the multi-round state contract."""

        docs = [doc for result in dag_result.results.values() for doc in result.docs]
        source_ids = list(dict.fromkeys(doc.id for doc in docs))
        source_urls = list(dict.fromkeys(doc.url for doc in docs if doc.url))

        # “confirmed”只用于通过最终语义核验的结论；其余报告仍可存在，但不冒充已确认事实。
        confirmed_conclusions = (
            list(report.key_findings)
            if final_review_status == "pass" and citation_validation.valid
            else []
        )
        final_conflicts = [
            ResearchConflict(
                claim=finding.claim or finding.issue,
                description=finding.reason or finding.issue,
                source="verifier" if final_review_status != "skipped" else "redteam",
                evidence_ids=list(finding.evidence_ids),
            )
            for finding in final_findings
            if finding.status == "contradicted"
        ]
        final_conflict_keys = {
            " ".join(item.claim.casefold().split()) for item in final_conflicts
        }
        conflicts = list(final_conflicts)
        for finding in pre_research_findings:
            if finding.status != "contradicted":
                continue
            claim = finding.claim or finding.issue
            if " ".join(claim.casefold().split()) in final_conflict_keys:
                continue
            resolved = research_follow_up_succeeded and final_review_status == "pass"
            conflicts.append(
                ResearchConflict(
                    claim=claim,
                    description=finding.reason or finding.issue,
                    source="redteam",
                    status=(
                        ResearchConflictStatus.RESOLVED
                        if resolved
                        else ResearchConflictStatus.OPEN
                    ),
                    evidence_ids=list(finding.evidence_ids),
                    resolution_reason=(
                        "Additional evidence was retrieved and the post-research review passed."
                        if resolved
                        else None
                    ),
                    resolved_at=utc_now() if resolved else None,
                )
            )
        rejected_evidence = ResearchPipeline._collect_rejected_evidence(dag_result.results)
        source_credibility = {
            doc.id: doc.credibility for doc in docs if doc.credibility is not None
        }
        return state.model_copy(
            update={
                "status": ResearchRunStatus.DONE,
                "current_round": len(research_rounds),
                "rounds": research_rounds,
                "pending_questions": [
                    gap.question for gap in evidence_gaps if gap.status == EvidenceGapStatus.OPEN
                ],
                "evidence_gaps": evidence_gaps,
                "confirmed_conclusions": confirmed_conclusions,
                "conflicts": conflicts,
                "rejected_evidence": rejected_evidence,
                "conflict_ids": [item.id for item in conflicts],
                "rejected_evidence_ids": [item.id for item in rejected_evidence],
                "visited_source_ids": source_ids,
                "visited_source_urls": source_urls,
                "source_credibility": source_credibility,
                "evidence_ids": [item.id for item in all_evidence],
                "budget": state.budget.model_copy(
                    update={
                        "elapsed_seconds": state.budget.elapsed_seconds + elapsed_seconds,
                        "llm_calls": state.budget.llm_calls + len(llm_calls),
                        "llm_input_tokens": state.budget.llm_input_tokens + input_tokens,
                        "llm_output_tokens": state.budget.llm_output_tokens + output_tokens,
                        "cost_usd": state.budget.cost_usd + cost_usd,
                        "documents_retrieved": len(docs),
                        "evidence_count": len(all_evidence),
                        "exhausted": exhausted_budget_limit is not None,
                        "exhausted_limit": exhausted_budget_limit,
                    }
                ),
                "stop_reason": stop_reason,
                "updated_at": utc_now(),
            }
        )

    @staticmethod
    def _collect_rejected_evidence(
        results: dict[str, ResearchNodeResult],
    ) -> list[RejectedEvidence]:
        """Promote Reader rejection diagnostics into queryable state records."""

        rejected: list[RejectedEvidence] = []
        for subtask_id, result in results.items():
            reader = result.diagnostics.get("reader", {})
            if not isinstance(reader, dict):
                continue
            doc_results = reader.get("doc_results", [])
            if not isinstance(doc_results, list):
                continue
            for doc_result in doc_results:
                if not isinstance(doc_result, dict):
                    continue
                error = doc_result.get("error")
                if not error or doc_result.get("status") not in {"FAILED", "FALLBACK"}:
                    continue
                rejected.append(
                    RejectedEvidence(
                        stage="reader",
                        reason=str(error),
                        subtask_id=subtask_id,
                        source_id=str(doc_result.get("source_id") or "") or None,
                    )
                )
        return rejected

    async def _run_follow_up_rounds(
        self,
        initial_subtasks: list[SubTask],
        combined_result: DAGExecutionResult[ResearchNodeResult],
        handler,
        research_query: str,
        budget_check,
        on_node_complete,
        llm_call_budget: LLMCallBudget | None,
    ) -> tuple[list[SubTask], list[ResearchRound], list[EvidenceGap], ResearchStopReason]:
        """Run bounded follow-up rounds for node failures and named coverage gaps."""

        all_subtasks = list(initial_subtasks)
        gaps = self._identify_evidence_gaps(initial_subtasks, combined_result)
        initial_evidence = self._dedupe_evidence(
            [
                item
                for result in combined_result.results.values()
                for item in result.evidence
            ]
        )
        # 节点成功只说明“抽到了某些证据”，不代表比较题中每个对象的每个必答维度均有覆盖。
        # 在首次写作前补齐这些确定性缺口，避免完全依赖 RedTeam 是否偶然指出遗漏。
        gaps.extend(self._identify_coverage_gaps(research_query, initial_evidence))
        rounds = [
            self._build_round_record(
                1,
                initial_subtasks,
                combined_result,
                opened_gap_ids=[gap.id for gap in gaps],
            )
        ]
        if not self.config.research_loop.enabled:
            return all_subtasks, rounds, gaps, ResearchStopReason.SINGLE_PASS_COMPLETE
        if not gaps:
            return all_subtasks, rounds, gaps, ResearchStopReason.GOAL_SATISFIED

        known_signatures = self._evidence_signatures(combined_result.results.values())
        stop_reason = ResearchStopReason.MAX_ROUNDS_REACHED
        for round_number in range(2, self.config.research_loop.max_rounds + 1):
            if budget_check() is not None:
                stop_reason = ResearchStopReason.BUDGET_EXHAUSTED
                break
            open_gaps = [gap for gap in gaps if gap.status == EvidenceGapStatus.OPEN]
            if not open_gaps:
                stop_reason = ResearchStopReason.GOAL_SATISFIED
                break
            selected_gaps = open_gaps[: self.config.research_loop.max_follow_up_subtasks]
            follow_up_subtasks = [
                SubTask(
                    id=f"round_{round_number}_task_{index}",
                    question=gap.question,
                    retrieval_query=gap.retrieval_query,
                    task_type="research",
                    retrieval_top_k=(
                        self.config.retrieval.top_k
                        * self.config.research_loop.follow_up_candidate_multiplier
                    ),
                )
                for index, gap in enumerate(selected_gaps, start=1)
            ]
            if llm_call_budget is not None:
                async with llm_call_budget.reserve_for_later(
                    self.config.research_loop.reserve_llm_calls_for_finalization
                ):
                    round_result = await self.dag.run_async(
                        follow_up_subtasks,
                        handler,
                        on_node_complete=on_node_complete,
                    )
            else:
                round_result = await self.dag.run_async(
                    follow_up_subtasks,
                    handler,
                    on_node_complete=on_node_complete,
                )
            all_subtasks.extend(follow_up_subtasks)

            previous_signatures = set(known_signatures)
            current_signatures = self._evidence_signatures(round_result.results.values())
            new_signatures = current_signatures - known_signatures
            known_signatures.update(current_signatures)
            resolved_gap_ids: list[str] = []
            updated_gaps: list[EvidenceGap] = []
            selected_by_task = dict(zip((item.id for item in follow_up_subtasks), selected_gaps))
            for gap in gaps:
                matching_task_id = next(
                    (
                        task_id
                        for task_id, selected_gap in selected_by_task.items()
                        if selected_gap.id == gap.id
                    ),
                    None,
                )
                if matching_task_id is None:
                    updated_gaps.append(gap)
                    continue
                result = round_result.results.get(matching_task_id)
                related_ids = list(dict.fromkeys([*gap.related_subtask_ids, matching_task_id]))
                matching_evidence = self._gap_matching_evidence(
                    gap,
                    result.evidence if result is not None else [],
                )
                if matching_evidence:
                    resolved_gap_ids.append(gap.id)
                    updated_gaps.append(
                        gap.model_copy(
                            update={
                                "status": EvidenceGapStatus.RESOLVED,
                                "related_subtask_ids": related_ids,
                                "related_evidence_ids": [item.id for item in matching_evidence],
                                "resolved_at": utc_now(),
                            }
                        )
                    )
                else:
                    updated_gaps.append(
                        gap.model_copy(update={"related_subtask_ids": related_ids})
                    )
            gaps = updated_gaps
            rounds.append(
                self._build_round_record(
                    round_number,
                    follow_up_subtasks,
                    round_result,
                    resolved_gap_ids=resolved_gap_ids,
                    previous_signatures=previous_signatures,
                )
            )
            self._merge_dag_results(combined_result, round_result)

            if not [gap for gap in gaps if gap.status == EvidenceGapStatus.OPEN]:
                stop_reason = ResearchStopReason.GOAL_SATISFIED
                break
            if (
                self.config.research_loop.stop_on_no_progress
                and len(new_signatures) < self.config.research_loop.min_new_evidence_per_round
            ):
                stop_reason = ResearchStopReason.NO_PROGRESS
                break
        return all_subtasks, rounds, gaps, stop_reason

    async def _run_redteam_follow_up_round(
        self,
        round_number: int,
        research_query: str,
        findings: list[RedTeamFinding],
        existing_result: DAGExecutionResult[ResearchNodeResult],
        handler,
        on_node_complete,
        llm_call_budget: LLMCallBudget | None,
        review_source: Literal["redteam", "verifier"] = "redteam",
    ) -> tuple[
        list[SubTask],
        DAGExecutionResult[ResearchNodeResult],
        ResearchRound,
        list[EvidenceGap],
        ResearchStopReason,
    ] | None:
        """Turn evidence-solvable RedTeam findings into one bounded research round."""

        gaps = self._redteam_evidence_gaps(
            findings,
            research_query,
            source=review_source,
            evidence=self._dedupe_evidence(
                [
                    item
                    for result in existing_result.results.values()
                    for item in result.evidence
                ]
            ),
        )
        if not gaps:
            return None
        selected_gaps = gaps[: self.config.research_loop.max_follow_up_subtasks]
        subtasks = [
            SubTask(
                id=f"round_{round_number}_{review_source}_{index}",
                question=gap.question,
                retrieval_query=gap.retrieval_query,
                task_type="research",
                retrieval_top_k=(
                    self.config.retrieval.top_k
                    * self.config.research_loop.follow_up_candidate_multiplier
                ),
            )
            for index, gap in enumerate(selected_gaps, start=1)
        ]
        previous_signatures = self._evidence_signatures(existing_result.results.values())
        if llm_call_budget is not None:
            async with llm_call_budget.reserve_for_later(
                self.config.research_loop.reserve_llm_calls_for_finalization
            ):
                result = await self.dag.run_async(
                    subtasks,
                    handler,
                    on_node_complete=on_node_complete,
                )
        else:
            result = await self.dag.run_async(
                subtasks,
                handler,
                on_node_complete=on_node_complete,
            )
        new_signatures = self._evidence_signatures(result.results.values()) - previous_signatures

        resolved_ids: list[str] = []
        updated_gaps: list[EvidenceGap] = []
        for gap, subtask in zip(selected_gaps, subtasks):
            node_result = result.results.get(subtask.id)
            related_ids = [*gap.related_subtask_ids, subtask.id]
            matching_evidence = self._gap_matching_evidence(
                gap,
                node_result.evidence if node_result is not None else [],
            )
            if matching_evidence:
                resolved_ids.append(gap.id)
                updated_gaps.append(
                    gap.model_copy(
                        update={
                            "status": EvidenceGapStatus.RESOLVED,
                            "related_subtask_ids": related_ids,
                            "related_evidence_ids": [item.id for item in matching_evidence],
                            "resolved_at": utc_now(),
                        }
                    )
                )
            else:
                updated_gaps.append(
                    gap.model_copy(update={"related_subtask_ids": related_ids})
                )

        research_round = self._build_round_record(
            round_number,
            subtasks,
            result,
            opened_gap_ids=[gap.id for gap in selected_gaps],
            resolved_gap_ids=resolved_ids,
            previous_signatures=previous_signatures,
        )
        if len(resolved_ids) == len(selected_gaps):
            stop_reason = ResearchStopReason.GOAL_SATISFIED
        elif len(new_signatures) < self.config.research_loop.min_new_evidence_per_round:
            stop_reason = ResearchStopReason.NO_PROGRESS
        elif round_number >= self.config.research_loop.max_rounds:
            stop_reason = ResearchStopReason.MAX_ROUNDS_REACHED
        else:
            # 第一版每次 RedTeam 攻击最多触发一轮，后续攻击结果仍可留作开放缺口。
            stop_reason = ResearchStopReason.FOLLOW_UP_COMPLETE
        return subtasks, result, research_round, updated_gaps, stop_reason

    @staticmethod
    def _redteam_evidence_gaps(
        findings: list[RedTeamFinding],
        research_query: str = "",
        *,
        source: Literal["redteam", "verifier"] = "redteam",
        evidence: list[Evidence] | None = None,
    ) -> list[EvidenceGap]:
        """Create bounded, entity-focused research gaps from evidence-solvable review issues."""

        research_categories = {
            "factuality",
            "citation_support",
            "coverage",
            "time_scope",
        }
        research_statuses = {"weakly_supported", "unsupported", "contradicted", "unclear"}
        eligible = [
            finding
            for finding in findings
            # category 是结构化路由信号；不再因 unclear/unsupported 就把逻辑或
            # 措辞问题误送到 Retriever。status 只在证据型类别内判断是否可操作。
            if finding.category in research_categories
            and (finding.status in research_statuses or finding.status is None)
            # 可通过删除/收窄现有文字解决的 citation_support 问题无需再消耗一次检索轮次。
            and not (
                finding.category == "citation_support"
                and finding.repair_action in {"narrow", "remove", "qualify"}
            )
        ]

        # 比较题常出现“ALCE 与 RAGTruth 都缺证据”这类宽泛意见。先从原问题抽取
        # 明确命名对象，再只在含缺失语义的 review 句子中匹配，避免把普通缩写误拆成任务。
        query_entities = ResearchPipeline._named_query_entities(research_query)
        entity_findings: dict[str, list[RedTeamFinding]] = {}
        findings_covered_by_entity: set[int] = set()
        for finding_index, finding in enumerate(eligible):
            text = "。".join(
                part
                for part in (
                    finding.issue,
                    finding.reason,
                    finding.suggested_fix,
                    finding.claim,
                )
                if part
            )
            sentences = re.split(r"[。！？!?;；.\n]+", text)
            for entity in query_entities:
                if any(
                    entity.casefold() in sentence.casefold()
                    and ResearchPipeline._contains_missing_evidence_marker(sentence)
                    for sentence in sentences
                ):
                    entity_findings.setdefault(entity, []).append(finding)
                    findings_covered_by_entity.add(finding_index)

        gaps: list[EvidenceGap] = []
        evidence_by_id = {item.id: item for item in evidence or []}
        for entity, related in entity_findings.items():
            reasons = list(dict.fromkeys(item.issue for item in related if item.issue))
            evidence_ids = list(
                dict.fromkeys(
                    evidence_id
                    for item in related
                    for evidence_id in item.evidence_ids
                )
            )
            gaps.append(
                EvidenceGap(
                    question=(
                        f"围绕原研究问题“{research_query}”，单独检索并核查 {entity} 的第一手来源；"
                        "优先原始论文、官方项目页或机构文档，并提取可回溯原文证据。"
                    ),
                    retrieval_query=(
                        f"{entity} original paper official benchmark dataset evaluation metrics"
                    ),
                    target_entities=[entity],
                    resolution_terms=[
                        "benchmark",
                        "dataset",
                        "evaluation",
                        "metric",
                        "citation",
                        "precision",
                        "recall",
                    ],
                    reason="；".join(reasons),
                    source=source,
                    priority="high",
                    related_evidence_ids=evidence_ids,
                )
            )

        # 没有明确对象或无法对象化的问题仍保留原有补研方式，保证通用研究题兼容。
        seen_questions = {" ".join(gap.question.casefold().split()) for gap in gaps}
        for finding_index, finding in enumerate(eligible):
            if finding_index in findings_covered_by_entity:
                continue
            if finding.category not in research_categories and finding.status not in research_statuses:
                continue
            focus = finding.claim or finding.issue
            question = f"{focus}；补充核查要求：{finding.suggested_fix}"
            normalized = " ".join(question.casefold().split())
            if normalized in seen_questions:
                continue
            priority = finding.severity if finding.severity in {"low", "medium", "high"} else "medium"
            # Verifier 经常已经知道“哪篇研究只有书目信息、缺少结论”。此时把对应
            # Evidence 的原文片段作为搜索锚点，比把整段中文审查意见交给搜索引擎更有效。
            anchored_evidence = [
                evidence_by_id[evidence_id]
                for evidence_id in finding.evidence_ids
                if evidence_id in evidence_by_id
            ]
            anchored_pairs: list[tuple[Evidence, str]] = []
            seen_anchored_queries: set[str] = set()
            for item in anchored_evidence:
                retrieval_query = ResearchPipeline._evidence_follow_up_query(item)
                if not retrieval_query or retrieval_query in seen_anchored_queries:
                    continue
                anchored_pairs.append((item, retrieval_query))
                seen_anchored_queries.add(retrieval_query)
                if len(anchored_pairs) >= 2:
                    break
            if anchored_pairs:
                for item, retrieval_query in anchored_pairs:
                    anchored_question = (
                        f"{question}；定向补齐来源线索：{retrieval_query[:160]}。"
                    )
                    anchored_normalized = " ".join(
                        anchored_question.casefold().split()
                    )
                    if anchored_normalized in seen_questions:
                        continue
                    gaps.append(
                        EvidenceGap(
                            question=anchored_question,
                            retrieval_query=retrieval_query,
                            reason=finding.issue,
                            source=source,
                            priority=priority,
                            related_evidence_ids=[item.id],
                        )
                    )
                    seen_questions.add(anchored_normalized)
                continue
            gaps.append(
                EvidenceGap(
                    question=question,
                    reason=finding.issue,
                    source=source,
                    priority=priority,
                    related_evidence_ids=list(finding.evidence_ids),
                )
            )
            seen_questions.add(normalized)
        return gaps

    @staticmethod
    def _evidence_follow_up_query(evidence: Evidence) -> str:
        """Build a bounded exact-study query from an already cited but incomplete source."""

        candidates = (evidence.quote, evidence.claim, evidence.source_title)
        anchor = next((" ".join(item.split()) for item in candidates if len(item.strip()) >= 20), "")
        if not anchor:
            return ""
        # Tavily and conventional search engines work better with one bibliographic sentence
        # than with the complete Verifier explanation or a multi-paragraph evidence quote.
        return f"{anchor[:240]} original paper full text findings conclusion results"

    @staticmethod
    def _gap_matching_evidence(gap: EvidenceGap, evidence: list[Evidence]) -> list[Evidence]:
        """Return evidence that satisfies explicit entity/dimension requirements for one gap.

        Generic retry gaps intentionally keep the original "any traceable evidence" behavior. Only
        entity-focused RedTeam gaps carry strict requirements, so existing retry semantics stay stable.
        """

        if not gap.target_entities and not gap.resolution_terms:
            return list(evidence)
        matched: list[Evidence] = []
        for item in evidence:
            haystack = " ".join(
                (item.source_title, item.claim, item.quote, item.source_url)
            ).casefold()
            entity_match = not gap.target_entities or any(
                entity.casefold() in haystack for entity in gap.target_entities
            )
            dimension_match = not gap.resolution_terms or any(
                ResearchPipeline._matches_coverage_term(haystack, term)
                for term in gap.resolution_terms
            )
            if entity_match and dimension_match:
                matched.append(item)
        return matched

    @staticmethod
    def _build_evidence_coverage_matrix(
        query: str,
        evidence: list[Evidence],
    ) -> EvidenceCoverageMatrix:
        """Map named research objects to goal/data/metrics/scope Evidence IDs.

        这是可解释的词法覆盖诊断，不代替 Verifier 的语义判断。查询没有明确
        专名时使用 overall 行，保证普通研究题仍能获得整体覆盖摘要。
        """

        dimension_terms = ResearchPipeline._coverage_dimension_terms()
        subjects = ResearchPipeline._named_query_entities(query) or ["overall"]
        rows: list[EvidenceCoverageSubject] = []
        for subject in subjects:
            subject_evidence: list[Evidence] = []
            for item in evidence:
                text = " ".join(
                    (item.source_title, item.claim, item.quote, item.source_url)
                ).casefold()
                if subject == "overall" or subject.casefold() in text:
                    subject_evidence.append(item)
            dimension_ids: dict[str, list[str]] = {}
            for dimension, terms in dimension_terms.items():
                dimension_ids[dimension] = [
                    item.id
                    for item in subject_evidence
                    if any(
                        ResearchPipeline._matches_coverage_term(
                            " ".join((item.source_title, item.claim, item.quote)),
                            term,
                        )
                        for term in terms
                    )
                ]
            rows.append(
                EvidenceCoverageSubject(
                    subject=subject,
                    evidence_ids=[item.id for item in subject_evidence],
                    dimension_evidence_ids=dimension_ids,
                    missing_dimensions=[
                        dimension for dimension, ids in dimension_ids.items() if not ids
                    ],
                )
            )
        return EvidenceCoverageMatrix(
            subjects=rows,
            complete=bool(rows) and all(not row.missing_dimensions for row in rows),
        )

    @staticmethod
    def _coverage_dimension_terms() -> dict[str, tuple[str, ...]]:
        """Return the shared lexical vocabulary used by coverage checks and follow-up gaps."""

        return {
            "goal": (
                "goal",
                "objective",
                "aim",
                "aims to",
                "designed to",
                "intended to",
                "tailored for",
                "created to",
                "in order to",
                "propose",
                "introduce",
                "purpose",
                "目标",
                "目的",
                "定位",
                "提出",
                "用于",
                "旨在",
                "面向",
                "为了",
            ),
            "data": (
                "dataset",
                "corpus",
                "sample",
                "task",
                "data",
                "benchmark",
                "数据集",
                "语料库",
                "语料",
                "样本",
                "任务",
                "数据",
                "基准",
            ),
            "metrics": (
                "metric",
                "evaluation",
                "evaluate",
                "precision",
                "recall",
                "accuracy",
                "score",
                "citation",
                "指标",
                "评估",
                "评测",
                "精确率",
                "召回率",
                "准确率",
                "引用",
            ),
            "scope": (
                "scope",
                "domain",
                "language",
                "application",
                "applicable",
                "use case",
                "范围",
                "领域",
                "语言",
                "适用",
                "应用",
                "场景",
            ),
        }

    @staticmethod
    def _matches_coverage_term(text: str, term: str) -> bool:
        """Match Latin vocabulary on token boundaries and CJK phrases as substrings.

        The old raw-substring check treated ``aim`` in ``claims`` and ``data`` in
        ``metadata`` as coverage. Boundary-aware matching keeps this diagnostic
        deterministic while removing those common false positives.
        """

        normalized_text = text.casefold()
        normalized_term = term.casefold().strip()
        if not normalized_term:
            return False
        if re.fullmatch(r"[a-z0-9 _-]+", normalized_term):
            phrase = re.escape(normalized_term).replace(r"\ ", r"\s+")
            return re.search(
                rf"(?<![a-z0-9]){phrase}(?![a-z0-9])",
                normalized_text,
            ) is not None
        return normalized_term in normalized_text

    @staticmethod
    def _coverage_is_required(
        coverage_matrix: EvidenceCoverageMatrix,
        *,
        query: str | None = None,
    ) -> bool:
        """Return whether the deterministic four-dimension gate applies to this query.

        goal/data/metrics/scope 是为多对象比较题设计的验收矩阵。事实问答或单对象研究
        不应因为缺少其中某一维而被错误标记为覆盖不完整。
        """

        named_subjects = [
            row for row in coverage_matrix.subjects if row.subject != "overall"
        ]
        if len(named_subjects) < 2:
            return False
        if query is None:
            # 兼容旧调用者；Pipeline 始终传入 query 以避免对所有双实体问题套用方法矩阵。
            return True
        normalized = query.casefold()
        return bool(
            re.search(r"(?:比较|对比|compare|comparison|versus|\bvs\.?\b)", normalized)
        )

    @staticmethod
    def _required_empirical_study_count(query: str) -> int:
        """Extract only explicit minimum-study obligations from Chinese or English queries.

        这里不猜测普通“比较研究”的数量；只处理用户明确写出的
        “至少 N 项实证研究”类硬约束，避免将普通事实题误判为比较题。
        """

        chinese_digits = {
            "一": 1,
            "二": 2,
            "两": 2,
            "三": 3,
            "四": 4,
            "五": 5,
            "六": 6,
            "七": 7,
            "八": 8,
            "九": 9,
            "十": 10,
        }
        chinese = re.search(
            r"至少\s*([1-9]\d*|[一二两三四五六七八九十])\s*(?:项|个|篇)?\s*(?:具名)?(?:的)?(?:实证)?研究",
            query,
        )
        if chinese:
            raw = chinese.group(1)
            return int(raw) if raw.isdigit() else chinese_digits[raw]
        english = re.search(
            r"\bat\s+least\s+([1-9]\d*)\s+(?:named\s+)?(?:empirical\s+)?stud(?:y|ies)\b",
            query,
            flags=re.IGNORECASE,
        )
        return int(english.group(1)) if english else 0

    @staticmethod
    def _named_study_result_identities(report: ResearchReport) -> set[str]:
        """Return identifiable studies that have a result/conclusion statement in the report.

        仅有“A 研究比较了 B”的书目或方法介绍不算完成实证结论覆盖。
        若下一条以“该研究”继续陈述结果，则继承上一条的具名研究标识。
        """

        result_pattern = re.compile(
            r"(发现|报告|表明|显示|得出|观察到|未发现|无显著|"
            r"结果(?:是|为|显示|表明)|"
            r"found|reported|showed|demonstrated|concluded|observed|no\s+evidence)",
            flags=re.IGNORECASE,
        )
        latin_author_pattern = re.compile(
            r"\b([A-Z][A-Za-zÀ-ÖØ-öø-ÿ'\-’]{2,})\s*(?:等人|et\s+al\.?)",
        )
        chinese_author_pattern = re.compile(r"([一-鿿]{2,6})(?:等人|团队)")
        title_pattern = re.compile(r"《([^》]{4,120})》")
        identities: set[str] = set()
        recent_identities: set[str] = set()
        source_identity_map: dict[str, set[str]] = {}
        for raw_finding in report.key_findings:
            citation_numbers = [
                int(number)
                for number in re.findall(r"\[(\d+)\]", raw_finding)
            ]
            cited_source_keys: list[str] = []
            for number in citation_numbers:
                if number < 1 or number > len(report.evidence):
                    continue
                title = report.evidence[number - 1].source_title.casefold()
                # PubMed/PMC 等同一论文的展示后缀不应被计为两项研究。
                normalized_title = re.sub(
                    r"\b(?:pmc|pubmed)\b.*$",
                    "",
                    title,
                )
                normalized_title = " ".join(
                    re.sub(r"[^a-z0-9一-鿿]+", " ", normalized_title).split()
                )
                if normalized_title:
                    cited_source_keys.append(normalized_title)
            finding = re.sub(r"\s*\[\d+\](?:\s*\[\d+\])*\s*$", "", raw_finding)
            explicit = {
                match.casefold()
                for match in latin_author_pattern.findall(finding)
            }
            explicit.update(
                match
                for match in chinese_author_pattern.findall(finding)
                if match not in {"该研究", "研究者", "这项研究"}
            )
            explicit.update(match.casefold() for match in title_pattern.findall(finding))
            if explicit:
                recent_identities = explicit
                for source_key in cited_source_keys:
                    source_identity_map.setdefault(source_key, set()).update(explicit)
            if not result_pattern.search(finding):
                continue
            if explicit:
                identities.update(explicit)
            elif re.match(
                r"^(?:该|这项|上述|其).*?(?:研究|试验|实验)",
                finding,
            ) and recent_identities:
                identities.update(recent_identities)
            elif (
                re.search(r"(?:研究|试验|实验|study|trial|experiment)", finding, re.I)
                and not re.search(r"(?:综述|review|meta-analysis)", finding, re.I)
            ):
                # 结果句未重复作者名时，用其引用的来源标题作为可审计研究标识。
                for source_key in cited_source_keys:
                    identities.update(
                        source_identity_map.get(
                            source_key,
                            {f"source:{source_key}"},
                        )
                    )
        return identities

    @staticmethod
    def _with_report_requirement_findings(
        findings: list[RedTeamFinding],
        *,
        task: ResearchTask,
        report: ResearchReport,
    ) -> list[RedTeamFinding]:
        """Add blocking findings for explicit, machine-checkable report duties."""

        updated = list(findings)
        required = ResearchPipeline._required_empirical_study_count(task.query)
        if required > 0:
            covered = ResearchPipeline._named_study_result_identities(report)
            issue_prefix = "报告未满足明确的研究数量约束："
            if len(covered) < required and not any(
                item.issue.startswith(issue_prefix) for item in updated
            ):
                updated.append(
                    RedTeamFinding(
                        issue=(
                            f"{issue_prefix}用户要求至少比较 {required} 项具名实证研究的结论，"
                            f"但当前报告仅能识别 {len(covered)} 项具有结果性陈述的具名研究。"
                        ),
                        severity="high",
                        suggested_fix=(
                            "补充原始研究证据，并对每项研究明确写出可引用支持的实证结果；"
                            "仅介绍研究对象、作者或方法不计为结论覆盖。"
                        ),
                        claim="报告已比较用户要求数量的具名实证研究结论。",
                        status="unsupported",
                        category="coverage",
                        repair_action="retrieve",
                    )
                )

        if ResearchPipeline._explicit_numeric_result_required(task.query):
            require_primary = ResearchPipeline._primary_numeric_source_required(
                task.query
            )
            supported = ResearchPipeline._numeric_metric_result_evidence_ids(
                report,
                require_primary=require_primary,
            )
            issue_prefix = "报告未满足明确的数值结果约束："
            if not supported and not any(
                item.issue.startswith(issue_prefix) for item in updated
            ):
                source_requirement = "第一手原始来源" if require_primary else "引用证据"
                updated.append(
                    RedTeamFinding(
                        issue=(
                            f"{issue_prefix}用户明确要求具体指标数值或论文报告结果，"
                            f"但当前报告没有由{source_requirement}直接支持的指标数值。"
                        ),
                        severity="high",
                        suggested_fix=(
                            "检索并解析原始论文的结果段或表格，写出带引用的具体数值；"
                            "若仍无法获得，应明确保持未解决状态，不能记为完整回答。"
                        ),
                        claim="报告已提供用户明确要求的具体指标数值或实验结果。",
                        status="unsupported",
                        category="coverage",
                        repair_action="retrieve",
                    )
                )
        _, _, methodology_findings = ResearchPipeline._methodology_quality_findings(
            task,
            report,
        )
        updated.extend(methodology_findings)
        return ResearchPipeline._dedupe_redteam_findings(updated)

    @staticmethod
    def _explicit_numeric_result_required(query: str) -> bool:
        """Recognize only explicit requests for reported numerical metric results."""

        return bool(
            re.search(
                r"(?:数值|数字结果|百分比|具体.{0,80}(?:结果|数值)|"
                r"论文(?:中|所)?报告(?:的)?结果|分别是多少|"
                r"specific.{0,80}(?:values?|results?)|exact.{0,80}(?:values?|results?)|"
                r"reported\s+(?:metric\s+)?results?|quantitative\s+results?)",
                query,
                flags=re.IGNORECASE | re.DOTALL,
            )
        )

    @staticmethod
    def _primary_numeric_source_required(query: str) -> bool:
        return bool(
            re.search(
                r"(?:原始论文|原文|第一手|官方项目|official|original\s+paper|primary\s+source)",
                query,
                flags=re.IGNORECASE,
            )
        )

    @staticmethod
    def _contains_metric_value(text: str) -> bool:
        metric = (
            r"(?:precision|recall|f1(?:[-\s]?score)?|accuracy|"
            r"hallucination\s+(?:rate|density|count)|"
            r"精确率|精准率|召回率|准确率|F1|幻觉率|幻觉密度|幻觉计数)"
        )
        number = r"(?:\d+(?:\.\d+)?\s*%|0\.\d+|\d+\.\d+|\d+(?![\d,]))"
        return bool(
            re.search(
                rf"(?:{metric}.{{0,80}}{number}|{number}.{{0,80}}{metric})",
                text,
                flags=re.IGNORECASE | re.DOTALL,
            )
        )

    @staticmethod
    def _is_primary_research_url(url: str) -> bool:
        host = (urlsplit(url).hostname or "").casefold()
        primary_hosts = (
            "aclanthology.org",
            "arxiv.org",
            "doi.org",
            "jmlr.org",
            "openreview.net",
            "proceedings.mlr.press",
            "proceedings.neurips.cc",
        )
        return host.endswith((".edu", ".gov")) or any(
            host == item or host.endswith(f".{item}") for item in primary_hosts
        )

    @staticmethod
    def _numeric_metric_result_evidence_ids(
        report: ResearchReport,
        *,
        require_primary: bool,
    ) -> set[str]:
        """Return cited Evidence IDs that directly contain a requested metric value."""

        supported: set[str] = set()
        for finding in report.key_findings:
            if not ResearchPipeline._contains_metric_value(finding):
                continue
            for raw_number in re.findall(r"\[(\d+)\]", finding):
                number = int(raw_number)
                if number < 1 or number > len(report.evidence):
                    continue
                evidence = report.evidence[number - 1]
                evidence_text = f"{evidence.claim}\n{evidence.quote}"
                if not ResearchPipeline._contains_metric_value(evidence_text):
                    continue
                if require_primary and not ResearchPipeline._is_primary_research_url(
                    evidence.source_url
                ):
                    continue
                supported.add(evidence.id)
        return supported

    @staticmethod
    def _derive_completion_status(
        *,
        quality_status: str,
        stop_reason: ResearchStopReason,
        coverage_matrix: EvidenceCoverageMatrix,
        evidence_gaps: list[EvidenceGap],
        query: str | None = None,
    ) -> str:
        """Collapse independent quality, budget, coverage and loop signals for consumers."""

        if stop_reason == ResearchStopReason.BUDGET_EXHAUSTED:
            return "budget_limited"
        if quality_status in {"needs_revision", "fail", "not_verified"}:
            return "quality_unresolved"
        if (
            ResearchPipeline._coverage_is_required(coverage_matrix, query=query)
            and not coverage_matrix.complete
        ):
            return "partial_coverage"
        has_open_gaps = any(gap.status == EvidenceGapStatus.OPEN for gap in evidence_gaps)
        if has_open_gaps and stop_reason in {
            ResearchStopReason.MAX_ROUNDS_REACHED,
            ResearchStopReason.NO_PROGRESS,
        }:
            return "partial_research"
        return "complete"

    @staticmethod
    def _identify_coverage_gaps(query: str, evidence: list[Evidence]) -> list[EvidenceGap]:
        """Convert each missing named subject/dimension cell into a focused research task."""

        matrix = ResearchPipeline._build_evidence_coverage_matrix(query, evidence)
        if not ResearchPipeline._coverage_is_required(matrix, query=query):
            return []
        terms_by_dimension = ResearchPipeline._coverage_dimension_terms()
        labels = {
            "goal": "评测目标",
            "data": "数据构成",
            "metrics": "评测指标",
            "scope": "适用范围与局限",
        }
        query_terms = {
            "goal": "objective purpose goal",
            "data": "dataset corpus tasks data",
            "metrics": "evaluation metrics precision recall accuracy",
            "scope": "scope applications use cases limitations",
        }
        gaps: list[EvidenceGap] = []
        evidence_by_id = {item.id: item for item in evidence}
        for row in matrix.subjects:
            # 已有证据标题通常比裸缩写更能消除同名实体歧义，例如 ALCE 可能被搜索引擎
            # 解释成品牌名。只采用包含实体且足够描述性的标题，短标题仍回退到实体名。
            descriptive_titles = [
                evidence_by_id[evidence_id].source_title.strip()
                for evidence_id in row.evidence_ids
                if evidence_id in evidence_by_id
                and row.subject.casefold()
                in evidence_by_id[evidence_id].source_title.casefold()
                and len(evidence_by_id[evidence_id].source_title.split()) >= 4
            ]
            search_anchor = descriptive_titles[0] if descriptive_titles else row.subject
            for dimension in row.missing_dimensions:
                label = labels[dimension]
                gaps.append(
                    EvidenceGap(
                        question=(
                            f"为原研究问题补充 {row.subject} 的{label}，优先使用原始论文、"
                            "官方项目页或作者机构页面，并抽取可直接引用的原文。"
                        ),
                        retrieval_query=(
                            f"{search_anchor} original paper official {query_terms[dimension]}"
                        ),
                        target_entities=[row.subject],
                        resolution_terms=list(terms_by_dimension[dimension]),
                        reason=f"覆盖检查发现 {row.subject} 的{label}尚无可用 Evidence。",
                        source="system",
                        priority="high",
                        related_evidence_ids=list(row.evidence_ids),
                    )
                )
        return gaps

    @staticmethod
    def _named_query_entities(query: str) -> list[str]:
        """Extract acronym/CamelCase/title-like entities without another model call."""

        return extract_named_research_entities(query)

    @staticmethod
    def _with_named_entity_retrieval_queries(
        query: str,
        subtasks: list[SubTask],
    ) -> list[SubTask]:
        """Give each named comparison object one focused initial search when capacity permits."""

        entities = ResearchPipeline._named_query_entities(query)
        research_subtasks = [item for item in subtasks if item.task_type == "research"]
        if len(entities) < 2:
            return subtasks

        covered_entities = {
            entity.casefold()
            for entity in entities
            if any(
                item.retrieval_query
                and entity.casefold() in item.retrieval_query.casefold()
                for item in research_subtasks
            )
        }
        remaining_entities = [
            entity for entity in entities if entity.casefold() not in covered_entities
        ]
        available_subtasks = [item for item in research_subtasks if not item.retrieval_query]
        # 不在这里扩张或重写 DAG；容量不足交给后续 coverage gap 明确补研。
        if len(available_subtasks) < len(remaining_entities):
            return subtasks

        assignments = {
            item.id: entity
            for item, entity in zip(
                available_subtasks,
                remaining_entities,
                strict=False,
            )
        }
        return [
            item.model_copy(
                update={
                    "retrieval_query": ResearchPipeline._entity_focused_retrieval_query(
                        assignments[item.id],
                        item.question,
                    )
                }
            )
            if item.id in assignments
            else item
            for item in subtasks
        ]

    @staticmethod
    def _entity_focused_retrieval_query(entity: str, question: str = "") -> str:
        """Build one focused query and retain a Planner-supplied acronym expansion when present."""

        expansion_match = re.search(
            rf"{re.escape(entity)}\s*[（(]([^）)]{{3,120}})[）)]",
            question,
            flags=re.IGNORECASE,
        )
        expansion = " ".join(expansion_match.group(1).split()) if expansion_match else ""
        prefix = f"{entity} {expansion}".strip()
        return (
            f"{prefix} original paper official benchmark dataset evaluation metrics "
            "scope limitations"
        )

    @staticmethod
    def _contains_missing_evidence_marker(text: str) -> bool:
        """Recognize explicit evidence absence, not merely a negative factual statement."""

        lowered = " ".join(text.casefold().split())
        english_patterns = (
            r"\bno\s+(?:traceable\s+|verifiable\s+|supporting\s+)?evidence\b",
            r"\bmissing\s+(?:traceable\s+|supporting\s+)?evidence\b",
            r"\black(?:s|ing)?\s+(?:traceable\s+|supporting\s+)?evidence\b",
            r"\bwithout\s+(?:traceable\s+|supporting\s+)?evidence\b",
            r"\b(?:entirely\s+|completely\s+)?absent\b",
            r"\bnot\s+(?:currently\s+)?covered\b",
        )
        chinese_markers = (
            "缺少",
            "缺乏",
            "缺失",
            "没有证据",
            "未包含",
            "未提供",
            "未覆盖",
        )
        return any(re.search(pattern, lowered) for pattern in english_patterns) or any(
            marker in text for marker in chinese_markers
        )

    def _compress_evidence_context(
        self,
        evidence: list[Evidence],
        *,
        research_query: str = "",
    ) -> CompressionResult:
        """Apply compression while preserving minimal named-dimension coverage."""

        if self.config.runtime.enable_compression:
            return self.evidence_compressor.compress(
                evidence,
                priority_evidence_ids=self._coverage_priority_evidence_ids(
                    research_query,
                    evidence,
                ),
            )
        total_chars = sum(len(item.claim) + len(item.quote) for item in evidence)
        credibility_counts = {
            tier: sum(1 for item in evidence if item.source_credibility_tier == tier)
            for tier in ("high", "medium", "low")
        }
        return CompressionResult(
            evidence=evidence,
            input_count=len(evidence),
            output_count=len(evidence),
            input_chars=total_chars,
            output_chars=total_chars,
            selected_credibility_counts=credibility_counts,
        )

    @staticmethod
    def _coverage_priority_evidence_ids(
        query: str,
        evidence: list[Evidence],
    ) -> set[str]:
        """Choose a small high-quality Evidence set that covers every currently covered cell."""

        if len(ResearchPipeline._named_query_entities(query)) < 2:
            return set()
        matrix = ResearchPipeline._build_evidence_coverage_matrix(query, evidence)
        uncovered_groups = [
            set(evidence_ids)
            for row in matrix.subjects
            for evidence_ids in row.dimension_evidence_ids.values()
            if evidence_ids
        ]
        by_id = {item.id: item for item in evidence}
        original_order = {item.id: index for index, item in enumerate(evidence)}
        selected: set[str] = set()
        while uncovered_groups:
            candidates = set().union(*uncovered_groups)

            def score(identifier: str) -> tuple[int, float, float, int]:
                item = by_id[identifier]
                credibility = (
                    item.source_credibility_score
                    if item.source_credibility_score is not None
                    else 0.5
                )
                return (
                    sum(identifier in group for group in uncovered_groups),
                    credibility,
                    item.confidence,
                    -original_order[identifier],
                )

            best = max(candidates, key=score)
            selected.add(best)
            uncovered_groups = [group for group in uncovered_groups if best not in group]
        return selected

    def _exhausted_budget_limit(
        self,
        task: ResearchTask,
        offsets: dict[int, int],
        started: float,
        *,
        base_usage: ResearchBudgetUsage | None = None,
    ) -> str | None:
        """Return the first exhausted phase-level budget, or None when work may continue."""

        calls = self._collect_llm_calls(offsets)
        base = base_usage or ResearchBudgetUsage()
        elapsed = base.elapsed_seconds + time.perf_counter() - started
        if task.max_runtime_seconds is not None and elapsed >= task.max_runtime_seconds:
            return "max_runtime_seconds"
        if task.max_llm_calls is not None and base.llm_calls + len(calls) >= task.max_llm_calls:
            return "max_llm_calls"
        total_tokens = base.llm_input_tokens + base.llm_output_tokens + sum(
            (call.input_tokens or 0) + (call.output_tokens or 0) for call in calls
        )
        if task.max_total_tokens is not None and total_tokens >= task.max_total_tokens:
            return "max_total_tokens"
        total_cost = base.cost_usd + sum(call.cost_usd or 0.0 for call in calls)
        if task.max_budget_usd is not None and total_cost >= task.max_budget_usd:
            return "max_budget_usd"
        return None

    def _load_resume_nodes(
        self,
        run_id: str,
    ) -> tuple[list[SubTask], dict[str, ResearchNodeResult], dict[str, NodeTrace]]:
        """Deserialize the persisted graph and only seed fully completed node outputs."""

        if self.memory_store is None:
            return [], {}, {}
        subtasks: list[SubTask] = []
        results: dict[str, ResearchNodeResult] = {}
        traces: dict[str, NodeTrace] = {}
        for row in self.memory_store.load_nodes(run_id):
            subtask = SubTask.model_validate(json.loads(row["subtask_json"]))
            subtasks.append(subtask)
            if not row.get("trace_json") or not row.get("result_json"):
                continue
            trace = NodeTrace.model_validate(json.loads(row["trace_json"]))
            if trace.status != NodeStatus.DONE:
                continue
            result = ResearchNodeResult.model_validate(json.loads(row["result_json"]))
            traces[subtask.id] = trace
            results[subtask.id] = result
        return subtasks, results, traces

    @staticmethod
    def _identify_evidence_gaps(
        subtasks: list[SubTask],
        dag_result: DAGExecutionResult[ResearchNodeResult],
    ) -> list[EvidenceGap]:
        """Convert observable node failures and empty extraction into follow-up questions."""

        trace_by_id = {trace.subtask_id: trace for trace in dag_result.traces}
        gaps: list[EvidenceGap] = []
        failed_statuses = {
            NodeStatus.FAILED,
            NodeStatus.TIMEOUT,
            NodeStatus.SKIPPED,
            NodeStatus.REJECTED,
        }
        for subtask in subtasks:
            if subtask.task_type != "research":
                continue
            trace = trace_by_id.get(subtask.id)
            result = dag_result.results.get(subtask.id)
            if trace is not None and trace.status in failed_statuses:
                reason, source, priority = (
                    "研究节点未成功完成，尚未获得可用证据。",
                    "system",
                    "high",
                )
            elif result is not None and not result.evidence:
                reason, source, priority = (
                    "检索或阅读完成，但没有抽取到可追溯证据。",
                    "reader",
                    "medium",
                )
            else:
                continue
            gaps.append(
                EvidenceGap(
                    question=subtask.question,
                    reason=reason,
                    source=source,
                    priority=priority,
                    related_subtask_ids=[subtask.id],
                )
            )
        return gaps

    @staticmethod
    def _build_round_record(
        round_number: int,
        subtasks: list[SubTask],
        dag_result: DAGExecutionResult[ResearchNodeResult],
        *,
        opened_gap_ids: list[str] | None = None,
        resolved_gap_ids: list[str] | None = None,
        previous_signatures: set[tuple[str, str]] | None = None,
    ) -> ResearchRound:
        """Build one compact round record from a standalone DAG execution."""

        trace_by_id = {trace.subtask_id: trace for trace in dag_result.traces}
        failed_statuses = {
            NodeStatus.FAILED,
            NodeStatus.TIMEOUT,
            NodeStatus.SKIPPED,
            NodeStatus.REJECTED,
        }
        failed_ids = [
            item.id
            for item in subtasks
            if trace_by_id.get(item.id) and trace_by_id[item.id].status in failed_statuses
        ]
        completed_ids = [
            item.id
            for item in subtasks
            if trace_by_id.get(item.id)
            and trace_by_id[item.id].status in {NodeStatus.DONE, NodeStatus.VERIFIED}
        ]
        docs = [doc for result in dag_result.results.values() for doc in result.docs]
        previous = previous_signatures or set()
        new_evidence = [
            item
            for result in dag_result.results.values()
            for item in result.evidence
            if (item.source_url, item.quote.strip()) not in previous
        ]
        started_times = [trace.started_at for trace in dag_result.traces if trace.started_at]
        finished_times = [trace.finished_at for trace in dag_result.traces]
        status = ResearchRoundStatus.PARTIAL if failed_ids else ResearchRoundStatus.DONE
        return ResearchRound(
            round_number=round_number,
            status=status,
            planned_subtask_ids=[item.id for item in subtasks],
            completed_subtask_ids=completed_ids,
            failed_subtask_ids=failed_ids,
            visited_source_ids=list(dict.fromkeys(doc.id for doc in docs)),
            visited_source_urls=list(dict.fromkeys(doc.url for doc in docs if doc.url)),
            new_evidence_ids=[item.id for item in new_evidence],
            opened_gap_ids=opened_gap_ids or [],
            resolved_gap_ids=resolved_gap_ids or [],
            started_at=min(started_times) if started_times else utc_now(),
            finished_at=max(finished_times) if finished_times else utc_now(),
        )

    @staticmethod
    def _evidence_signatures(
        results,
    ) -> set[tuple[str, str]]:
        """Compare cross-round progress by source and exact quote instead of random Evidence IDs."""

        return {
            (item.source_url, item.quote.strip())
            for result in results
            for item in result.evidence
        }

    @staticmethod
    def _source_key(doc: SourceDoc) -> str:
        """Return a stable cross-round source key, including non-HTTP local sources."""

        try:
            return canonicalize_url(doc.url)
        except ValueError:
            # 本地语料和记忆检索可能使用非 HTTP URL；此时退回原 URL，再退回文档 ID。
            return doc.url.strip() or doc.id

    @staticmethod
    def _filter_follow_up_docs_by_entity(
        retrieval_query: str,
        docs: list[SourceDoc],
    ) -> tuple[list[SourceDoc], dict[str, object]]:
        """Reject off-entity follow-up results before they consume Reader calls.

        Initial research remains recall-oriented. Follow-up tasks are created for a known missing
        object, so returning no document is safer than admitting pages for an acronym collision.
        """

        entities = ResearchPipeline._named_query_entities(retrieval_query)
        if not entities:
            return docs, {
                "follow_up_entity_filter_applied": False,
                "follow_up_entity_filter_target": None,
                "entity_mismatched_docs_removed": 0,
            }
        target = entities[0]
        matched = [
            doc
            for doc in docs
            if target.casefold()
            in " ".join((doc.title, doc.content[:1200], doc.url)).casefold()
        ]
        return matched, {
            "follow_up_entity_filter_applied": True,
            "follow_up_entity_filter_target": target,
            "entity_mismatched_docs_removed": len(docs) - len(matched),
        }

    @staticmethod
    def _merge_dag_results(
        target: DAGExecutionResult[ResearchNodeResult],
        addition: DAGExecutionResult[ResearchNodeResult],
    ) -> None:
        """Merge a completed follow-up DAG into the run-wide replay view."""

        target.results.update(addition.results)
        target.traces.extend(addition.traces)
        target.planned_order.extend(addition.planned_order)
        target.started_order.extend(addition.started_order)
        target.completed_order.extend(addition.completed_order)

    def _resolve_llm_client(
        self,
        agent: str,
        profile_name: str | None,
    ) -> LLMClient:
        """Resolve an injected or configured client for one LLM-backed Agent."""

        if agent in self._injected_llm_clients:
            return self._injected_llm_clients[agent]
        if self._default_llm_client is not None:
            return self._default_llm_client

        # None 表示兼容旧配置并使用 llm.active_profile；平面配置使用固定缓存键。
        cache_key = profile_name or self.config.llm.active_profile or "__flat__"
        if cache_key not in self._profile_llm_clients:
            self._profile_llm_clients[cache_key] = create_llm_client(
                self.config.llm,
                profile_name=profile_name,
            )
        return self._profile_llm_clients[cache_key]

    def _bind_memory_scope(self, task: ResearchTask) -> None:
        """Apply task metadata to the optional historical Evidence provider."""

        provider = self.retriever.provider
        if not isinstance(provider, HybridSearchProvider):
            return
        memory_provider = provider.providers.get("memory")
        if isinstance(memory_provider, EvidenceMemorySearchProvider):
            memory_provider.set_scope(
                domain=task.domain,
                task_query=task.query,
                current_run_id=task.run_id,
            )

    def _unique_llm_clients(self) -> list[LLMClient]:
        """Return every active client once, even when Agents share a Profile."""

        candidates = [
            *self._injected_llm_clients.values(),
            *self._profile_llm_clients.values(),
        ]
        if self._default_llm_client is not None:
            candidates.insert(0, self._default_llm_client)

        unique: list[LLMClient] = []
        seen: set[int] = set()
        for client in candidates:
            if id(client) in seen:
                continue
            seen.add(id(client))
            unique.append(client)
        return unique

    def _collect_llm_calls(
        self,
        offsets: dict[int, int],
    ) -> list[LLMCallTrace]:
        """Merge traces from independent clients in chronological order."""

        calls = [
            trace
            for client in self.llm_clients
            for trace in client.trace_collector.since(offsets[id(client)])
        ]
        return sorted(calls, key=lambda trace: trace.started_at)

    @staticmethod
    def _collect_dependency_evidence(
        dependency_results: dict[str, ResearchNodeResult],
    ) -> list[Evidence]:
        """Collect transitive evidence available from direct dependency outputs."""

        evidence: list[Evidence] = []
        for result in dependency_results.values():
            evidence.extend(result.inherited_evidence)
            evidence.extend(result.evidence)
        return ResearchPipeline._dedupe_evidence(evidence)

    @staticmethod
    def _collect_dependency_findings(
        dependency_results: dict[str, ResearchNodeResult],
    ) -> list[str]:
        """Collect findings in dependency declaration order."""

        return [
            finding
            for result in dependency_results.values()
            for finding in result.findings
        ]

    @staticmethod
    def _build_retrieval_question(subtask: SubTask, inherited_findings: list[str]) -> str:
        """Add compact upstream context to a downstream retrieval query."""

        base_question = subtask.retrieval_query or subtask.question
        if not inherited_findings:
            return base_question
        context = "；".join(inherited_findings[:5])
        return f"{base_question}\n上游研究发现：{context}"

    @staticmethod
    def _synthesize_node_findings(
        subtask: SubTask,
        inherited_findings: list[str],
        inherited_evidence: list[Evidence],
    ) -> list[str]:
        """Create deterministic synthesis output until an LLM Synthesizer handles DAG nodes."""

        if inherited_findings:
            return inherited_findings[:8]
        if inherited_evidence:
            return [item.claim for item in inherited_evidence[:8]]
        return [f"{subtask.question}：上游节点未提供可综合的研究结果。"]

    @staticmethod
    def _dedupe_evidence(evidence: list[Evidence]) -> list[Evidence]:
        """Deduplicate evidence by stable evidence id while preserving order."""

        seen: set[str] = set()
        deduped: list[Evidence] = []
        for item in evidence:
            if item.id in seen:
                continue
            seen.add(item.id)
            deduped.append(item)
        return deduped

    @staticmethod
    def _collect_leaf_findings(
        subtasks: list[SubTask],
        results: dict[str, ResearchNodeResult],
    ) -> list[str]:
        """Prefer final leaf-node findings for RedTeam review."""

        dependency_ids = {
            dependency
            for subtask in subtasks
            for dependency in subtask.depends_on
        }
        return [
            finding
            for subtask in subtasks
            if subtask.id not in dependency_ids and subtask.id in results
            for finding in results[subtask.id].findings
        ]

    @staticmethod
    def _build_node_results_replay(
        results: dict[str, ResearchNodeResult],
    ) -> dict[str, dict[str, object]]:
        """Serialize node outputs without duplicating inherited evidence bodies."""

        return {
            node_id: {
                "subtask_id": result.subtask_id,
                "task_type": result.task_type,
                "dependency_ids": result.dependency_ids,
                "inherited_evidence_ids": result.inherited_evidence_ids,
                "docs": result.docs,
                "chunks": result.chunks,
                "evidence": result.evidence,
                "findings": result.findings,
                "diagnostics": result.diagnostics,
            }
            for node_id, result in results.items()
        }

    @staticmethod
    def _with_redteam_findings(
        report: ResearchReport,
        redteam_findings: list,
    ) -> ResearchReport:
        """Attach internal findings while keeping backend review logs out of user Markdown."""

        markdown = ResearchPipeline._remove_markdown_section(
            report.markdown,
            "## RedTeam 检查",
        )
        return report.model_copy(
            update={
                "redteam_findings": redteam_findings,
                "markdown": markdown,
            }
        )

    @staticmethod
    def _finding_substance_findings(
        task: ResearchTask,
        report: ResearchReport,
    ) -> tuple[list[FindingSubstanceDiagnostic], list[RedTeamFinding]]:
        """Turn deterministic malformed findings into blocking internal issues."""

        diagnostics = assess_report_findings(
            task.query,
            report.key_findings,
            report.evidence,
        )
        findings = [
            RedTeamFinding(
                claim=item.text,
                status="unsupported",
                severity="high",
                issue=(
                    "最终关键发现不是完整、可核查的研究结论："
                    f"{item.status}。"
                ),
                reason=item.reason,
                suggested_fix="删除该条，或使用现有证据重写为完整且直接受支持的命题。",
                category="coverage",
                repair_action="remove",
            )
            for item in diagnostics
            if item.status != "substantive"
        ]
        return diagnostics, findings

    @staticmethod
    def _methodology_quality_findings(
        task: ResearchTask,
        report: ResearchReport,
    ) -> tuple[
        list[MethodologyObligationDiagnostic],
        list[SourceAbsenceDiagnostic],
        list[RedTeamFinding],
    ]:
        """Enforce operational definitions and directly evidenced absence claims."""

        finding_inputs = report_finding_inputs(report.key_findings, report.evidence)
        methodology = assess_methodology_obligations(
            task.query,
            finding_inputs,
            report.evidence,
        )
        absence = assess_source_absence_claims(finding_inputs, report.evidence)
        findings: list[RedTeamFinding] = []
        for item in methodology:
            if item.status == "satisfied":
                continue
            evidence_available = item.status == "missing"
            findings.append(
                RedTeamFinding(
                    issue=(
                        "报告缺少方法型问题的可操作定义："
                        f"{item.obligation_type}（{item.status}）。"
                    ),
                    severity="high",
                    suggested_fix=(
                        "使用现有证据在关键发现中补全定义，并保留直接引用。"
                        if evidence_available
                        else "补充能够直接支持该定义前提的证据；证据不足时保持未解决。"
                    ),
                    claim=item.obligation_type,
                    status="unsupported",
                    evidence_ids=item.evidence_ids,
                    reason=item.reason,
                    category="coverage",
                    repair_action="qualify" if evidence_available else "retrieve",
                )
            )
        for item in absence:
            if item.status != "unverified":
                continue
            findings.append(
                RedTeamFinding(
                    issue="关键发现把本次检索未发现的内容写成了来源本身缺失的事实。",
                    severity="high",
                    suggested_fix="删除该缺失性事实，或移入局限性并限定为本次检索范围。",
                    claim=item.text,
                    status="unsupported",
                    evidence_ids=item.evidence_ids,
                    reason=item.reason,
                    category="citation_support",
                    repair_action="remove",
                )
            )
        return methodology, absence, findings

    @staticmethod
    def _actionable_redteam_findings(
        findings: list[RedTeamFinding],
        *,
        coverage_matrix: EvidenceCoverageMatrix | None = None,
    ) -> list[RedTeamFinding]:
        """Return blocking findings, excluding supported claims and optional perspectives."""

        return [
            item
            for item in findings
            if item.status != "supported"
            # Verifier 协议将 none 定义为已支持或仅观察。此类项可保留审计，
            # 但不应在没有修复动作的情况下阻断最终交付。
            and item.repair_action != "none"
            and not ResearchPipeline._is_advisory_finding(
                item,
                coverage_matrix=coverage_matrix,
            )
        ]

    @staticmethod
    def _advisory_redteam_findings(
        findings: list[RedTeamFinding],
        *,
        coverage_matrix: EvidenceCoverageMatrix | None = None,
    ) -> list[RedTeamFinding]:
        """Keep optional low-severity perspectives visible without blocking final acceptance."""

        return [
            item
            for item in findings
            if ResearchPipeline._is_advisory_finding(
                item,
                coverage_matrix=coverage_matrix,
            )
        ]

    @staticmethod
    def _is_advisory_finding(
        finding: RedTeamFinding,
        *,
        coverage_matrix: EvidenceCoverageMatrix | None = None,
    ) -> bool:
        """Recognize optional perspectives without downgrading a wholly absent core subject."""

        is_missing_perspective = (
            finding.severity == "low"
            and finding.category == "coverage"
            and finding.issue.strip().startswith("可能遗漏视角：")
        )
        if not is_missing_perspective:
            return False
        if coverage_matrix is None:
            return True
        issue = finding.issue.casefold()
        for row in coverage_matrix.subjects:
            if row.subject == "overall" or row.subject.casefold() not in issue:
                continue
            # 用户点名的研究对象若四个核心维度全空，属于未回答主问题，不是可选扩展。
            if len(row.missing_dimensions) == len(coverage_matrix.dimensions):
                return False
        return True

    @staticmethod
    def _text_only_revision_findings(
        findings: list[RedTeamFinding],
    ) -> list[RedTeamFinding]:
        """Return the safe subset that can be repaired without adding evidence.

        Citation-support findings are repairable only when Verifier explicitly requests deletion,
        narrowing, or qualification. Narrowing/qualification additionally require an existing
        Evidence reference. Evidence-seeking findings remain outside this compatibility helper.
        """

        repairable, _, _ = ResearchPipeline._route_terminal_revision_findings(
            findings,
            allow_terminal_removal=False,
        )
        return repairable

    @staticmethod
    def _route_terminal_revision_findings(
        findings: list[RedTeamFinding],
        *,
        allow_terminal_removal: bool,
    ) -> tuple[list[RedTeamFinding], list[RedTeamFinding], list[dict[str, str]]]:
        """Split final findings into text-repairable and evidence-deferred groups.

        ``retrieve`` 只有在研究机会已经耗尽、问题指向明确 claim，且 Verifier 的建议
        明确允许删除时，才会降级为 ``remove``。这使无证据的附加句可以被安全删掉，
        但事实、覆盖及时效缺口仍保持阻断状态。
        """

        repairable: list[RedTeamFinding] = []
        deferred: list[RedTeamFinding] = []
        conversions: list[dict[str, str]] = []
        for item in ResearchPipeline._actionable_redteam_findings(findings):
            if item.category in {"logic", "other"}:
                repairable.append(item)
                continue
            if item.category != "citation_support":
                deferred.append(item)
                continue
            if item.repair_action == "remove":
                repairable.append(item)
                continue
            if item.repair_action in {"narrow", "qualify"} and item.evidence_ids:
                repairable.append(item)
                continue
            can_convert_to_remove = (
                allow_terminal_removal
                and item.repair_action == "retrieve"
                and bool((item.claim or "").strip())
                and item.status in {"weakly_supported", "unsupported", "contradicted"}
                and ResearchPipeline._suggests_claim_removal(item.suggested_fix)
            )
            if can_convert_to_remove:
                repairable.append(item.model_copy(update={"repair_action": "remove"}))
                conversions.append(
                    {
                        "claim": item.claim or "",
                        "from_action": "retrieve",
                        "to_action": "remove",
                        "reason": (
                            "research_exhausted_and_verifier_explicitly_allows_removal"
                        ),
                    }
                )
                continue
            deferred.append(item)
        return repairable, deferred, conversions

    @staticmethod
    def _suggests_claim_removal(suggested_fix: str) -> bool:
        """Recognize an explicit deletion option without guessing from generic insufficiency."""

        lowered = suggested_fix.casefold()
        return any(
            marker in lowered
            for marker in (
                "remove",
                "delete",
                "omit",
                "drop this",
                "删除",
                "移除",
                "删去",
                "去掉",
                "省略",
            )
        )

    @staticmethod
    def _dedupe_redteam_findings(
        findings: list[RedTeamFinding],
    ) -> list[RedTeamFinding]:
        """Preserve finding order while avoiding duplicate deferred/reverification issues."""

        seen: set[tuple[str, str, str]] = set()
        result: list[RedTeamFinding] = []
        for item in findings:
            signature = (
                item.category,
                " ".join((item.claim or "").casefold().split()),
                " ".join(item.issue.casefold().split()),
            )
            if signature in seen:
                continue
            seen.add(signature)
            result.append(item)
        return result

    @staticmethod
    def _has_llm_call_capacity(
        call_budget: LLMCallBudget | None,
        *,
        required_calls: int,
    ) -> bool:
        """Check call quota before entering a stage that must reserve revision plus verification."""

        if call_budget is None:
            return True
        return call_budget.max_calls - call_budget.used_calls >= required_calls

    @staticmethod
    def _filter_deterministic_citation_findings(
        findings: list[RedTeamFinding],
        validation: CitationValidation,
    ) -> tuple[list[RedTeamFinding], list[RedTeamFinding]]:
        """Suppress only structural citation claims disproved by the local validator.

        citation_support 属于语义判断，不能在这里过滤；只有明确标记为
        citation_integrity 的编号/证据表意见才由确定性结果裁决。
        """

        if not validation.valid:
            return findings, []
        accepted = [item for item in findings if item.category != "citation_integrity"]
        suppressed = [item for item in findings if item.category == "citation_integrity"]
        return accepted, suppressed

    @staticmethod
    def _effective_review_status(
        trace: RedTeamTrace,
        actionable_findings: list[RedTeamFinding],
        *,
        reviewed_findings: list[RedTeamFinding] | None = None,
        suppressed_findings: list[RedTeamFinding] | None = None,
    ) -> str:
        """Derive the pipeline decision after deterministic checks and supported-item filtering."""

        if actionable_findings:
            # 保留的阻断项与模型误报 pass 冲突时，以结构化 Finding 为准，不能让
            # deferred 证据缺口因为复核摘要遗漏而被错误放行。
            if trace.overall_status in {"needs_revision", "fail"}:
                return trace.overall_status
            return "needs_revision"

        if trace.overall_status not in {"needs_revision", "fail"}:
            return "pass"

        # 合法清空路径包括：低严重度建议项，以及被本地引用校验证伪的结构性意见。
        # 除此之外，Verifier 明确失败却没有可执行项属于协议不一致，必须 fail closed。
        if reviewed_findings is None:
            return trace.overall_status
        unresolved_non_advisory = any(
            item.severity.casefold() not in {"low", "info"}
            and (item.status or "").casefold()
            not in {"supported", "weakly_supported"}
            for item in reviewed_findings
        )
        if unresolved_non_advisory:
            return trace.overall_status
        if not reviewed_findings and not suppressed_findings:
            return trace.overall_status
        return "pass"

    @staticmethod
    def _remove_markdown_section(markdown: str, heading: str) -> str:
        """Remove one internal level-two section while preserving neighboring user sections."""

        lines = markdown.rstrip("\n").splitlines()
        try:
            start = lines.index(heading)
        except ValueError:
            return markdown.rstrip() + "\n"

        end = len(lines)
        for index in range(start + 1, len(lines)):
            if lines[index].startswith("## "):
                end = index
                break
        updated_lines = [*lines[:start], *lines[end:]]
        while updated_lines and not updated_lines[-1].strip():
            updated_lines.pop()
        return "\n".join(updated_lines) + "\n"

    @staticmethod
    def _with_public_coverage_disclosure(
        report: ResearchReport,
        coverage_matrix: EvidenceCoverageMatrix,
    ) -> ResearchReport:
        """State incomplete named-subject coverage without exposing internal matrix terminology."""

        # 模型提示已经禁止暴露内部诊断名；这里再做确定性兜底，避免偶发地把
        # “依据覆盖矩阵/维度标注”写进面向用户的报告。
        def sanitize(text: str) -> str:
            return (
                text.replace("依据覆盖矩阵", "根据现有材料")
                .replace("覆盖矩阵", "现有材料")
                .replace("维度标注为缺失", "相关信息仍不充分")
                .replace("MISSING cells", "unsupported aspects")
                .replace("MISSING", "unsupported")
            )

        report = report.model_copy(
            update={
                "summary": sanitize(report.summary),
                "key_findings": [sanitize(item) for item in report.key_findings],
                "limitations": [sanitize(item) for item in report.limitations],
                "markdown": sanitize(report.markdown),
            }
        )
        if not ResearchPipeline._coverage_is_required(
            coverage_matrix,
            query=report.query,
        ):
            markdown = report.markdown
            for heading in ("## 回答范围", "## Answer Scope"):
                markdown = ResearchPipeline._remove_markdown_section(markdown, heading)
            return report.model_copy(update={"markdown": markdown})

        rows = [
            row
            for row in coverage_matrix.subjects
            if row.subject != "overall" and row.missing_dimensions
        ]
        # ResearchReport 只保留 query；根据现有 Markdown 标题判断渲染语言，避免更改公共 schema。
        english = "## Evidence" in report.markdown or "## Summary" in report.markdown
        markdown = report.markdown
        for heading in ("## 回答范围", "## Answer Scope"):
            markdown = ResearchPipeline._remove_markdown_section(markdown, heading)
        if not rows:
            return report.model_copy(update={"markdown": markdown})

        labels = (
            {
                "goal": "evaluation objective",
                "data": "data",
                "metrics": "metrics",
                "scope": "scope",
            }
            if english
            else {
                "goal": "评测目标",
                "data": "数据",
                "metrics": "指标",
                "scope": "适用范围",
            }
        )
        disclosure = ["## Answer Scope" if english else "## 回答范围"]
        for row in rows:
            missing = [labels.get(item, item) for item in row.missing_dimensions]
            covered = [
                labels.get(item, item)
                for item in coverage_matrix.dimensions
                if item not in row.missing_dimensions
            ]
            if not covered:
                if english:
                    disclosure.append(
                        f"- **{row.subject}**: this search did not retrieve enough "
                        f"verifiable material to describe {', '.join(missing)} reliably."
                    )
                else:
                    disclosure.append(
                        f"- **{row.subject}**：本次检索未取得足以核验的来源，"
                        f"因此无法可靠说明其{'、'.join(missing)}。"
                    )
            else:
                if english:
                    disclosure.append(
                        f"- **{row.subject}**: available sources support "
                        f"{', '.join(covered)}; {', '.join(missing)} still lack sufficient material."
                    )
                else:
                    disclosure.append(
                        f"- **{row.subject}**：现有来源可支持{'、'.join(covered)}；"
                        f"{'、'.join(missing)}仍缺少充分材料。"
                    )

        lines = markdown.rstrip().splitlines()
        try:
            insert_at = lines.index("## Evidence" if english else "## 证据表")
        except ValueError:
            insert_at = len(lines)
        updated = [
            *lines[:insert_at],
            "",
            *disclosure,
            "",
            *lines[insert_at:],
        ]
        return report.model_copy(update={"markdown": "\n".join(updated).strip() + "\n"})

    def _build_quality_diagnostics(
        self,
        results: dict[str, ResearchNodeResult],
        all_evidence: list[Evidence],
        report: ResearchReport,
        llm_calls: list[LLMCallTrace],
        compression_result: CompressionResult,
    ) -> dict[str, object]:
        """Summarize evidence coverage and Reader/Retriever behavior for replay analysis."""

        report_evidence_ids = {item.id for item in report.evidence}
        reader_calls = [trace for trace in llm_calls if trace.agent == "reader"]
        per_subtask: dict[str, dict[str, object]] = {}
        safety_level_counts = {"safe": 0, "suspicious": 0, "high": 0}
        safety_categories: set[str] = set()
        retained_source_ids: set[str] = set()
        excluded_source_ids: set[str] = set()
        for subtask_id, result in results.items():
            if result.task_type != "research":
                continue
            reader_diag = result.diagnostics.get("reader", {})
            retrieval_diag = result.diagnostics.get("retrieval", {})
            safety_diag = (
                retrieval_diag.get("content_safety", {})
                if isinstance(retrieval_diag, dict)
                else {}
            )
            if isinstance(safety_diag, dict):
                raw_counts = safety_diag.get("risk_level_counts", {})
                if isinstance(raw_counts, dict):
                    for level in safety_level_counts:
                        safety_level_counts[level] += int(raw_counts.get(level, 0))
                safety_categories.update(
                    str(item) for item in safety_diag.get("signal_categories", [])
                )
                retained_source_ids.update(
                    str(item) for item in safety_diag.get("retained_source_ids", [])
                )
                excluded_source_ids.update(
                    str(item) for item in safety_diag.get("excluded_source_ids", [])
                )
            per_subtask[subtask_id] = {
                "task_type": result.task_type,
                "retrieved_docs_count": len(result.docs),
                "read_docs_count": reader_diag.get("read_docs_count", len(result.docs))
                if isinstance(reader_diag, dict)
                else len(result.docs),
                "evidence_count": len(result.evidence),
                "unique_source_count": len({item.source_id for item in result.evidence}),
                "used_in_report_count": sum(1 for item in result.evidence if item.id in report_evidence_ids),
                "fallback_used": bool(reader_diag.get("fallback_used", False))
                if isinstance(reader_diag, dict)
                else False,
                "llm_failure_fallback_used": bool(
                    reader_diag.get("llm_failure_fallback_used", False)
                )
                if isinstance(reader_diag, dict)
                else False,
                "empty_result_rule_assist_used": bool(
                    reader_diag.get("empty_result_rule_assist_used", False)
                )
                if isinstance(reader_diag, dict)
                else False,
                "retrieved_source_ids": retrieval_diag.get("retrieved_source_ids", [])
                if isinstance(retrieval_diag, dict)
                else [],
                "content_safety": safety_diag,
            }

        return {
            "evidence_quality": {
                "total_evidence_count": len(all_evidence),
                "report_evidence_count": len(report.evidence),
                "unique_source_count": len({item.source_id for item in all_evidence}),
                "unique_quote_count": len({f"{item.source_id}:{item.quote}" for item in all_evidence}),
                "extraction_method_counts": ResearchPipeline._count_evidence_methods(all_evidence),
                "report_extraction_method_counts": ResearchPipeline._count_evidence_methods(report.evidence),
                "unused_evidence_ids": [
                    item.id for item in all_evidence if item.id not in report_evidence_ids
                ],
                "subtasks_without_evidence": [
                    subtask_id
                    for subtask_id, summary in per_subtask.items()
                    if summary["evidence_count"] == 0
                ],
            },
            "reader_retriever": {
                "per_subtask": per_subtask,
                "llm_failure_fallback_subtask_count": sum(
                    bool(summary["llm_failure_fallback_used"])
                    for summary in per_subtask.values()
                ),
                "empty_result_rule_assist_subtask_count": sum(
                    bool(summary["empty_result_rule_assist_used"])
                    for summary in per_subtask.values()
                ),
                "reader_llm_call_count": len(reader_calls),
                "reader_failed_call_count": sum(1 for trace in reader_calls if trace.status == "FAILED"),
                "reader_retry_call_count": sum(1 for trace in reader_calls if trace.attempt_count > 1),
                "reader_total_attempt_count": sum(trace.attempt_count for trace in reader_calls),
                "reader_output_limit_count": sum(
                    1 for trace in reader_calls if trace.finish_reason == "length"
                ),
            },
            "content_safety": {
                "enabled": self.config.content_safety.enabled,
                "mode": self.config.content_safety.mode,
                "risk_level_counts": safety_level_counts,
                "signal_categories": sorted(safety_categories),
                "retained_source_ids": sorted(retained_source_ids),
                "excluded_source_ids": sorted(excluded_source_ids),
            },
            "context_compression": compression_result.model_dump(mode="json", exclude={"evidence"}),
        }

    @staticmethod
    def _count_evidence_methods(evidence: list[Evidence]) -> dict[str, int]:
        """Count evidence by extraction method for quality and fallback analysis."""

        counts: dict[str, int] = {}
        for item in evidence:
            counts[item.extraction_method] = counts.get(item.extraction_method, 0) + 1
        return counts

    def _safe_error(self, exc: Exception) -> str:
        """Redact configured API keys before persisting a terminal run error."""

        message = f"{type(exc).__name__}: {exc}"
        secret_env_names = {self.config.retrieval.api_key_env}
        for profile in self.config.llm.profiles.values():
            secret_env_names.add(profile.api_key_env)
        secret_env_names.add(self.config.llm.api_key_env)
        for env_name in secret_env_names:
            secret = os.environ.get(env_name)
            if secret:
                message = message.replace(secret, "[REDACTED]")
        return message

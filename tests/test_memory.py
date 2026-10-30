from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from deepresearch.config import AppConfig, MemoryConfig, RetrievalConfig, RuntimeConfig
from deepresearch.memory import (
    EvidenceCompressor,
    EvidenceVectorIndex,
    OpenAICompatibleEmbeddingProvider,
    SQLiteMemoryStore,
)
from deepresearch.orchestrator import ResearchPipeline
from deepresearch.retrieval.hybrid import HybridSearchProvider
from deepresearch.retrieval.memory import EvidenceMemorySearchProvider
from deepresearch.schema import (
    Evidence,
    EvidenceGap,
    ResearchBudgetUsage,
    ResearchRound,
    ResearchRoundStatus,
    ResearchRunStatus,
    ResearchState,
    ResearchStopReason,
    ResearchTask,
)
from deepresearch.tools import ApprovalStatus, ToolCallRecord


def _evidence(
    identifier: str,
    source_id: str,
    confidence: float,
    *,
    quote: str | None = None,
) -> Evidence:
    return Evidence(
        id=identifier,
        claim=f"claim {identifier}",
        quote=quote or f"quote {identifier}",
        source_id=source_id,
        source_title=f"source {source_id}",
        source_url=f"https://{source_id}.example",
        confidence=confidence,
        extraction_method="llm",
    )


def test_evidence_compressor_preserves_original_objects_and_source_diversity():
    evidence = [
        _evidence("a1", "a", 0.99),
        _evidence("a2", "a", 0.98),
        _evidence("b1", "b", 0.80),
    ]

    result = EvidenceCompressor(max_evidence=2, max_chars=1000).compress(evidence)

    assert [item.id for item in result.evidence] == ["a1", "b1"]
    assert result.evidence[0] is evidence[0]
    assert result.dropped_evidence_ids == ["a2"]
    assert result.truncated is True


def test_evidence_compressor_uses_source_credibility_in_quality_ranking():
    weak_source = _evidence("weak", "weak_source", 0.90).model_copy(
        update={"source_credibility_score": 0.20, "source_credibility_tier": "low"}
    )
    strong_source = _evidence("strong", "strong_source", 0.82).model_copy(
        update={"source_credibility_score": 0.90, "source_credibility_tier": "high"}
    )

    result = EvidenceCompressor(max_evidence=1, max_chars=1000).compress(
        [weak_source, strong_source]
    )

    assert [item.id for item in result.evidence] == ["strong"]
    assert result.selected_credibility_counts == {"high": 1, "medium": 0, "low": 0}
    assert result.dropped_low_credibility_ids == ["weak"]


def test_evidence_compressor_keeps_priority_coverage_before_higher_scored_duplicates():
    ordinary = _evidence("ordinary", "shared", 0.99)
    required = _evidence("required", "shared", 0.70)
    other_source = _evidence("other", "other", 0.95)

    result = EvidenceCompressor(max_evidence=2, max_chars=1000).compress(
        [ordinary, required, other_source],
        priority_evidence_ids={"required"},
    )

    assert [item.id for item in result.evidence] == ["required", "other"]
    assert result.dropped_evidence_ids == ["ordinary"]


def test_sqlite_memory_store_round_trips_run_and_evidence(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite3")
    task = ResearchTask(query="test memory", run_id="memory_run")
    evidence = [_evidence("e1", "source", 0.9)]

    store.start_run(task)
    store.save_evidence(task.run_id, evidence)
    store.finish_run(task.run_id, replay_path="runs/replay.json")

    run = store.get_run(task.run_id)
    assert run is not None
    assert run["status"] == "DONE"
    assert run["schema_version"] == "1.0"
    assert store.load_evidence(task.run_id) == evidence


def test_sqlite_memory_store_round_trips_unified_research_state(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite3")
    task = ResearchTask(query="test state", run_id="state_run")
    gap = EvidenceGap(
        question="还缺少什么证据？",
        reason="当前来源没有回答该问题。",
        source="reader",
        related_subtask_ids=["task_1"],
    )
    state = ResearchState(
        run_id=task.run_id,
        query=task.query,
        status=ResearchRunStatus.DONE,
        current_round=1,
        rounds=[
            ResearchRound(
                round_number=1,
                status=ResearchRoundStatus.DONE,
                planned_subtask_ids=["task_1"],
                completed_subtask_ids=["task_1"],
                opened_gap_ids=[gap.id],
            )
        ],
        pending_questions=[gap.question],
        evidence_gaps=[gap],
        budget=ResearchBudgetUsage(max_rounds=2, evidence_count=1),
        stop_reason=ResearchStopReason.SINGLE_PASS_COMPLETE,
    )

    store.start_run(task)
    store.save_research_state(state)

    loaded = store.load_research_state(task.run_id)
    assert loaded == state
    assert loaded is not None
    assert loaded.evidence_gaps[0].related_subtask_ids == ["task_1"]


def test_sqlite_memory_store_persists_idempotent_approval_lifecycle(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite3")
    request = store.request_approval(
        run_id="approval_run",
        agent="writer",
        tool_name="publish_report",
        action_signature="stable_action_hash",
        argument_keys=["report_id"],
        risk_level="high_risk",
        reason="tool_requires_operator_approval",
        expires_in_seconds=60,
    )
    duplicate = store.request_approval(
        run_id="approval_run",
        agent="writer",
        tool_name="publish_report",
        action_signature="stable_action_hash",
        argument_keys=["report_id"],
        risk_level="high_risk",
        reason="tool_requires_operator_approval",
        expires_in_seconds=60,
    )

    assert duplicate.approval_id == request.approval_id
    assert request.status == ApprovalStatus.PENDING
    decided = store.decide_approval(
        request.approval_id,
        decision=ApprovalStatus.APPROVED,
        decision_key="decision_key_1",
    )
    repeated = store.decide_approval(
        request.approval_id,
        decision=ApprovalStatus.APPROVED,
        decision_key="decision_key_1",
    )
    assert repeated == decided

    claimed = store.claim_approval_execution(request.approval_id)
    assert claimed is not None
    assert claimed.status == ApprovalStatus.EXECUTING
    assert store.claim_approval_execution(request.approval_id) is None

    executed = store.mark_approval_executed(request.approval_id, call_id="tool_call_1")
    assert executed.status == ApprovalStatus.EXECUTED
    assert store.get_approval(request.approval_id) == executed
    assert store.list_approvals(run_id="approval_run") == [executed]


def test_sqlite_memory_store_expires_pending_approval_fail_closed(tmp_path):
    database = tmp_path / "memory.sqlite3"
    store = SQLiteMemoryStore(database)
    request = store.request_approval(
        run_id="approval_run",
        agent="writer",
        tool_name="publish_report",
        action_signature="expiring_action",
        argument_keys=[],
        risk_level="read_only",
        reason="tool_requires_operator_approval",
        expires_in_seconds=0,
    )

    # A new store instance simulates process restart; expiration remains durable.
    reopened = SQLiteMemoryStore(database)
    expired = reopened.get_approval(request.approval_id)
    assert expired is not None
    assert expired.status == ApprovalStatus.EXPIRED
    try:
        reopened.decide_approval(
            request.approval_id,
            decision=ApprovalStatus.APPROVED,
            decision_key="too_late",
        )
    except ValueError as exc:
        assert "EXPIRED" in str(exc)
    else:  # pragma: no cover - explicit assertion message is clearer than pytest.raises here.
        raise AssertionError("expired approval must not become approved")


def test_sqlite_memory_store_round_trips_compact_tool_records(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite3")
    record = ToolCallRecord(
        call_id="tool_call_1",
        run_id="tool_run",
        agent="researcher",
        tool_name="search_sources",
        risk_level="read_only",
        status="DONE",
        action_signature="stable_signature",
        argument_keys=["query", "top_k"],
        result_summary={"source_ids": ["source_1"]},
    )

    store.save_tool_call_record(record)

    assert store.find_completed_tool_call(
        run_id="tool_run",
        action_signature="stable_signature",
    ) == record
    assert store.list_tool_call_records(run_id="tool_run") == [record]


def test_numpy_vector_index_retrieves_relevant_evidence():
    relevant = _evidence("citation", "source_a", 0.9, quote="引用质量需要检查引用覆盖率。")
    unrelated = _evidence("latency", "source_b", 0.9, quote="并发执行可以降低运行耗时。")
    index = EvidenceVectorIndex(dimensions=256)
    index.build([unrelated, relevant])

    matches = index.search("如何检查引用质量和引用覆盖率？", top_k=1)

    assert matches[0].evidence.id == "citation"
    assert matches[0].score > 0


def test_openai_compatible_embedding_provider_normalizes_and_orders_vectors():
    class FakeEmbeddings:
        def __init__(self):
            self.request = None

        def create(self, **kwargs):
            self.request = kwargs
            return SimpleNamespace(
                data=[
                    SimpleNamespace(index=1, embedding=[0.0, 3.0, 4.0]),
                    SimpleNamespace(index=0, embedding=[2.0, 0.0, 0.0]),
                ]
            )

    embeddings = FakeEmbeddings()
    client = SimpleNamespace(embeddings=embeddings)
    provider = OpenAICompatibleEmbeddingProvider(
        model="embedding-model",
        dimensions=3,
        api_key_env="UNUSED_IN_INJECTED_TEST",
        sdk_client=client,
    )

    vectors = provider.embed(["first", "second"])

    assert embeddings.request == {
        "model": "embedding-model",
        "input": ["first", "second"],
        "dimensions": 3,
    }
    np.testing.assert_allclose(vectors[0], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(vectors[1], [0.0, 0.6, 0.8])


def test_memory_search_provider_preserves_originating_run(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite3")
    task = ResearchTask(query="prior research", run_id="prior_run")
    store.start_run(task)
    store.save_evidence(
        task.run_id,
        [_evidence("citation", "source_a", 0.9, quote="引用质量需要检查引用覆盖率。")],
    )
    store.finish_run(task.run_id, replay_path="prior.json")
    provider = EvidenceMemorySearchProvider(
        store,
        dimensions=256,
        min_similarity=0.01,
        scope_min_query_overlap=0.0,
    )

    docs, diagnostics = provider.search_with_diagnostics("引用覆盖率", top_k=2)

    assert docs[0].metadata["originating_run_id"] == "prior_run"
    assert docs[0].metadata["evidence_id"] == "citation"
    assert diagnostics["index_type"] == "numpy_hashing"
    assert diagnostics["indexed_evidence_count"] == 1
    assert diagnostics["embedding_cache_misses"] == 1

    _, cached_diagnostics = provider.search_with_diagnostics("引用覆盖率", top_k=2)
    assert cached_diagnostics["embedding_cache_hits"] == 1
    assert cached_diagnostics["embedding_cache_misses"] == 0
    assert cached_diagnostics["embedding_model"] == "mixed-ngram-sha256"


def test_memory_search_scope_filters_unrelated_domains(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite3")
    for run_id, domain, evidence_id in (
        ("run_a", "evaluation", "evaluation_evidence"),
        ("run_b", "security", "security_evidence"),
    ):
        task = ResearchTask(query="引用质量研究", run_id=run_id, domain=domain)
        store.start_run(task)
        store.save_evidence(
            run_id,
            [_evidence(evidence_id, run_id, 0.9, quote="引用质量需要证据支持。")],
        )
        store.finish_run(run_id, replay_path=f"{run_id}.json")
    provider = EvidenceMemorySearchProvider(
        store,
        dimensions=256,
        min_similarity=0.01,
        scope_by_domain=True,
        scope_min_query_overlap=0.01,
    )
    provider.set_scope(
        domain="evaluation",
        task_query="如何评测引用质量？",
        current_run_id="current_run",
    )

    docs, diagnostics = provider.search_with_diagnostics("引用质量", top_k=5)

    assert len(docs) == 1
    assert docs[0].metadata["originating_run_id"] == "run_a"
    assert diagnostics["scope_domain"] == "evaluation"
    assert diagnostics["indexed_evidence_count"] == 1


def test_pipeline_adds_historical_evidence_provider_to_hybrid(tmp_path):
    database = tmp_path / "state.sqlite3"
    store = SQLiteMemoryStore(database)
    prior_task = ResearchTask(query="prior", run_id="prior")
    store.start_run(prior_task)
    store.save_evidence(
        prior_task.run_id,
        [_evidence("historical", "source", 0.9, quote="历史证据支持共享记忆。")],
    )
    store.finish_run(prior_task.run_id, replay_path="prior.json")
    config = AppConfig(
        retrieval=RetrievalConfig(
            provider="hybrid",
            hybrid_providers=["local"],
            corpus_dir="data/local_corpus",
        ),
        runtime=RuntimeConfig(run_dir=tmp_path / "runs"),
        memory=MemoryConfig(
            enabled=True,
            sqlite_path=database,
            retrieval_enabled=True,
            min_similarity=0.01,
            scope_min_query_overlap=0.0,
        ),
    )

    pipeline = ResearchPipeline(config)

    assert isinstance(pipeline.retriever.provider, HybridSearchProvider)
    assert "memory" in pipeline.retriever.provider.providers
    _, diagnostics = pipeline.retriever.provider.search_with_diagnostics("共享记忆", top_k=3)
    assert diagnostics["provider_diagnostics"]["memory"]["indexed_evidence_count"] == 1


def test_pipeline_persists_state_and_records_compression(tmp_path):
    database = tmp_path / "state.sqlite3"
    config = AppConfig(
        retrieval=RetrievalConfig(corpus_dir="data/local_corpus", top_k=2),
        runtime=RuntimeConfig(
            run_dir=tmp_path / "runs",
            max_subtasks=1,
            enable_redteam=False,
            enable_compression=True,
        ),
        memory=MemoryConfig(
            enabled=True,
            sqlite_path=database,
            compression_max_evidence=1,
            compression_max_chars=10_000,
        ),
    )
    task = ResearchTask(query="如何评测引用质量和幻觉率？")

    report = ResearchPipeline(config).run(task)

    replay_path = next((tmp_path / "runs").glob(f"*_{task.run_id}_replay.json"))
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    assert replay["synthesis_context"]["output_count"] == 1
    assert len(replay["synthesis_context"]["evidence"]) == 1
    assert replay["diagnostics"]["context_compression"]["output_count"] == 1
    assert len(report.evidence) == 1

    store = SQLiteMemoryStore(database)
    run = store.get_run(task.run_id)
    assert run is not None and run["status"] == "DONE"
    assert len(store.load_nodes(task.run_id)) == 1
    assert len(store.load_evidence(task.run_id)) == len(replay["evidence"])
    state = store.load_research_state(task.run_id)
    assert state is not None
    assert state.status == ResearchRunStatus.DONE
    assert state.current_round == 1
    # A one-item compressed context cannot establish all citation/hallucination
    # methodology obligations, so persistence must retain a reliable-failure state.
    assert state.stop_reason == ResearchStopReason.QUALITY_UNRESOLVED
    assert state.evidence_ids == [item["id"] for item in replay["evidence"]]
    assert replay["research_state"] == state.model_dump(mode="json")

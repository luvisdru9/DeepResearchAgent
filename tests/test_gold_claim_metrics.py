import pytest

from deepresearch.evaluation.benchmark import AtomicClaim
from deepresearch.evaluation.claims import (
    assess_gold_claim_coverage,
    weighted_gold_claim_recall,
)
from deepresearch.evaluation.metrics import score_report
from deepresearch.schema import Evidence, ResearchReport


def _report() -> ResearchReport:
    evidence = Evidence(
        id="ev_1",
        claim="DAG 支持独立任务并发执行。",
        quote="共享记忆保存跨 Agent 的状态和证据。",
        source_id="doc_1",
        source_title="Source",
        source_url="https://example.com/source",
    )
    return ResearchReport(
        run_id="gold_claim_run",
        query="为什么需要 DAG 和共享记忆？",
        title="Report",
        summary="summary",
        key_findings=[
            "DAG 可以并发执行彼此独立的研究节点。 [1]",
            "共享记忆保存跨 Agent 的状态与证据。 [1]",
        ],
        evidence=[evidence],
        markdown=(
            "# Report\n\n## 关键发现\n"
            "- DAG 可以并发执行彼此独立的研究节点。 [1]\n"
            "- 共享记忆保存跨 Agent 的状态与证据。 [1]\n\n"
            "## 证据表\n1. evidence\n"
        ),
    )


def test_weighted_gold_claim_recall_preserves_ids_weights_and_variants():
    gold = [
        AtomicClaim(
            id="c1",
            text="DAG 支持无依赖节点并行调度",
            accepted_variants=["DAG 可以并发执行彼此独立的研究节点"],
            weight=2.0,
        ),
        AtomicClaim(
            id="c2",
            text="共享记忆保存跨 Agent 状态与证据",
            weight=1.0,
        ),
        AtomicClaim(
            id="c3",
            text="系统必须解决错误记忆污染",
            weight=1.0,
        ),
    ]

    assessments = assess_gold_claim_coverage(_report(), gold)

    assert [item.gold_claim_id for item in assessments] == ["c1", "c2", "c3"]
    assert [item.status for item in assessments] == [
        "covered",
        "covered",
        "not_covered",
    ]
    assert weighted_gold_claim_recall(assessments) == pytest.approx(0.75)


def test_v2_gold_claims_replace_legacy_exact_string_fact_score():
    gold = [
        AtomicClaim(
            id="c1",
            text="任务图表达依赖",
            accepted_variants=["DAG 可以并发执行彼此独立的研究节点"],
            weight=1.0,
        )
    ]

    result = score_report(
        "sample",
        _report(),
        ["这一整句不会原样出现在报告中"],
        gold_claims=gold,
    )

    assert result.metadata["legacy_fact_accuracy"] == 0.0
    assert result.fact_accuracy == 1.0
    assert result.weighted_claim_recall == 1.0
    assert result.metadata["gold_claim_method"] == "lexical_gold_claim_v1"

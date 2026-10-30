import pytest
from pydantic import ValidationError

from deepresearch.agents.evaluator import LLMEvaluatorAgent, LLMJudgeOutput
from deepresearch.evaluation.benchmark import AtomicClaim, AtomicClaimSatisfactionRoute
from deepresearch.evaluation.claims import extract_atomic_claims
from deepresearch.evaluation.metrics import score_report
from deepresearch.evaluation.service import (
    _consensus_diagnostics,
    aggregate_judge_results,
    summarize_automatic_consensus,
)
from deepresearch.llm import MockLLMClient
from deepresearch.judge_protocol import judge_protocol_fingerprint, judge_system_prompt
from deepresearch.schema import EvalResult, Evidence, ResearchReport, ResearchTask


def _report() -> ResearchReport:
    evidence = Evidence(
        id="ev_1",
        claim="引用质量需要检查证据支持关系。",
        quote="引用是否真正支持对应结论决定引用精度。",
        source_id="source_1",
        source_title="Citation Guide",
        source_url="https://example.com/citation",
    )
    return ResearchReport(
        run_id="judge_run",
        query="如何评测引用质量？",
        title="Report",
        summary="summary",
        key_findings=["引用精度需要检查证据支持关系。 [1]"],
        evidence=[evidence],
        markdown=(
            "# Report\n\n## 关键发现\n- 引用精度需要检查证据支持关系。 [1]\n\n"
            "## 证据表\n1. 引用是否真正支持对应结论决定引用精度。\n"
        ),
    )


def test_rule_metrics_check_claim_evidence_overlap():
    report = _report().model_copy(
        update={"key_findings": ["月球完全由奶酪构成。 [1]"]}
    )

    result = score_report("sample", report, [], 0.1)

    assert result.citation_coverage == 1.0
    assert result.citation_precision == 0.0
    assert result.hallucination_rate == 1.0
    assert result.metadata["evaluator"] == "rule_claim_support"
    assert result.citation_existence == 1.0
    assert result.citation_correctness == 0.0
    assert result.citation_completeness == 0.0
    assert result.metadata["claim_status_counts"]["unsupported"] == 1


def test_judge_protocol_v5_preserves_history_and_requires_explicit_routes():
    assert judge_protocol_fingerprint("1.0") == (
        "sha256:f7eb95425dfb573b6aa4a3c91fd543cce766d5bd6cb2096baa5a1d50486631ce"
    )
    assert judge_protocol_fingerprint("2.0") != judge_protocol_fingerprint("1.0")
    assert "paper title" not in judge_system_prompt("1.0")
    assert "paper title" in judge_system_prompt("2.0")
    assert "might contain it" in judge_system_prompt("2.0")
    assert judge_protocol_fingerprint("3.0") not in {
        judge_protocol_fingerprint("1.0"),
        judge_protocol_fingerprint("2.0"),
    }
    assert "mapped quote omits it" in judge_system_prompt("3.0")
    assert "atomic-claim analysis" in judge_system_prompt("3.0")
    assert judge_protocol_fingerprint("4.0") not in {
        judge_protocol_fingerprint("1.0"),
        judge_protocol_fingerprint("2.0"),
        judge_protocol_fingerprint("3.0"),
    }
    assert "alternative satisfaction routes" in judge_system_prompt("4.0")
    assert "complete semantic requirement" in judge_system_prompt("4.0")
    assert judge_protocol_fingerprint("5.0") not in {
        judge_protocol_fingerprint(version) for version in ("1.0", "2.0", "3.0", "4.0")
    }
    assert "wording aliases" in judge_system_prompt("5.0")
    assert "matched_route_id" in judge_system_prompt("5.0")


def test_llm_evaluator_returns_five_dimension_grounded_metrics():
    client = MockLLMClient(
        responses=[
            """
            {
              "factuality": 5,
              "completeness": 4,
              "citation_quality": 5,
              "logical_structure": 4,
              "writing_quality": 4,
              "overall_reason": "The claim is directly supported.",
              "claim_assessments": [
                {
                  "claim": "引用精度需要检查证据支持关系。",
                  "status": "supported",
                  "evidence_ids": ["ev_1"],
                  "reason": "The quote states the same criterion."
                }
              ]
            }
            """
        ]
    )
    evaluator = LLMEvaluatorAgent(client)
    task = ResearchTask(query="如何评测引用质量？", run_id="judge_run")

    result = evaluator.evaluate("sample", task, _report(), ["引用精度"], 0.2)

    assert result.fact_accuracy == 1.0
    assert result.answer_completeness == 0.8
    assert result.citation_precision == 1.0
    assert result.hallucination_rate == 0.0
    assert result.metadata["evaluator"] == "llm_judge"
    assert result.metadata["judge_scores"]["citation_quality"] == 5
    assert result.metadata["judge_call_resource"]["status"] == "DONE"
    assert result.metadata["judge_call_resource"]["attempt_count"] == 1


def test_llm_judge_rejects_score_without_claim_assessment():
    """总分不能替代逐条事实核验，否则引用精度会被错误计算为零。"""

    with pytest.raises(ValidationError):
        LLMJudgeOutput.model_validate(
            {
                "factuality": 5,
                "completeness": 5,
                "citation_quality": 5,
                "logical_structure": 5,
                "writing_quality": 5,
                "overall_reason": "Only aggregate scores were returned.",
                "claim_assessments": [],
            }
        )


def test_llm_evaluator_falls_back_when_claim_assessments_are_empty():
    """Judge 连续返回不可核验总分时，评测应降级而不是产出伪精确指标。"""

    client = MockLLMClient(
        max_retries=0,
        responses=[
            """
            {
              "factuality": 5,
              "completeness": 5,
              "citation_quality": 5,
              "logical_structure": 5,
              "writing_quality": 5,
              "overall_reason": "No claim-level details.",
              "claim_assessments": []
            }
            """
        ],
    )
    evaluator = LLMEvaluatorAgent(client)
    task = ResearchTask(query="如何评测引用质量？", run_id="judge_run")

    result = evaluator.evaluate("sample", task, _report(), ["引用精度"], 0.2)

    assert result.metadata["evaluator"] == "rule_fallback"
    assert "LLMResponseValidationError" in result.metadata["judge_error"]
    assert result.citation_precision == 1.0
    assert result.metadata["judge_call_resource"]["status"] == "FAILED"


def test_repeated_judges_report_variance_and_claim_agreement():
    first = score_report("sample", _report(), ["引用精度"], 0.2)
    first = first.model_copy(
        update={
            "fact_accuracy": 1.0,
            "citation_precision": 1.0,
            "metadata": {
                "evaluator": "llm_judge",
                "judge_profile": "judge_a",
                "source_diversity": 1,
                "claim_assessments": [
                    {"claim_id": "claim_1", "claim": "claim", "status": "supported"}
                ],
                "gold_claim_assessments": [
                    {"gold_claim_id": "gold_1", "status": "covered", "report_claim_ids": ["claim_1"]}
                ],
            },
        }
    )
    second = first.model_copy(
        update={
            "fact_accuracy": 0.6,
            "citation_precision": 0.0,
            "metadata": {
                "evaluator": "llm_judge",
                "judge_profile": "judge_b",
                "source_diversity": 1,
                "claim_assessments": [
                    {"claim_id": "claim_1", "claim": "claim", "status": "unsupported"}
                ],
                "gold_claim_assessments": [
                    {"gold_claim_id": "gold_1", "status": "not_covered", "report_claim_ids": []}
                ],
            },
        }
    )

    result = aggregate_judge_results([first, second])

    assert result.fact_accuracy == pytest.approx(0.8)
    assert result.metadata["judge_count"] == 2
    assert result.metadata["claim_status_agreement"] == 0.0
    assert result.metadata["cohen_kappa"] == 0.0
    assert result.metadata["metric_stddev"]["fact_accuracy"] == pytest.approx(0.2)
    assert result.metadata["label_aggregation_policy"] == (
        "majority_with_conservative_tie"
    )
    assert result.metadata["aggregate_claim_assessments"][0]["status"] == "unsupported"
    assert result.metadata["aggregate_claim_assessments"][0]["aggregation_reason"] == (
        "conservative_tie_break"
    )
    assert result.metadata["aggregate_gold_claim_assessments"][0]["status"] == (
        "not_covered"
    )


def test_three_repeated_judges_use_actual_majority_before_conservative_tie_break():
    """三次重复中 2:1 的多数结果不能被单个更保守标签覆盖。"""

    base = score_report("sample", _report(), ["引用精度"], 0.2)

    def with_status(claim_status: str, gold_status: str):
        return base.model_copy(
            update={
                "metadata": {
                    "evaluator": "llm_judge",
                    "judge_profile": "judge",
                    "source_diversity": 1,
                    "claim_assessments": [
                        {
                            "claim_id": "claim_1",
                            "claim": "claim",
                            "status": claim_status,
                        }
                    ],
                    "gold_claim_assessments": [
                        {
                            "gold_claim_id": "gold_1",
                            "status": gold_status,
                            "report_claim_ids": (
                                ["claim_1"] if gold_status != "not_covered" else []
                            ),
                        }
                    ],
                }
            }
        )

    result = aggregate_judge_results(
        [
            with_status("supported", "covered"),
            with_status("supported", "covered"),
            with_status("unsupported", "not_covered"),
        ]
    )

    assert result.metadata["judge_count"] == 3
    assert result.metadata["aggregate_claim_assessments"][0]["status"] == "supported"
    assert result.metadata["aggregate_claim_assessments"][0]["aggregation_reason"] == (
        "majority_vote"
    )
    assert result.metadata["aggregate_gold_claim_assessments"][0]["status"] == "covered"


def _hierarchical_result(
    profile: str,
    claim_status: str,
    gold_status: str,
    *,
    evaluator: str = "llm_judge",
) -> EvalResult:
    base = score_report("sample", _report(), ["引用精度"], 0.2)
    return base.model_copy(
        update={
            "metadata": {
                "evaluator": evaluator,
                "judge_profile": profile,
                "judge_model": profile,
                "judge_provider": "mock",
                "judge_protocol_version": "2.0",
                "judge_protocol_fingerprint": "sha256:test",
                "source_diversity": 1,
                "claim_assessments": [
                    {
                        "claim_id": "claim_1",
                        "claim": "claim",
                        "status": claim_status,
                    }
                ],
                "gold_claim_assessments": [
                    {
                        "gold_claim_id": "gold_1",
                        "status": gold_status,
                        "report_claim_ids": (
                            ["claim_1"] if gold_status != "not_covered" else []
                        ),
                        "required": True,
                        "weight": 2.0,
                    }
                ],
            }
        }
    )


def _hierarchical_aggregate(results):
    return aggregate_judge_results(
        results,
        aggregation_policy="hierarchical_model_consensus",
        expected_profiles=["deepseek", "glm"],
        repetitions_per_profile=3,
        min_successful_per_profile=2,
        min_resolved_rate=0.85,
        min_cross_model_agreement=0.80,
        min_cohen_kappa=0.60,
    )


def test_hierarchical_consensus_marks_two_unanimous_models_high_confidence():
    result = _hierarchical_aggregate(
        [
            *[_hierarchical_result("deepseek", "supported", "covered") for _ in range(3)],
            *[_hierarchical_result("glm", "supported", "covered") for _ in range(3)],
        ]
    )

    claim = result.metadata["aggregate_claim_assessments"][0]
    assert claim["status"] == "supported"
    assert claim["confidence"] == "high"
    assert claim["resolution_status"] == "resolved"
    assert result.metadata["automatic_consensus"]["gate"]["passed"] is True
    assert result.metadata["human_calibrated"] is False
    assert result.metadata["formal_benchmark_eligible"] is False


def test_hierarchical_consensus_marks_two_to_one_model_majority_medium():
    result = _hierarchical_aggregate(
        [
            _hierarchical_result("deepseek", "supported", "covered"),
            _hierarchical_result("deepseek", "supported", "covered"),
            _hierarchical_result(
                "deepseek", "partially_supported", "partially_covered"
            ),
            *[_hierarchical_result("glm", "supported", "covered") for _ in range(3)],
        ]
    )

    claim = result.metadata["aggregate_claim_assessments"][0]
    assert claim["status"] == "supported"
    assert claim["confidence"] == "medium"
    assert claim["resolution_status"] == "resolved"


def test_hierarchical_gate_rejects_consensus_that_required_gold_is_incomplete():
    result = _hierarchical_aggregate(
        [
            *[
                _hierarchical_result("deepseek", "supported", "partially_covered")
                for _ in range(3)
            ],
            *[
                _hierarchical_result("glm", "supported", "partially_covered")
                for _ in range(3)
            ],
        ]
    )

    gate = result.metadata["automatic_consensus"]["gate"]
    assert gate["passed"] is False
    assert gate["required_gold_claim_unresolved_ids"] == []
    assert gate["required_gold_claim_incomplete_ids"] == ["gold_1"]
    assert "required_gold_claim_not_fully_covered" in gate["failure_reasons"]


def test_consensus_diagnostics_exposes_prevalence_without_relaxing_kappa():
    rows = []
    for index in range(10):
        rows.append(
            {
                "resolution_status": "resolved" if index else "unresolved",
                "confidence": "high" if index else "low",
                "model_majorities": {
                    "flash": {"status": "supported"},
                    "pro": {
                        "status": "partially_supported" if index == 0 else "supported"
                    },
                },
            }
        )

    diagnostics = _consensus_diagnostics(rows, ["flash", "pro"])

    assert diagnostics["cross_model_agreement"] == pytest.approx(0.9)
    assert diagnostics["expected_chance_agreement"] == pytest.approx(0.9)
    assert diagnostics["cohen_kappa"] == pytest.approx(0.0)
    assert diagnostics["dominant_label_share"] == {"flash": 1.0, "pro": 0.9}
    assert diagnostics["prevalence_warning"] is True
    assert "gate remains active" in diagnostics["prevalence_note"]


def test_hierarchical_cross_model_conflict_is_conservative_and_unresolved():
    result = _hierarchical_aggregate(
        [
            *[_hierarchical_result("deepseek", "supported", "covered") for _ in range(3)],
            *[
                _hierarchical_result("glm", "unsupported", "not_covered")
                for _ in range(3)
            ],
        ]
    )

    claim = result.metadata["aggregate_claim_assessments"][0]
    gold = result.metadata["aggregate_gold_claim_assessments"][0]
    gate = result.metadata["automatic_consensus"]["gate"]
    assert claim["status"] == "unsupported"
    assert claim["confidence"] == "low"
    assert claim["resolution_status"] == "unresolved"
    assert claim["conflict"] is True
    assert gold["status"] == "not_covered"
    assert gate["passed"] is False
    assert "required_gold_claim_unresolved" in gate["failure_reasons"]


def test_hierarchical_fallback_is_excluded_from_vote_and_blocks_gate():
    result = _hierarchical_aggregate(
        [
            *[_hierarchical_result("deepseek", "supported", "covered") for _ in range(3)],
            _hierarchical_result("glm", "supported", "covered"),
            _hierarchical_result("glm", "supported", "covered"),
            _hierarchical_result(
                "glm",
                "unsupported",
                "not_covered",
                evaluator="rule_fallback",
            ),
        ]
    )

    claim = result.metadata["aggregate_claim_assessments"][0]
    gate = result.metadata["automatic_consensus"]["gate"]
    assert claim["status"] == "supported"
    assert claim["model_majorities"]["glm"]["votes"] == ["supported", "supported"]
    assert result.metadata["judge_fallback_count"] == 1
    assert gate["passed"] is False
    assert "llm_fallback_present" in gate["failure_reasons"]


def test_hierarchical_consensus_fails_closed_when_one_profile_has_too_few_calls():
    result = _hierarchical_aggregate(
        [
            *[_hierarchical_result("deepseek", "supported", "covered") for _ in range(3)],
            _hierarchical_result("glm", "supported", "covered"),
        ]
    )

    claim = result.metadata["aggregate_claim_assessments"][0]
    gate = result.metadata["automatic_consensus"]["gate"]
    assert claim["resolution_status"] == "degraded_single_model"
    assert claim["confidence"] == "low"
    assert gate["passed"] is False
    assert "profile_unavailable:glm" in gate["failure_reasons"]


def test_hierarchical_consensus_aligns_reordered_claims_by_stable_id():
    deepseek = _hierarchical_result("deepseek", "supported", "covered")
    glm = _hierarchical_result("glm", "supported", "covered")
    for result in (deepseek, glm):
        result.metadata["claim_assessments"].append(
            {
                "claim_id": "claim_2",
                "claim": "second claim",
                "status": "partially_supported",
            }
        )
    glm.metadata["claim_assessments"].reverse()

    result = aggregate_judge_results(
        [deepseek, deepseek.model_copy(deep=True), glm, glm.model_copy(deep=True)],
        aggregation_policy="hierarchical_model_consensus",
        expected_profiles=["deepseek", "glm"],
        repetitions_per_profile=2,
        min_successful_per_profile=2,
    )

    by_id = {
        item["claim_id"]: item
        for item in result.metadata["aggregate_claim_assessments"]
    }
    assert by_id["claim_1"]["status"] == "supported"
    assert by_id["claim_2"]["status"] == "partially_supported"
    assert all(item["resolution_status"] == "resolved" for item in by_id.values())


def test_automatic_consensus_summary_never_claims_human_calibration():
    result = _hierarchical_aggregate(
        [
            *[_hierarchical_result("deepseek", "supported", "covered") for _ in range(3)],
            *[_hierarchical_result("glm", "supported", "covered") for _ in range(3)],
        ]
    )

    summary = summarize_automatic_consensus([result])

    assert summary is not None
    assert summary["reference_source"] == "multi_model_automatic_consensus"
    assert summary["human_calibrated"] is False
    assert summary["formal_benchmark_eligible"] is False
    assert "annotator_id" not in str(summary)


def test_llm_judge_keeps_supported_contradicted_uncited_and_miscited_separate():
    """B2 验收样例：四类 claim 状态不能被折叠成“是否有引用”。"""

    report = _report().model_copy(
        update={
            "key_findings": [
                "引用精度需要检查证据支持关系。 [1]",
                "引用存在即可证明结论正确。 [1]",
                "无引用陈述。",
                "引用编号不存在。 [2]",
            ]
        }
    )
    client = MockLLMClient(
        responses=[
            """
            {
              "factuality": 3,
              "completeness": 3,
              "citation_quality": 2,
              "logical_structure": 4,
              "writing_quality": 4,
              "overall_reason": "Mixed grounding quality.",
              "claim_assessments": [
                {"claim_index": 1, "claim": "a", "status": "supported", "evidence_ids": ["ev_1"], "reason": "direct"},
                {"claim_index": 2, "claim": "b", "status": "contradicted", "evidence_ids": ["ev_1"], "reason": "evidence says existence is insufficient"},
                {"claim_index": 3, "claim": "c", "status": "unsupported", "evidence_ids": [], "reason": "no evidence"},
                {"claim_index": 4, "claim": "d", "status": "unsupported", "evidence_ids": [], "reason": "bad pointer"}
              ]
            }
            """
        ]
    )

    result = LLMEvaluatorAgent(client).evaluate(
        "sample",
        ResearchTask(query="test", run_id="judge_run"),
        report,
        [],
        0.2,
    )

    statuses = [item["status"] for item in result.metadata["claim_assessments"]]
    assert statuses == ["supported", "contradicted", "uncited", "miscited"]
    assert result.citation_existence == pytest.approx(0.75)
    assert result.citation_correctness == pytest.approx(1 / 3)
    assert result.citation_completeness == pytest.approx(0.25)
    assert result.hallucination_rate == pytest.approx(0.75)


def test_llm_judge_cannot_use_uncited_evidence_to_raise_citation_correctness():
    report = _report()
    extra = report.evidence[0].model_copy(update={"id": "ev_2"})
    report = report.model_copy(update={"evidence": [report.evidence[0], extra]})
    client = MockLLMClient(
        max_retries=0,
        responses=[
            """
            {
              "factuality": 5,
              "completeness": 5,
              "citation_quality": 5,
              "logical_structure": 5,
              "writing_quality": 5,
              "overall_reason": "Used another evidence row.",
              "claim_assessments": [
                {"claim_index": 1, "claim": "x", "status": "supported", "evidence_ids": ["ev_2"], "reason": "not actually cited"}
              ]
            }
            """
        ],
    )

    result = LLMEvaluatorAgent(client).evaluate(
        "sample", ResearchTask(query="test"), report, [], 0.1
    )

    assert result.metadata["evaluator"] == "rule_fallback"
    assert "not cited by claim" in result.metadata["judge_error"]


def test_b3_judge_protocol_uses_stable_ids_and_weighted_gold_coverage():
    report = _report()
    report_claim_id = extract_atomic_claims(report)[0].id
    gold = AtomicClaim(
        id="gold.c1",
        text="引用精度必须检查证据支持关系",
        satisfaction_routes=[
            AtomicClaimSatisfactionRoute(
                id="gold.c1.main",
                text="引用精度必须检查证据支持关系",
            )
        ],
        weight=2.0,
    )
    client = MockLLMClient(
        responses=[
            f"""
            {{
              "factuality": 5,
              "completeness": 5,
              "citation_quality": 5,
              "logical_structure": 4,
              "writing_quality": 4,
              "overall_reason": "Grounded and complete.",
              "claim_assessments": [
                {{
                  "claim_id": "{report_claim_id}",
                  "claim": "引用精度需要检查证据支持关系。",
                  "status": "supported",
                  "evidence_ids": ["ev_1"],
                  "reason": "direct"
                }}
              ],
              "gold_claim_assessments": [
                {{
                  "gold_claim_id": "gold.c1",
                  "status": "covered",
                  "report_claim_ids": ["{report_claim_id}"],
                  "matched_route_id": "gold.c1.main",
                  "reason": "same requirement"
                }}
              ]
            }}
            """
        ]
    )

    result = LLMEvaluatorAgent(client, shuffle_claims=True).evaluate(
        "sample",
        ResearchTask(query="test"),
        report,
        [gold.text],
        0.1,
        gold_claims=[gold],
        claim_order_seed=7,
    )

    assert result.weighted_claim_recall == 1.0
    assert result.metadata["judge_protocol_version"] == "5.0"
    assert result.metadata["judge_protocol_fingerprint"].startswith("sha256:")
    assert result.metadata["claim_order_seed"] == 7
    assert result.metadata["gold_claim_assessments"][0]["gold_claim_id"] == "gold.c1"
    assert result.metadata["gold_claim_assessments"][0]["matched_route_id"] == "gold.c1.main"


def test_judge_v5_rejects_covered_gold_without_valid_route():
    report = _report()
    report_claim_id = extract_atomic_claims(report)[0].id
    gold = AtomicClaim(id="gold.c1", text="引用精度必须检查证据支持关系")
    output = LLMJudgeOutput.model_validate(
        {
            "factuality": 5,
            "completeness": 5,
            "citation_quality": 5,
            "logical_structure": 5,
            "writing_quality": 5,
            "overall_reason": "brief",
            "claim_assessments": [
                {
                    "claim_id": report_claim_id,
                    "claim": report.key_findings[0],
                    "status": "supported",
                    "evidence_ids": ["ev_1"],
                }
            ],
            "gold_claim_assessments": [
                {
                    "gold_claim_id": "gold.c1",
                    "status": "covered",
                    "report_claim_ids": [report_claim_id],
                    "matched_route_id": "gold.c1.unknown",
                }
            ],
        }
    )

    with pytest.raises(ValueError, match="must match one valid route"):
        LLMEvaluatorAgent._validate_and_align_output(
            output,
            report,
            extract_atomic_claims(report),
            extract_atomic_claims(report),
            [gold],
            require_ids=True,
            require_route_match=True,
        )


def test_llm_judge_retries_once_when_schema_valid_output_omits_a_claim():
    """稳定 ID 漏评属于可纠正格式错误，不应直接伪装成规则 Judge 结果。"""

    report = _report().model_copy(
        update={
            "key_findings": [
                "引用精度需要检查证据支持关系。 [1]",
                "引用链接存在不等于语义支持。 [1]",
            ]
        }
    )
    claims = extract_atomic_claims(report)
    client = MockLLMClient(
        max_retries=0,
        responses=[
            f"""
            {{
              "factuality": 5,
              "completeness": 4,
              "citation_quality": 5,
              "logical_structure": 4,
              "writing_quality": 4,
              "overall_reason": "One row was accidentally omitted.",
              "claim_assessments": [
                {{"claim_id": "{claims[0].id}", "claim": "first", "status": "supported", "evidence_ids": ["ev_1"], "reason": "direct"}}
              ]
            }}
            """,
            f"""
            {{
              "factuality": 5,
              "completeness": 5,
              "citation_quality": 5,
              "logical_structure": 4,
              "writing_quality": 4,
              "overall_reason": "Complete replacement.",
              "claim_assessments": [
                {{"claim_id": "{claims[0].id}", "claim": "first", "status": "supported", "evidence_ids": ["ev_1"], "reason": "direct"}},
                {{"claim_id": "{claims[1].id}", "claim": "second", "status": "supported", "evidence_ids": ["ev_1"], "reason": "direct"}}
              ]
            }}
            """,
        ],
    )

    result = LLMEvaluatorAgent(client, shuffle_claims=True).evaluate(
        "sample",
        ResearchTask(query="如何评测引用质量？", run_id="judge_run"),
        report,
        [],
        0.2,
        claim_order_seed=7,
    )

    assert result.metadata["evaluator"] == "llm_judge"
    assert result.metadata["judge_alignment_retry_used"] is True
    assert len(result.metadata["claim_assessments"]) == 2
    assert [trace.agent for trace in client.trace_collector.since(0)] == [
        "evaluator",
        "evaluator_alignment_retry",
    ]

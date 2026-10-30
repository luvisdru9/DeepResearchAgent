import asyncio

import pytest

from deepresearch.agents.redteam import LLMRedTeamAgent
from deepresearch.llm import MockLLMClient
from deepresearch.schema import Evidence, ResearchReport, ResearchTask


def _evidence() -> list[Evidence]:
    """Return one stable evidence item for RedTeam citation validation."""

    return [
        Evidence(
            id="ev_supported",
            claim="证据先行提高结论可追溯性。",
            quote="结构化证据记录来源与原文片段，使关键结论可以回溯。",
            source_id="doc_1",
            source_title="Evidence Design",
            source_url="https://example.com/evidence",
            confidence=0.9,
            supports_subtask="task_1",
            extraction_method="llm",
        )
    ]


def _report() -> ResearchReport:
    """Return a small draft report that the RedTeam agent can review."""

    return ResearchReport(
        run_id="run_redteam",
        query="如何降低研究报告幻觉？",
        title="测试报告",
        summary="测试摘要。",
        key_findings=["证据先行可以降低所有幻觉。 [1]"],
        evidence=_evidence(),
        limitations=[],
        markdown=(
            "# 测试报告\n\n"
            "## 关键发现\n"
            "- 证据先行可以降低所有幻觉。 [1]\n\n"
            "## RedTeam 检查\n"
            "- 未发现明显结构性问题。\n"
        ),
    )


def test_llm_redteam_reviews_report_with_valid_evidence_ids():
    client = MockLLMClient(
        responses=[
            """
            {
              "overall_status": "needs_revision",
              "summary": "报告存在过度推断。",
              "findings": [
                {
                  "claim": "证据先行可以降低所有幻觉。",
                  "status": "weakly_supported",
                  "severity": "high",
                  "issue": "证据只支持可追溯性，不能证明降低所有幻觉。",
                  "evidence_ids": ["ev_supported"],
                  "reason": "quote 没有给出幻觉率实验结果。",
                  "suggested_fix": "将结论降级为提高可追溯性，避免声称降低所有幻觉。",
                  "category": "citation_support",
                  "repair_action": "narrow"
                }
              ],
              "missing_perspectives": ["缺少反例与失败案例"]
            }
            """
        ]
    )
    redteam = LLMRedTeamAgent(client)

    findings = asyncio.run(
        redteam.review_async(
            _report().key_findings,
            _evidence(),
            task=ResearchTask(query="如何降低研究报告幻觉？"),
            report=_report(),
        )
    )

    assert redteam.last_trace.used_provider == "llm"
    assert redteam.last_trace.overall_status == "needs_revision"
    assert findings[0].status == "weakly_supported"
    assert findings[0].evidence_ids == ["ev_supported"]
    assert findings[0].repair_action == "narrow"
    assert findings[1].issue.startswith("可能遗漏视角")
    assert client.trace_collector.since(0)[0].agent == "redteam"


def test_redteam_prompt_keeps_confidence_and_source_authority_separate():
    evidence = [
        _evidence()[0].model_copy(
            update={"source_credibility_score": 0.2, "source_credibility_tier": "low"}
        )
    ]
    agent = LLMRedTeamAgent(MockLLMClient())

    messages = agent._build_messages(
        _report().key_findings,
        evidence,
        task=ResearchTask(query="测试元数据语义"),
        report=_report().model_copy(update={"evidence": evidence}),
        mode="verify",
        prior_findings=[],
    )
    prompt = "\n".join(message.content for message in messages)

    assert "reader_extraction_confidence=0.90" in prompt
    assert "source_authority_heuristic_tier=low" in prompt
    assert "not source authority" in prompt
    assert '"repair_action":"retrieve|narrow|remove|qualify|none"' in prompt
    assert "at least N empirical studies" in prompt
    assert "method, or bibliographic identity without a study result does not count" in prompt


def test_llm_redteam_rejects_unknown_evidence_id_when_fallback_disabled():
    client = MockLLMClient(
        responses=[
            """
            {
              "overall_status": "needs_revision",
              "summary": "引用了不存在的证据。",
              "findings": [
                {
                  "claim": "无效引用。",
                  "status": "unsupported",
                  "severity": "high",
                  "issue": "该审查结果引用了不存在的 evidence id。",
                  "evidence_ids": ["ev_missing"],
                  "reason": "ev_missing 不在输入证据表中。",
                  "suggested_fix": "只引用输入证据表中的 evidence id。"
                }
              ],
              "missing_perspectives": []
            }
            """
        ]
    )
    redteam = LLMRedTeamAgent(client, fallback_to_rule=False)

    with pytest.raises(ValueError, match="unknown evidence IDs"):
        asyncio.run(redteam.review_async(_report().key_findings, _evidence(), report=_report()))


def test_llm_redteam_verify_mode_records_separate_verifier_call():
    client = MockLLMClient(
        responses=[
            """
            {
              "overall_status": "pass",
              "summary": "绝对化表述已经删除，未发现剩余问题。",
              "findings": [],
              "missing_perspectives": []
            }
            """
        ]
    )
    redteam = LLMRedTeamAgent(client)

    findings = asyncio.run(
        redteam.review_async(
            ["结构化证据有助于提高结论可追溯性。 [1]"],
            _evidence(),
            report=_report(),
            mode="verify",
            prior_findings=[],
        )
    )

    assert findings == []
    assert redteam.last_trace.overall_status == "pass"
    assert client.trace_collector.since(0)[0].agent == "verifier"

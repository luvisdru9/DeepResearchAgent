import asyncio

import pytest

from deepresearch.agents.synthesizer import (
    LLMSynthesizerAgent,
    SynthesizedFinding,
    SynthesizerAgent,
    SynthesizerOutput,
)
from deepresearch.llm import MockLLMClient
from deepresearch.schema import (
    Evidence,
    EvidenceCoverageMatrix,
    EvidenceCoverageSubject,
    RedTeamFinding,
    ResearchReport,
    ResearchTask,
)


def _evidence() -> list[Evidence]:
    """Return stable evidence IDs so citation binding can be asserted."""

    return [
        Evidence(
            id="ev_grounded",
            claim="结构化证据能够提高结论的可追溯性。",
            quote="结构化证据记录来源与原文片段，使关键结论可以回溯。",
            source_id="doc_1",
            source_title="Evidence Design",
            source_url="https://example.com/evidence",
            confidence=0.9,
            supports_subtask="task_1",
        )
    ]


def test_llm_synthesizer_binds_valid_evidence_ids_to_local_citations():
    client = MockLLMClient(
        responses=[
            """
            {
              "title": "可追溯研究报告",
              "summary": "报告基于给定证据总结。",
              "key_findings": [
                {
                  "text": "结构化证据提高了结论的可追溯性。",
                  "evidence_ids": ["ev_grounded"]
                }
              ],
              "limitations": ["当前仅有一个来源。"]
            }
            """
        ]
    )
    synthesizer = LLMSynthesizerAgent(client)

    report = asyncio.run(
        synthesizer.synthesize_async(
            ResearchTask(query="如何提高研究结论的可追溯性？"),
            _evidence(),
        )
    )

    assert report.title == "可追溯研究报告"
    assert report.key_findings == ["结构化证据提高了结论的可追溯性。 [1]"]
    assert "## 证据表" in report.markdown
    # 草稿阶段不生成“未发现问题”的占位结论，避免 RedTeam 审查系统自身文本。
    assert "## RedTeam 检查" not in report.markdown
    assert synthesizer.last_trace.used_provider == "llm"
    assert client.trace_collector.since(0)[0].agent == "synthesizer"


def test_llm_synthesizer_renders_english_report_chrome_for_english_task():
    output = SynthesizerOutput(
        title="Remote Work Evidence",
        summary="The evidence is mixed.",
        key_findings=[
            SynthesizedFinding(
                text="Effects vary by context.",
                evidence_ids=["ev_grounded"],
            )
        ],
        limitations=["The evidence is heterogeneous."],
    )

    report = LLMSynthesizerAgent(MockLLMClient())._build_report(
        ResearchTask(query="What are the effects?", language="en"),
        output,
        _evidence(),
        [],
    )

    assert "## Summary" in report.markdown
    assert "## Key Findings" in report.markdown
    assert "## Evidence" in report.markdown
    assert "## Limitations" in report.markdown
    assert "Source:" in report.markdown
    assert "## 摘要" not in report.markdown


def test_methodology_prompt_requires_cited_operational_findings():
    task = ResearchTask(query="如何评测引用质量和幻觉率？")

    messages = LLMSynthesizerAgent._build_messages(task, _evidence(), [])
    system = messages[0].content

    assert "directly cited key_findings" in system
    assert "proportion of evidence-requiring atomic claims" in system
    assert "semantic entailment/support" in system
    assert "unsupported, contradicted, miscited, and uncited" in system


def test_red_blue_prompt_distinguishes_subject_content_from_internal_logs():
    task = ResearchTask(query="Red-Blue 对抗核查如何改进深度研究报告？")

    messages = LLMSynthesizerAgent._build_messages(task, _evidence(), [])
    system = messages[0].content

    assert "evidence-grounded Blue revision" in system
    assert "Verifier rechecking after revision" in system
    assert "Do not expose this run's internal" in system


def test_red_blue_initial_synthesis_retries_missing_process_obligations():
    evidence = [
        Evidence(
            id="ev_blue",
            claim="Blue 根据证据修订报告。",
            quote="Red Agent 指出问题，然后 Blue Agent 根据证据修订报告。",
            source_id="doc_blue",
            source_title="Red-Blue",
            source_url="local://blue",
        ),
        Evidence(
            id="ev_verify",
            claim="Verifier 在修订后核验并标记不确定性。",
            quote="Verifier 可以在修订后输出最终质量检查，标记仍然存在的不确定性。",
            source_id="doc_verify",
            source_title="Verification",
            source_url="local://verify",
        ),
    ]
    client = MockLLMClient(
        responses=[
            """
            {"title":"不完整流程","summary":"只描述攻击。","key_findings":[
              {"text":"Red Agent 主动查找报告缺陷。","evidence_ids":["ev_blue"]}
            ],"limitations":[]}
            """,
            """
            {"title":"完整流程","summary":"描述证据修订和复核。","key_findings":[
              {"text":"Blue Agent 根据 Evidence 修订报告。","evidence_ids":["ev_blue"]},
              {"text":"Verifier 在修订后重新核验，并标记仍然存在的不确定性。","evidence_ids":["ev_verify"]}
            ],"limitations":[]}
            """,
        ]
    )
    synthesizer = LLMSynthesizerAgent(client, fallback_to_rule=False)

    report = asyncio.run(
        synthesizer.synthesize_async(
            ResearchTask(query="Red-Blue 对抗核查如何改进深度研究报告？"),
            evidence,
        )
    )

    assert report.title == "完整流程"
    assert [item.agent for item in client.trace_collector.since(0)] == [
        "synthesizer",
        "synthesizer_methodology_retry",
    ]
    assert all(
        item["status"] == "satisfied"
        for item in synthesizer.last_trace.methodology_diagnostics
    )


def test_initial_synthesis_retries_one_non_substantive_finding():
    client = MockLLMClient(
        responses=[
            """
            {
              "title": "初稿",
              "summary": "包含截断内容。",
              "key_findings": [{
                "text": "Enabling Large Language Models to Generate Text with Citations...",
                "evidence_ids": ["ev_grounded"]
              }],
              "limitations": []
            }
            """,
            """
            {
              "title": "修复后的报告",
              "summary": "关键发现是完整命题。",
              "key_findings": [{
                "text": "结构化证据提高了结论的可追溯性。",
                "evidence_ids": ["ev_grounded"]
              }],
              "limitations": []
            }
            """,
        ]
    )
    synthesizer = LLMSynthesizerAgent(client)

    report = asyncio.run(
        synthesizer.synthesize_async(ResearchTask(query="如何提高可追溯性？"), _evidence())
    )

    assert report.title == "修复后的报告"
    assert [item.agent for item in client.trace_collector.since(0)] == [
        "synthesizer",
        "synthesizer_substance_retry",
    ]
    assert all(
        item["status"] == "substantive"
        for item in synthesizer.last_trace.substance_diagnostics
    )


def test_rule_synthesizer_replaces_truncated_claim_with_complete_quote():
    evidence = _evidence()[0].model_copy(
        update={
            "claim": "Evidence Design...",
            "quote": "结构化证据记录来源与原文片段，使关键结论可以回溯。",
        }
    )

    report = SynthesizerAgent().synthesize(
        ResearchTask(query="如何提高可追溯性？"),
        [evidence],
    )

    assert report.evidence == [evidence]
    assert report.key_findings == [
        "结构化证据记录来源与原文片段，使关键结论可以回溯。 [1]"
    ]
    assert any("完整原文片段" in item for item in report.limitations)


def test_blue_revision_prunes_bibliographic_or_truncated_finding():
    client = MockLLMClient(
        responses=[
            """
            {
              "title": "修订报告",
              "summary": "删除无实质内容的条目。",
              "key_findings": [
                {
                  "text": "Citation Research Paper...",
                  "evidence_ids": ["ev_grounded"]
                },
                {
                  "text": "结构化证据提高了结论的可追溯性。",
                  "evidence_ids": ["ev_grounded"]
                }
              ],
              "limitations": []
            }
            """
        ]
    )
    synthesizer = LLMSynthesizerAgent(client)
    task = ResearchTask(query="如何提高可追溯性？")
    draft = ResearchReport(
        run_id=task.run_id,
        query=task.query,
        title="初稿",
        summary="摘要",
        key_findings=["旧结论。 [1]"],
        evidence=_evidence(),
        markdown="# 初稿\n\n## 关键发现\n- 旧结论。 [1]\n",
    )

    revised = asyncio.run(synthesizer.revise_async(task, draft, _evidence(), []))

    assert revised.key_findings == ["结构化证据提高了结论的可追溯性。 [1]"]
    assert [
        item["status"] for item in synthesizer.last_trace.substance_diagnostics
    ] == ["truncated_fragment", "substantive"]


def test_blue_revision_rejects_output_when_every_finding_is_non_substantive():
    client = MockLLMClient(
        responses=[
            """
            {
              "title": "无效修订",
              "summary": "没有完整结论。",
              "key_findings": [{
                "text": "Abstract: citation quality...",
                "evidence_ids": ["ev_grounded"]
              }],
              "limitations": []
            }
            """
        ]
    )
    synthesizer = LLMSynthesizerAgent(client)
    task = ResearchTask(query="如何提高可追溯性？")
    draft = ResearchReport(
        run_id=task.run_id,
        query=task.query,
        title="初稿",
        summary="摘要",
        key_findings=["旧结论。 [1]"],
        evidence=_evidence(),
        markdown="# 初稿\n\n## 关键发现\n- 旧结论。 [1]\n",
    )

    revised = asyncio.run(synthesizer.revise_async(task, draft, _evidence(), []))

    assert revised is draft
    assert synthesizer.last_trace.used_provider == "draft"
    assert "no substantive findings" in (synthesizer.last_trace.error or "")


def test_llm_synthesizer_adds_deterministic_low_credibility_limitation():
    client = MockLLMClient(
        responses=[
            """
            {
              "title": "低可信来源报告",
              "summary": "报告保留来源风险提示。",
              "key_findings": [{
                "text": "这是一条待复核结论。",
                "evidence_ids": ["ev_grounded"]
              }],
              "limitations": []
            }
            """
        ]
    )
    evidence = [
        _evidence()[0].model_copy(
            update={"source_credibility_score": 0.3, "source_credibility_tier": "low"}
        )
    ]

    report = asyncio.run(
        LLMSynthesizerAgent(client).synthesize_async(
            ResearchTask(query="测试低可信来源提示"),
            evidence,
        )
    )

    assert any("低可信度来源" in item for item in report.limitations)
    assert "低可信度来源" in report.markdown


def test_llm_synthesizer_replaces_model_low_credibility_disclosure_with_one_local_count():
    evidence = [
        _evidence()[0].model_copy(
            update={"source_credibility_score": 0.3, "source_credibility_tier": "low"}
        )
    ]
    output = SynthesizerOutput(
        title="报告",
        summary="摘要",
        key_findings=[SynthesizedFinding(text="结论", evidence_ids=[evidence[0].id])],
        limitations=[
            "报告使用了一条来自低可信度规则启发式来源的证据。",
            "其他局限。",
        ],
    )

    report = LLMSynthesizerAgent(MockLLMClient())._build_report(
        ResearchTask(query="test"),
        output,
        evidence,
        [],
    )

    assert sum("低可信度" in item for item in report.limitations) == 1
    assert "其他局限。" in report.limitations


def test_synthesizer_prompt_distinguishes_extraction_confidence_from_source_authority():
    evidence = [
        _evidence()[0].model_copy(
            update={"source_credibility_score": 0.3, "source_credibility_tier": "low"}
        )
    ]

    messages = LLMSynthesizerAgent._build_messages(
        ResearchTask(query="测试元数据语义"), evidence, []
    )
    prompt = "\n".join(message.content for message in messages)

    assert "reader_extraction_confidence=0.90" in prompt
    assert "source_authority_heuristic_tier=low" in prompt
    assert "does not measure the authority of the source" in prompt
    assert "Do not emit a key finding that merely says a bibliography" in prompt
    assert "make the head-to-head comparison explicit" in prompt
    assert "include a directly cited result or conclusion for N distinct" in prompt
    assert "state the analysis unit, numerator/denominator for rates" in prompt
    assert "proposed adaptation or recommendation" in prompt


def test_synthesizer_prompt_includes_deterministic_coverage_matrix():
    matrix = EvidenceCoverageMatrix(
        subjects=[
            EvidenceCoverageSubject(
                subject="ALCE",
                evidence_ids=["ev_grounded"],
                dimension_evidence_ids={
                    "goal": ["ev_grounded"],
                    "data": [],
                    "metrics": ["ev_grounded"],
                    "scope": [],
                },
                missing_dimensions=["data", "scope"],
            )
        ]
    )

    messages = LLMSynthesizerAgent._build_messages(
        ResearchTask(query="Compare ALCE"),
        _evidence(),
        [],
        coverage_matrix=matrix,
    )
    prompt = "\n".join(message.content for message in messages)

    assert "subject=ALCE" in prompt
    assert "goal=ev_grounded" in prompt
    assert "data=MISSING" in prompt
    assert "Never claim an entire subject is absent" in prompt


def test_llm_report_lists_only_cited_evidence_with_unambiguous_quality_labels():
    client = MockLLMClient(
        responses=[
            """
            {
              "title": "精简证据表",
              "summary": "只保留被引用证据。",
              "key_findings": [{
                "text": "结构化证据提高可追溯性。",
                "evidence_ids": ["ev_grounded"]
              }],
              "limitations": []
            }
            """
        ]
    )
    cited = _evidence()[0].model_copy(
        update={"source_credibility_score": 0.8, "source_credibility_tier": "high"}
    )
    unused = cited.model_copy(
        update={
            "id": "ev_unused",
            "claim": "无关上下文。",
            "quote": "这条证据没有被报告结论引用。",
            "source_id": "doc_unused",
            "source_title": "Unused source",
            "source_url": "https://example.com/unused",
        }
    )

    report = asyncio.run(
        LLMSynthesizerAgent(client).synthesize_async(
            ResearchTask(query="测试证据表精简"),
            [cited, unused],
        )
    )

    assert [item.id for item in report.evidence] == ["ev_grounded"]
    assert "Unused source" not in report.markdown
    assert "抽取置信度：0.90" in report.markdown
    assert "来源可信度（规则估计）：high" in report.markdown


def test_llm_synthesizer_revises_draft_from_actionable_redteam_findings():
    client = MockLLMClient(
        responses=[
            """
            {
              "title": "修订后的可追溯研究报告",
              "summary": "修订后避免了绝对化表述。",
              "key_findings": [
                {
                  "text": "结构化证据有助于提高结论可追溯性。",
                  "evidence_ids": ["ev_grounded"]
                }
              ],
              "limitations": ["当前证据不能证明完全消除幻觉。"]
            }
            """
        ]
    )
    synthesizer = LLMSynthesizerAgent(client)
    task = ResearchTask(query="如何提高研究结论的可追溯性？")
    draft = ResearchReport(
        run_id=task.run_id,
        query=task.query,
        title="初稿",
        summary="初稿摘要。",
        key_findings=["结构化证据可以完全消除幻觉。 [1]"],
        evidence=_evidence(),
        markdown="# 初稿\n\n## 关键发现\n- 结构化证据可以完全消除幻觉。 [1]\n",
    )
    review = [
        RedTeamFinding(
            claim="结构化证据可以完全消除幻觉。",
            status="weakly_supported",
            severity="high",
            issue="结论超过证据支持范围。",
            reason="证据只说明可追溯性。",
            suggested_fix="改为有助于提高可追溯性。",
            evidence_ids=["ev_grounded"],
        )
    ]

    report = asyncio.run(
        synthesizer.revise_async(task, draft, _evidence(), review)
    )

    assert report.title == "修订后的可追溯研究报告"
    assert all("完全消除" not in finding for finding in report.key_findings)
    assert "当前证据不能证明完全消除幻觉" in report.markdown
    assert report.key_findings == ["结构化证据有助于提高结论可追溯性。 [1]"]
    assert synthesizer.last_trace.used_provider == "llm"
    trace = client.trace_collector.since(0)[0]
    assert trace.agent == "blue_revision"


def test_blue_revision_prunes_one_uncited_finding_without_discarding_valid_repairs():
    """单条无引用输出应被删除，不应让其他有根据的修订整体回退。"""

    client = MockLLMClient(
        responses=[
            """
            {
              "title": "局部修订",
              "summary": "保留有根据内容。",
              "key_findings": [
                {
                  "text": "结构化证据有助于提高可追溯性。",
                  "evidence_ids": ["ev_grounded"]
                },
                {
                  "text": "这条表述没有引用。",
                  "evidence_ids": []
                }
              ],
              "limitations": []
            }
            """
        ]
    )
    synthesizer = LLMSynthesizerAgent(client, fallback_to_rule=True)
    task = ResearchTask(query="测试局部无引用修订")
    draft = ResearchReport(
        run_id=task.run_id,
        query=task.query,
        title="草稿",
        summary="草稿摘要。",
        key_findings=["结构化证据完全消除幻觉。 [1]"],
        evidence=_evidence(),
        markdown="# 草稿\n",
    )

    report = asyncio.run(
        synthesizer.revise_async(
            task,
            draft,
            _evidence(),
            [
                RedTeamFinding(
                    issue="草稿超出证据支持范围。",
                    suggested_fix="删除无引用主张并收窄结论。",
                    category="citation_support",
                    repair_action="narrow",
                    evidence_ids=["ev_grounded"],
                )
            ],
        )
    )

    assert report.title == "局部修订"
    assert report.key_findings == ["结构化证据有助于提高可追溯性。 [1]"]
    assert "这条表述没有引用" not in report.markdown
    assert synthesizer.last_trace.used_provider == "llm"


def test_blue_revision_still_fails_when_every_finding_is_uncited():
    output = SynthesizerOutput(
        title="无引用修订",
        summary="无可交付内容。",
        key_findings=[SynthesizedFinding(text="无引用结论。", evidence_ids=[])],
    )

    with pytest.raises(ValueError, match="no evidence-grounded findings"):
        LLMSynthesizerAgent._drop_uncited_revision_findings(
            output,
            require_citations=True,
        )

def test_llm_synthesizer_rejects_unknown_evidence_id_and_falls_back():
    client = MockLLMClient(
        responses=[
            """
            {
              "title": "含无效引用的报告",
              "summary": "该输出引用了不存在的证据。",
              "key_findings": [
                {"text": "未经支持的结论。", "evidence_ids": ["ev_invented"]}
              ],
              "limitations": []
            }
            """
        ]
    )
    synthesizer = LLMSynthesizerAgent(client, fallback_to_rule=True)

    report = asyncio.run(
        synthesizer.synthesize_async(
            ResearchTask(query="测试无效引用"),
            _evidence(),
        )
    )

    assert report.title.startswith("Deep Research Report:")
    assert synthesizer.last_trace.used_provider == "rule"
    assert synthesizer.last_trace.fallback_used is True
    assert "unknown evidence IDs" in (synthesizer.last_trace.error or "")


def test_llm_synthesizer_can_disable_rule_fallback():
    client = MockLLMClient(
        responses=[
            """
            {
              "title": "缺少引用",
              "summary": "输出未引用证据。",
              "key_findings": [{"text": "无引用结论。", "evidence_ids": []}],
              "limitations": []
            }
            """
        ]
    )
    synthesizer = LLMSynthesizerAgent(client, fallback_to_rule=False)

    with pytest.raises(ValueError, match="has no evidence citation"):
        asyncio.run(
            synthesizer.synthesize_async(
                ResearchTask(query="测试关闭回退"),
                _evidence(),
            )
        )


def test_final_text_revision_cannot_introduce_previously_uncited_evidence():
    """终局文字修订只能删减或收窄，不能借机改用草稿之外的证据。"""

    client = MockLLMClient(
        responses=[
            """
            {
              "title": "越界修订",
              "summary": "修订引用了新证据。",
              "key_findings": [{
                "text": "这条结论改用了草稿中未引用的证据。",
                "evidence_ids": ["ev_unused"]
              }],
              "limitations": []
            }
            """
        ]
    )
    synthesizer = LLMSynthesizerAgent(client, fallback_to_rule=False)
    task = ResearchTask(query="测试终局修订证据边界")
    cited = _evidence()[0]
    unused = cited.model_copy(
        update={
            "id": "ev_unused",
            "source_id": "doc_2",
            "source_title": "Unused Evidence",
            "source_url": "https://example.com/unused",
            "claim": "另一条结论。",
            "quote": "这是完整上下文中可用但草稿未引用的证据。",
        }
    )
    draft = ResearchReport(
        run_id=task.run_id,
        query=task.query,
        title="草稿",
        summary="草稿摘要。",
        key_findings=["结构化证据提高可追溯性。 [1]"],
        evidence=[cited],
        markdown="# 草稿\n\n- 结构化证据提高可追溯性。 [1]\n",
    )
    findings = [
        RedTeamFinding(
            claim="结构化证据提高可追溯性。",
            issue="需要收窄措辞。",
            suggested_fix="只保留已有证据支持的范围。",
            category="citation_support",
            repair_action="narrow",
            evidence_ids=["ev_grounded"],
        )
    ]

    with pytest.raises(ValueError, match="introduced new evidence IDs: ev_unused"):
        asyncio.run(
            synthesizer.revise_text_only_async(
                task,
                draft,
                [cited, unused],
                findings,
            )
        )

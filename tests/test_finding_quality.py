from deepresearch.quality import assess_finding_substance, assess_report_findings
from deepresearch.schema import Evidence


def _paper() -> Evidence:
    return Evidence(
        id="ev_paper",
        claim="该研究发现引用能提高可验证性。",
        quote="Our results show that citations improve verifiability.",
        source_id="paper",
        source_title="Enabling Large Language Models to Generate Text with Citations",
        source_url="https://example.com/paper",
    )


def test_finding_quality_detects_truncated_and_non_propositional_fragments():
    diagnostics = assess_report_findings(
        "如何评价引用质量？",
        [
            "RAGTruth: A Hallucination Corpus for Developing Trustworthy Models... [1]",
            "Abstract: hallucination evaluation",
            "结构化引用提高了报告的可验证性。 [1]",
        ],
        [_paper()],
    )

    assert [item.status for item in diagnostics] == [
        "truncated_fragment",
        "non_propositional",
        "substantive",
    ]


def test_finding_quality_rejects_title_only_but_allows_bibliometric_query():
    ordinary = assess_finding_substance(
        "引用能否降低幻觉？",
        _paper().source_title,
        finding_index=1,
        evidence=[_paper()],
    )
    bibliometric = assess_finding_substance(
        "列出引用生成研究的论文列表并进行文献计量分析",
        _paper().source_title,
        finding_index=1,
        evidence=[_paper()],
    )

    assert ordinary.status == "bibliographic_only"
    assert bibliometric.status == "substantive"
    assert bibliometric.allowed_by_query is True


def test_finding_quality_does_not_reject_complete_identity_plus_result_claim():
    diagnostic = assess_finding_substance(
        "比较引用生成研究",
        "Gao 等人的研究发现引用能够提高生成文本的可验证性。",
        finding_index=1,
        evidence=[_paper()],
    )

    assert diagnostic.status == "substantive"


def test_finding_quality_rejects_author_lists_and_explicit_truncation_shapes():
    diagnostics = assess_report_findings(
        "比较引用质量和幻觉评测方法",
        [
            "Authors: Tianyu Gao, Howard Yen, Jiatong Yu [1]",
            "作者：Tianyu Gao、Howard Yen、Jiatong Yu [1]",
            "Abstract... (truncated) [1]",
            "关键词：citation generation, hallucination",
        ],
        [_paper()],
    )

    assert [item.status for item in diagnostics] == [
        "bibliographic_only",
        "bibliographic_only",
        "truncated_fragment",
        "non_propositional",
    ]


def test_finding_quality_keeps_complete_multilingual_research_propositions():
    diagnostics = assess_report_findings(
        "比较引用质量研究",
        [
            "Gao 等人发现，自动引用评测结果与人工判断具有较强相关性。 [1]",
            "The study found that citation-aware generation improved verifiability. [1]",
        ],
        [_paper()],
    )

    assert all(item.status == "substantive" for item in diagnostics)

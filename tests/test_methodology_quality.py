from deepresearch.quality.methodology import (
    assess_methodology_obligations,
    assess_source_absence_claims,
    report_finding_inputs,
)
from deepresearch.schema import Evidence


QUERY = "如何评测 DeepResearchAgent 的引用质量和幻觉率？"


def _evidence() -> list[Evidence]:
    return [
        Evidence(
            id="ev_coverage",
            claim="Citation completeness asks whether each statement cites evidence.",
            quote="Every statement requiring citations is checked for citation completeness.",
            source_id="s1",
            source_title="Citation evaluation",
            source_url="https://example.com/coverage",
        ),
        Evidence(
            id="ev_correctness",
            claim="Citation correctness checks entailment.",
            quote="A citation is correct when the cited passage entails and supports the claim.",
            source_id="s2",
            source_title="Citation correctness",
            source_url="https://example.com/correctness",
        ),
        Evidence(
            id="ev_hallucination",
            claim="Hallucination is checked at claim level using support relations.",
            quote="Break claims into atomic claims and label unsupported or contradictory content.",
            source_id="s3",
            source_title="Hallucination taxonomy",
            source_url="https://example.com/hallucination",
        ),
    ]


def _complete_findings() -> list[tuple[str, list[str]]]:
    return [
        (
            "引用覆盖率以需要证据的原子主张为单位，分子是具有有效引用的主张数，"
            "分母是需要证据的原子主张总数。",
            ["ev_coverage"],
        ),
        (
            "引用正确性检查证据是否在语义上蕴含或支持对应主张，不能只检查链接存在。",
            ["ev_correctness"],
        ),
        (
            "幻觉率在原子主张层分别计数无支持、矛盾、错引和无引用陈述。",
            ["ev_hallucination"],
        ),
    ]


def test_methodology_gate_accepts_three_complete_cited_definitions():
    diagnostics = assess_methodology_obligations(
        QUERY,
        _complete_findings(),
        _evidence(),
    )

    assert [item.status for item in diagnostics] == [
        "satisfied",
        "satisfied",
        "satisfied",
    ]
    assert diagnostics[0].matched_finding_indexes == [1]


def test_methodology_gate_distinguishes_missing_wording_from_missing_evidence():
    diagnostics = assess_methodology_obligations(
        QUERY,
        _complete_findings()[:2],
        _evidence(),
    )
    insufficient = assess_methodology_obligations(
        QUERY,
        _complete_findings()[:2],
        _evidence()[:2],
    )

    assert diagnostics[2].status == "missing"
    assert insufficient[2].status == "evidence_insufficient"


def test_methodology_gate_does_not_accept_definition_only_in_summary_input():
    diagnostics = assess_methodology_obligations(
        QUERY,
        [],
        _evidence(),
    )

    assert all(item.status == "missing" for item in diagnostics)


def test_source_absence_requires_direct_assertion_in_mapped_evidence():
    unverified = assess_source_absence_claims(
        [("该论文未报告实验结果。", ["ev_coverage"])],
        _evidence(),
    )
    direct_evidence = _evidence() + [
        Evidence(
            id="ev_absence",
            claim="The paper does not report experiments.",
            quote="This paper does not provide an empirical experiment.",
            source_id="s4",
            source_title="Scope note",
            source_url="https://example.com/scope",
        )
    ]
    supported = assess_source_absence_claims(
        [("该论文未报告实验结果。", ["ev_absence"])],
        direct_evidence,
    )

    assert unverified[0].status == "unverified"
    assert supported[0].status == "directly_supported"


def test_source_absence_detects_claim_that_evidence_did_not_directly_define_metric():
    diagnostics = assess_source_absence_claims(
        [("现有证据未直接定义这些指标或类别。", ["ev_coverage"])],
        _evidence(),
    )

    assert diagnostics[0].status == "unverified"


def test_source_absence_detects_reversed_not_in_evidence_wording():
    diagnostics = assess_source_absence_claims(
        [("具体类别名未在现有证据中给出。", ["ev_hallucination"])],
        _evidence(),
    )

    assert diagnostics[0].status == "unverified"


def test_source_absence_detects_unattributed_empirical_validation_gap():
    diagnostics = assess_source_absence_claims(
        [("该机制预期降低幻觉率，但尚缺实证数据验证。", ["ev_coverage"])],
        _evidence(),
    )

    assert diagnostics[0].status == "unverified"


def test_report_finding_inputs_maps_only_valid_local_citations():
    mapped = report_finding_inputs(
        ["第一条。 [2] [99]", "第二条没有引用。"],
        _evidence(),
    )

    assert mapped == [
        ("第一条。 [2] [99]", ["ev_correctness"]),
        ("第二条没有引用。", []),
    ]


def test_red_blue_process_obligations_require_grounded_revision_and_recheck():
    query = "Red-Blue 对抗核查如何改进深度研究报告？"
    evidence = [
        Evidence(
            id="ev_blue",
            claim="Blue revises the report based on evidence.",
            quote="Red Agent attacks the report, then Blue Agent revises it based on Evidence.",
            source_id="s_blue",
            source_title="Red-Blue process",
            source_url="local://blue",
        ),
        Evidence(
            id="ev_verify",
            claim="Verifier checks after revision and marks uncertainty.",
            quote="Verifier performs verification after revision and marks remaining uncertainty.",
            source_id="s_verify",
            source_title="Verification process",
            source_url="local://verify",
        ),
    ]

    diagnostics = assess_methodology_obligations(
        query,
        [
            ("Blue 根据 Evidence 修订报告。", ["ev_blue"]),
            ("Verifier 在修订后重新核验，并标记仍然存在的不确定性。", ["ev_verify"]),
        ],
        evidence,
    )

    assert [item.obligation_type for item in diagnostics] == [
        "evidence_grounded_blue_revision",
        "post_revision_verification",
    ]
    assert [item.status for item in diagnostics] == ["satisfied", "satisfied"]


def test_red_blue_process_obligation_does_not_invent_unsupported_workflow_detail():
    query = "Red-Blue 对抗核查如何改进深度研究报告？"
    evidence = [
        Evidence(
            id="ev_blue",
            claim="Blue revises the report based on evidence.",
            quote="Blue Agent revises the report based on Evidence.",
            source_id="s_blue",
            source_title="Red-Blue process",
            source_url="local://blue",
        )
    ]

    diagnostics = assess_methodology_obligations(query, [], evidence)

    assert diagnostics[0].status == "missing"
    assert diagnostics[1].status == "evidence_insufficient"

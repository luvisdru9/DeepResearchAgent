from deepresearch.evaluation.claims import assess_claim_citations, extract_atomic_claims
from deepresearch.schema import Evidence, ResearchReport


def _report(finding: str) -> ResearchReport:
    evidence = [
        Evidence(
            id="ev_1",
            claim="DAG 表达依赖关系。",
            quote="无依赖节点可以并发执行。",
            source_id="doc_1",
            source_title="Source 1",
            source_url="https://one.example/source",
        ),
        Evidence(
            id="ev_2",
            claim="共享记忆保存证据。",
            quote="状态可供多个 Agent 复用。",
            source_id="doc_2",
            source_title="Source 2",
            source_url="https://two.example/source",
        ),
    ]
    return ResearchReport(
        run_id="claim_run",
        query="test",
        title="Report",
        summary="summary",
        key_findings=[finding],
        evidence=evidence,
        markdown=(
            f"# Report\n\n## 关键发现\n- {finding}\n\n## 证据表\n"
            "1. DAG evidence\n2. memory evidence\n"
        ),
    )


def test_atomic_claims_inherit_finding_level_citations_and_keep_mapping():
    report = _report("DAG 表达依赖关系；共享记忆保存证据。 [1] [2]")

    claims = extract_atomic_claims(report)

    assert [item.text for item in claims] == ["DAG 表达依赖关系；", "共享记忆保存证据。"]
    assert claims[0].citation_origin == "finding_inherited"
    assert claims[0].evidence_ids == ["ev_1", "ev_2"]
    assert claims[1].citation_origin == "clause"
    assert claims[1].evidence_ids == ["ev_1", "ev_2"]
    assert claims[0].id.startswith("claim_")


def test_atomic_claims_distinguish_invalid_and_missing_citations():
    report = _report("第一项没有引用；第二项错引。 [3]")

    claims = extract_atomic_claims(report)

    assert claims[0].valid_citation_numbers == []
    assert claims[0].invalid_citation_numbers == [3]
    assert claims[1].invalid_citation_numbers == [3]
    assert all(not item.evidence_ids for item in claims)


def test_claim_assessment_distinguishes_supported_uncited_miscited_and_unsupported():
    supported = assess_claim_citations(_report("DAG 表达依赖关系。 [1]"))[0]
    uncited = assess_claim_citations(_report("DAG 表达依赖关系。"))[0]
    miscited = assess_claim_citations(_report("DAG 表达依赖关系。 [3]"))[0]
    unsupported = assess_claim_citations(_report("月球完全由奶酪构成。 [1]"))[0]

    assert supported.status == "supported"
    assert uncited.status == "uncited"
    assert miscited.status == "miscited"
    assert unsupported.status == "unsupported"

from deepresearch.quality import validate_report_citations
from deepresearch.schema import Evidence, ResearchReport


def _report(*, finding: str = "结论有证据支持。 [1]", markdown_number: int = 1) -> ResearchReport:
    evidence = Evidence(
        id="ev_1",
        claim="结论有证据支持。",
        quote="可追溯证据支持该结论。",
        source_id="doc_1",
        source_title="Source",
        source_url="https://example.com/source",
    )
    return ResearchReport(
        run_id="run_citation",
        query="test",
        title="Citation report",
        summary="summary",
        key_findings=[finding],
        evidence=[evidence],
        markdown=(
            "# Citation report\n\n"
            "## 关键发现\n"
            f"- {finding}\n\n"
            "## 证据表\n"
            f"{markdown_number}. 可追溯证据支持该结论。  \n"
            "   来源：[Source](https://example.com/source)\n\n"
            "## 局限性\n"
            "- none\n"
        ),
    )


def test_citation_validator_returns_stable_number_to_evidence_mapping():
    result = validate_report_citations(_report())

    assert result.valid is True
    assert result.citation_map == {1: "ev_1"}
    assert result.cited_numbers == [1]
    assert result.evidence_table_entry_count == 1


def test_citation_validator_reports_missing_and_out_of_range_citations():
    invalid_number = validate_report_citations(_report(finding="错误编号。 [2]"))
    uncited = validate_report_citations(_report(finding="没有编号。"))
    malformed_table = validate_report_citations(_report(markdown_number=2))

    assert invalid_number.valid is False
    assert invalid_number.invalid_numbers == [2]
    assert uncited.uncited_finding_indexes == [1]
    assert malformed_table.valid is False
    assert "expected [1], got [2]" in malformed_table.issues[0]


def test_citation_validator_ignores_indented_numbered_lists_inside_evidence_quotes():
    report = _report()
    report = report.model_copy(
        update={
            "markdown": report.markdown.replace(
                "1. 可追溯证据支持该结论。  ",
                "1. 指标包含：\n   1. precision\n   2. recall  ",
            )
        }
    )

    result = validate_report_citations(report)

    assert result.valid is True
    assert result.evidence_table_entry_count == 1

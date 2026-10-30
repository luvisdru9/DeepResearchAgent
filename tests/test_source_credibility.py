from deepresearch.quality import assess_source_credibility
from deepresearch.schema import SourceDoc


def test_source_credibility_records_explainable_institutional_signals():
    doc = SourceDoc(
        id="official",
        title="Official study",
        url="https://agency.gov/research",
        content="A" * 600,
        source_type="web",
        published_at="2026-01-01",
        metadata={"content_origin": "document_loader"},
    )

    assessed = assess_source_credibility(doc)

    assert assessed.credibility is not None
    assert assessed.credibility.tier == "high"
    assert "institutional_domain" in assessed.credibility.signals
    assert "publication_date_available" in assessed.credibility.signals
    assert "full_content_available" in assessed.credibility.signals


def test_source_credibility_marks_short_user_generated_snippet_conservatively():
    doc = SourceDoc(
        id="ugc",
        title="Short post",
        url="https://medium.com/example/post",
        content="short snippet",
        source_type="web",
        metadata={"content_origin": "tavily_snippet"},
    )

    assessed = assess_source_credibility(doc)

    assert assessed.credibility is not None
    assert assessed.credibility.tier == "low"
    assert "user_generated_platform" in assessed.credibility.signals
    assert "snippet_only" in assessed.credibility.signals


def test_source_credibility_does_not_treat_reference_aggregator_as_primary_authority():
    doc = SourceDoc(
        id="secondary",
        title="Indexed paper summary",
        url="https://www.semanticscholar.org/paper/example",
        content="A" * 600,
        source_type="web",
        published_at="2026-01-01",
        metadata={"content_origin": "tavily_raw_content"},
    )

    assessed = assess_source_credibility(doc)

    assert assessed.credibility is not None
    assert assessed.credibility.tier == "medium"
    assert assessed.credibility.score == 0.52
    assert "secondary_reference_platform" in assessed.credibility.signals


def test_source_credibility_marks_primary_research_repository():
    doc = SourceDoc(
        id="paper",
        title="Original paper",
        url="https://arxiv.org/abs/2305.14627",
        content="A" * 600,
        source_type="web",
    )

    assessed = assess_source_credibility(doc)

    assert assessed.credibility is not None
    assert assessed.credibility.tier == "high"
    assert assessed.credibility.score == 0.79
    assert "primary_research_repository" in assessed.credibility.signals


def test_source_credibility_keeps_generic_full_web_page_at_medium():
    doc = SourceDoc(
        id="generic",
        title="Generic web article",
        url="https://example.com/article",
        content="A" * 600,
        source_type="web",
        published_at="2026-01-01",
        metadata={"content_origin": "tavily_raw_content"},
    )

    assessed = assess_source_credibility(doc)

    assert assessed.credibility is not None
    assert assessed.credibility.tier == "medium"
    assert assessed.credibility.score == 0.62

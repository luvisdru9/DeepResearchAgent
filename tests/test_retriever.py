from deepresearch.retrieval.local import LocalSearchProvider


def test_local_search_scores_chinese_query_terms():
    provider = LocalSearchProvider("data/local_corpus")

    docs, diagnostics = provider.search_with_diagnostics("如何评测引用质量和幻觉率？", top_k=2)

    assert docs
    assert diagnostics["ranking"][0]["score"] > 0
    assert any("引用质量" in item["matched_terms"] for item in diagnostics["ranking"])

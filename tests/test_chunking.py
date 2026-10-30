from deepresearch.retrieval import TextChunker
from deepresearch.schema import SourceChunk, SourceDoc


def test_text_chunker_preserves_exact_offsets_and_overlap():
    content = ("第一段用于验证字符偏移。" * 20) + "\n\n" + ("第二段继续提供内容。" * 20)
    doc = SourceDoc(id="doc_a", title="A", url="https://example.com/a", content=content)
    chunker = TextChunker(max_chars=256, overlap_chars=32, max_chunks_per_doc=4)

    chunks = chunker.chunk_document(doc)

    assert len(chunks) >= 2
    assert all(chunk.text == content[chunk.start_char : chunk.end_char] for chunk in chunks)
    assert all(len(chunk.text) <= 256 for chunk in chunks)
    assert chunks[1].start_char <= chunks[0].end_char
    assert chunks[0].source_id == doc.id


def test_text_chunker_interleaves_sources_for_reader():
    docs = [
        SourceDoc(id="a", title="A", url="https://example.com/a", content="A" * 700),
        SourceDoc(id="b", title="B", url="https://example.com/b", content="B" * 700),
    ]
    chunker = TextChunker(max_chars=256, overlap_chars=16, max_chunks_per_doc=3)

    chunks = chunker.chunk_documents(docs)
    reader_docs = chunker.to_reader_docs(docs, chunks)

    assert [chunk.source_id for chunk in chunks[:4]] == ["a", "b", "a", "b"]
    assert reader_docs[0].id == "a"
    assert reader_docs[0].metadata["chunk_id"] == chunks[0].id
    assert reader_docs[0].content == chunks[0].text


def test_chunk_ranking_promotes_metric_sections_without_losing_source_diversity():
    chunks = [
        SourceChunk(
            id="paper_a_chunk_0",
            source_id="paper_a",
            source_title="Paper A",
            source_url="https://example.test/a",
            text="Introduction and general background only.",
            chunk_index=0,
            start_char=0,
            end_char=41,
        ),
        SourceChunk(
            id="paper_b_chunk_0",
            source_id="paper_b",
            source_title="Paper B",
            source_url="https://example.test/b",
            text="Overview of the research problem.",
            chunk_index=0,
            start_char=0,
            end_char=33,
        ),
        SourceChunk(
            id="paper_a_chunk_1",
            source_id="paper_a",
            source_title="Paper A",
            source_url="https://example.test/a",
            text="Evaluation metrics include citation precision, recall, and accuracy.",
            chunk_index=1,
            start_char=42,
            end_char=109,
        ),
        SourceChunk(
            id="paper_b_chunk_1",
            source_id="paper_b",
            source_title="Paper B",
            source_url="https://example.test/b",
            text="The dataset scope, benchmark tasks, and limitations are documented.",
            chunk_index=1,
            start_char=34,
            end_char=101,
        ),
    ]

    ranked = TextChunker.rank_for_query(
        chunks,
        "比较数据、评测指标、适用范围和局限",
    )

    assert [chunk.id for chunk in ranked[:2]] == ["paper_a_chunk_1", "paper_b_chunk_1"]
    assert [chunk.source_id for chunk in ranked[:2]] == ["paper_a", "paper_b"]


def test_query_chunking_scans_late_pdf_sections_before_applying_limit():
    content = (
        ("Background material without measurements. " * 40)
        + ("Method discussion without results. " * 40)
        + "Evaluation results: Precision 82.1 Recall 79.4 F1 80.7."
    )
    doc = SourceDoc(
        id="paper",
        title="Long paper",
        url="https://example.com/paper.pdf",
        content=content,
    )
    chunker = TextChunker(max_chars=256, overlap_chars=16, max_chunks_per_doc=2)

    bounded_opening = chunker.chunk_document(doc)
    selected = chunker.chunk_documents_for_query(
        [doc],
        "What precision recall and F1 results were reported?",
    )

    assert all("F1 80.7" not in chunk.text for chunk in bounded_opening)
    assert any("F1 80.7" in chunk.text for chunk in selected)
    assert len(selected) == 2


def test_chunk_ranking_promotes_cross_language_metric_definition_facets():
    chunks = [
        SourceChunk(
            id="paper_chunk_0",
            source_id="paper",
            source_title="Paper",
            source_url="https://example.test/paper",
            text="The benchmark reports strong citation quality on one dataset.",
            chunk_index=0,
            start_char=0,
            end_char=61,
        ),
        SourceChunk(
            id="paper_chunk_1",
            source_id="paper",
            source_title="Paper",
            source_url="https://example.test/paper",
            text=(
                "Each statement cites supporting passages. Citation quality asks whether the answer "
                "is well supported by the cited passages and no irrelevant passages are cited."
            ),
            chunk_index=1,
            start_char=62,
            end_char=220,
        ),
    ]

    ranked = TextChunker.rank_for_query(chunks, "引用覆盖率、引用准确性和归因一致性")

    assert ranked[0].id == "paper_chunk_1"


def test_chunk_ranking_prioritizes_hallucination_support_taxonomy_over_generic_terms():
    chunks = [
        SourceChunk(
            id="ragtruth_generic",
            source_id="ragtruth",
            source_title="RAGTruth",
            source_url="https://example.test/ragtruth",
            text=(
                "The static method evaluates hallucination detection on multiple datasets and "
                "reports benchmark performance for language models."
            ),
            chunk_index=2,
            start_char=0,
            end_char=130,
        ),
        SourceChunk(
            id="ragtruth_taxonomy",
            source_id="ragtruth",
            source_title="RAGTruth",
            source_url="https://example.test/ragtruth",
            text=(
                "Annotators mark unsupported statements and contradictory claims by checking "
                "the support relation against a supporting passage."
            ),
            chunk_index=3,
            start_char=131,
            end_char=260,
        ),
    ]

    ranked = TextChunker.rank_for_query(chunks, "如何定义并评测原子主张层的幻觉率？")

    assert ranked[0].id == "ragtruth_taxonomy"

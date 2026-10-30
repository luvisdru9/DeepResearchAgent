"""Deterministic source chunking with exact character offsets."""

from __future__ import annotations

from deepresearch.quality.evidence_facets import chunk_facet_match_count
from deepresearch.schema import SourceChunk, SourceDoc
from deepresearch.text import tokenize_mixed_text


_RESEARCH_SECTION_TERMS = (
    "benchmark",
    "corpus",
    "dataset",
    "evaluation",
    "experiment",
    "limitation",
    "metric",
    "precision",
    "recall",
    "scope",
    "基准",
    "局限",
    "范围",
    "评估",
    "评测",
    "实验",
    "数据",
    "指标",
    "精确率",
    "召回率",
    "适用",
)
_OPERATIONAL_DEFINITION_TERMS = (
    "recall = 1 if",
    "precision = 1 if",
    "fully supports the segment",
    "evident conflict",
    "subtle conflict",
    "evident introduction",
    "subtle introduction",
    "numerator",
    "denominator",
    "analysis unit",
    "分子",
    "分母",
    "分析单位",
)


class TextChunker:
    """Split source text into bounded overlapping spans without rewriting source text."""

    def __init__(
        self,
        *,
        max_chars: int = 2500,
        overlap_chars: int = 200,
        max_chunks_per_doc: int = 4,
    ) -> None:
        self.max_chars = max(256, max_chars)
        # overlap 必须小于窗口，否则游标无法前进。
        self.overlap_chars = min(max(0, overlap_chars), self.max_chars - 1)
        self.max_chunks_per_doc = max(1, max_chunks_per_doc)

    def chunk_document(self, doc: SourceDoc) -> list[SourceChunk]:
        """Create exact source spans, preferring nearby sentence or line boundaries."""

        return self._chunk_document(doc, max_chunks=self.max_chunks_per_doc)

    def _chunk_document(
        self,
        doc: SourceDoc,
        *,
        max_chunks: int | None,
    ) -> list[SourceChunk]:
        """Internal exact-offset splitter; ``None`` scans the complete document."""

        content = doc.content
        if not content:
            return []
        chunks: list[SourceChunk] = []
        start = 0
        while start < len(content) and (
            max_chunks is None or len(chunks) < max_chunks
        ):
            hard_end = min(len(content), start + self.max_chars)
            end = self._preferred_end(content, start, hard_end)
            if end <= start:
                end = hard_end
            index = len(chunks)
            chunks.append(
                SourceChunk(
                    id=f"{doc.id}_chunk_{index}",
                    source_id=doc.id,
                    source_title=doc.title,
                    source_url=doc.url,
                    text=content[start:end],
                    chunk_index=index,
                    start_char=start,
                    end_char=end,
                    metadata={"source_type": doc.source_type},
                )
            )
            if end >= len(content):
                break
            start = max(start + 1, end - self.overlap_chars)
        return chunks

    def chunk_documents(self, docs: list[SourceDoc]) -> list[SourceChunk]:
        """Interleave document chunks so one long source cannot monopolize Reader capacity."""

        per_doc = [self.chunk_document(doc) for doc in docs]
        interleaved: list[SourceChunk] = []
        for index in range(max((len(chunks) for chunks in per_doc), default=0)):
            interleaved.extend(chunks[index] for chunks in per_doc if index < len(chunks))
        return interleaved

    def chunk_documents_for_query(
        self,
        docs: list[SourceDoc],
        query: str,
    ) -> list[SourceChunk]:
        """Scan complete documents, then retain a bounded relevant set per source.

        ``chunk_document`` keeps its historical bounded behavior for callers that
        do not have a query. Research Pipeline uses this method so a metrics or
        results section near the end of a paper can outrank its introduction.
        """

        all_chunks: list[SourceChunk] = []
        for doc in docs:
            all_chunks.extend(self._chunk_document(doc, max_chunks=None))
        ranked = self.rank_for_query(all_chunks, query)
        selected: list[SourceChunk] = []
        selected_per_source: dict[str, int] = {}
        for chunk in ranked:
            count = selected_per_source.get(chunk.source_id, 0)
            if count >= self.max_chunks_per_doc:
                continue
            selected.append(chunk)
            selected_per_source[chunk.source_id] = count + 1
        return selected

    @staticmethod
    def rank_for_query(chunks: list[SourceChunk], query: str) -> list[SourceChunk]:
        """Prioritize relevant paper sections while retaining source-level round robin.

        Reader 的调用上限通常小于总 chunk 数。若固定从正文开头读取，摘要和导航会占满
        预算，方法、指标与局限章节永远进不了 Reader。这里仅改变 chunk 顺序，不改写文本、
        ID 或字符偏移；每个来源仍轮流贡献一个候选，避免单篇长文垄断上下文。
        """

        if not chunks:
            return []
        query_terms = tokenize_mixed_text(query)

        def definition_terms(chunk: SourceChunk) -> set[str]:
            text = chunk.text.casefold()
            return {
                term.casefold()
                for term in _OPERATIONAL_DEFINITION_TERMS
                if term.casefold() in text
            }

        def score(chunk: SourceChunk) -> tuple[int, int]:
            text = chunk.text.casefold()
            chunk_terms = tokenize_mixed_text(text)
            lexical_overlap = len(query_terms.intersection(chunk_terms))
            section_hits = sum(term.casefold() in text for term in _RESEARCH_SECTION_TERMS)
            definition_hits = sum(
                term.casefold() in text for term in _OPERATIONAL_DEFINITION_TERMS
            )
            # 中英文查询与英文论文之间可能没有词面重合。对明确要求指标定义、
            # 分母或错误分类的任务，使用小型概念别名表提升真正的定义段，
            # 避免摘要中的泛化结果占满 Reader 的有限调用预算。
            facet_hits = chunk_facet_match_count(query, text)
            # 每个 facet 命中代表查询明确要求的“定义槽位”。其权重需要高于摘要中
            # 多个泛化词的偶然重合，否则有限 Reader 窗口仍会错过分母、分类等方法段。
            # 定义短语优先于只重复指标名称的结果表或讨论段。该权重只影响
            # 已由 query/facet 命中的候选顺序，不生成或改写任何证据内容。
            return (
                lexical_overlap * 3
                + section_hits
                + facet_hits * 20
                + definition_hits * 12,
                -chunk.chunk_index,
            )

        source_order: list[str] = []
        per_source: dict[str, list[SourceChunk]] = {}
        for chunk in chunks:
            if chunk.source_id not in per_source:
                source_order.append(chunk.source_id)
                per_source[chunk.source_id] = []
            per_source[chunk.source_id].append(chunk)
        for source_id, source_chunks in per_source.items():
            # 仅按相关性取 Top-K 会反复选中多个包含同一术语的段落，并把紧邻的
            # 定义续段挤出 Reader 预算。这里做确定性的定义槽位去重：新定义加分，
            # 已覆盖定义等量扣分。它不理解或生成结论，只扩大有限窗口的概念覆盖。
            remaining = list(source_chunks)
            diversified: list[SourceChunk] = []
            covered_definition_terms: set[str] = set()
            base_scores = {chunk.id: score(chunk) for chunk in source_chunks}
            term_sets = {chunk.id: definition_terms(chunk) for chunk in source_chunks}
            while remaining:
                best = max(
                    remaining,
                    key=lambda chunk: (
                        base_scores[chunk.id][0]
                        + 12
                        * len(term_sets[chunk.id] - covered_definition_terms)
                        - 12
                        * len(term_sets[chunk.id] & covered_definition_terms),
                        base_scores[chunk.id][0],
                        base_scores[chunk.id][1],
                    ),
                )
                remaining.remove(best)
                diversified.append(best)
                covered_definition_terms.update(term_sets[best.id])
            per_source[source_id] = diversified

        ranked: list[SourceChunk] = []
        for index in range(max(len(items) for items in per_source.values())):
            ranked.extend(
                per_source[source_id][index]
                for source_id in source_order
                if index < len(per_source[source_id])
            )
        return ranked

    @staticmethod
    def to_reader_docs(docs: list[SourceDoc], chunks: list[SourceChunk]) -> list[SourceDoc]:
        """Expose chunks through the existing Reader interface while preserving source identity."""

        doc_by_id = {doc.id: doc for doc in docs}
        reader_docs: list[SourceDoc] = []
        for chunk in chunks:
            source = doc_by_id[chunk.source_id]
            reader_docs.append(
                source.model_copy(
                    update={
                        "content": chunk.text,
                        "metadata": {
                            **source.metadata,
                            "chunk_id": chunk.id,
                            "chunk_index": chunk.chunk_index,
                            "chunk_start_char": chunk.start_char,
                            "chunk_end_char": chunk.end_char,
                        },
                    }
                )
            )
        return reader_docs

    def _preferred_end(self, content: str, start: int, hard_end: int) -> int:
        if hard_end >= len(content):
            return hard_end
        # 至少保留半个窗口，避免在很靠前的标点处产生碎片。
        lower_bound = start + self.max_chars // 2
        candidates = [
            content.rfind(separator, lower_bound, hard_end)
            for separator in ("\n\n", "\n", "。", "！", "？", ". ", "! ", "? ")
        ]
        boundary = max(candidates, default=-1)
        if boundary < lower_bound:
            return hard_end
        return boundary + 1

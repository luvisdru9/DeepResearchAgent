"""SearchProvider adapter for historical Evidence stored in SQLite."""

from __future__ import annotations

import hashlib

import numpy as np

from deepresearch.memory.embeddings import EmbeddingProvider, HashingEmbeddingProvider
from deepresearch.memory.store import SQLiteMemoryStore
from deepresearch.memory.vector_index import EvidenceVectorIndex
from deepresearch.retrieval.base import SearchProvider
from deepresearch.schema import SourceDoc
from deepresearch.text import tokenize_mixed_text


class EvidenceMemorySearchProvider(SearchProvider):
    """Rebuild a small local vector index and return traceable historical evidence."""

    def __init__(
        self,
        store: SQLiteMemoryStore,
        *,
        dimensions: int = 512,
        min_similarity: float = 0.05,
        max_results: int = 5,
        embedding_provider: EmbeddingProvider | None = None,
        cache_enabled: bool = True,
        scope_by_domain: bool = True,
        scope_max_age_days: int | None = 3650,
        scope_min_query_overlap: float = 0.05,
    ) -> None:
        self.store = store
        self.embedding_provider = embedding_provider or HashingEmbeddingProvider(
            dimensions=dimensions
        )
        self.dimensions = self.embedding_provider.dimensions
        self.min_similarity = min_similarity
        self.max_results = max_results
        self.cache_enabled = cache_enabled
        self.scope_by_domain = scope_by_domain
        self.scope_max_age_days = scope_max_age_days
        self.scope_min_query_overlap = scope_min_query_overlap
        self._active_domain: str | None = None
        self._active_task_query: str | None = None
        self._active_run_id: str | None = None

    def set_scope(
        self,
        *,
        domain: str | None,
        task_query: str,
        current_run_id: str,
    ) -> None:
        """Bind task-level filters before DAG nodes issue their subtask queries."""

        self._active_domain = domain
        self._active_task_query = task_query
        self._active_run_id = current_run_id

    def search(self, query: str, top_k: int = 5) -> list[SourceDoc]:
        docs, _ = self.search_with_diagnostics(query, top_k=top_k)
        return docs

    def search_with_diagnostics(
        self, query: str, top_k: int = 5
    ) -> tuple[list[SourceDoc], dict[str, object]]:
        all_records = self.store.load_all_evidence(
            domain=self._active_domain if self.scope_by_domain else None,
            max_age_days=self.scope_max_age_days,
            exclude_run_id=self._active_run_id,
        )
        records = [
            record
            for record in all_records
            if self._query_overlap(record.task_query, self._active_task_query or query)
            >= self.scope_min_query_overlap
        ]
        evidence = [record.evidence for record in records]
        vectors: list[np.ndarray] = []
        cache_hits = 0
        cache_misses = 0
        for record in records:
            item = record.evidence
            text = f"{item.claim} {item.quote} {item.source_title}"
            content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            cached = (
                self.store.load_evidence_vector(
                    run_id=record.run_id,
                    evidence_id=item.id,
                    provider=self.embedding_provider.provider_name,
                    model=self.embedding_provider.model,
                    version=self.embedding_provider.version,
                    content_hash=content_hash,
                )
                if self.cache_enabled
                else None
            )
            if cached is not None and cached[0] == self.dimensions:
                vector = np.frombuffer(cached[1], dtype=np.float32).copy()
                if vector.shape == (self.dimensions,):
                    vectors.append(vector)
                    cache_hits += 1
                    continue
            vector = self.embedding_provider.embed([text])[0]
            vectors.append(vector)
            cache_misses += 1
            if self.cache_enabled:
                self.store.save_evidence_vector(
                    run_id=record.run_id,
                    evidence_id=item.id,
                    provider=self.embedding_provider.provider_name,
                    model=self.embedding_provider.model,
                    version=self.embedding_provider.version,
                    dimensions=self.dimensions,
                    content_hash=content_hash,
                    vector=vector.astype(np.float32).tobytes(),
                )
        index = EvidenceVectorIndex(embedding_provider=self.embedding_provider)
        matrix = (
            np.vstack(vectors)
            if vectors
            else np.empty((0, self.dimensions), dtype=np.float32)
        )
        index.build_from_vectors(evidence, matrix)
        effective_top_k = min(top_k, self.max_results)
        matches = index.search(
            query,
            top_k=effective_top_k,
            min_similarity=self.min_similarity,
        )
        docs: list[SourceDoc] = []
        ranking: list[dict[str, object]] = []
        for rank, match in enumerate(matches, start=1):
            record = records[match.index]
            item = match.evidence
            source_id = "memory_" + hashlib.sha256(
                f"{record.run_id}:{item.id}".encode("utf-8")
            ).hexdigest()[:16]
            docs.append(
                SourceDoc(
                    id=source_id,
                    title=item.source_title,
                    url=item.source_url,
                    content=f"{item.claim}\n\n原文证据：{item.quote}",
                    source_type="memory",
                    score=match.score,
                    metadata={
                        "provider": "memory",
                        "originating_run_id": record.run_id,
                        "evidence_id": item.id,
                        "original_source_id": item.source_id,
                        "extraction_method": item.extraction_method,
                        "confidence": item.confidence,
                    },
                )
            )
            ranking.append(
                {
                    "rank": rank,
                    "source_id": source_id,
                    "source_title": item.source_title,
                    "score": match.score,
                    "originating_run_id": record.run_id,
                    "evidence_id": item.id,
                    "returned": True,
                }
            )
        return docs, {
            "provider": "memory",
            "index_type": "numpy_hashing",
            "embedding_provider": self.embedding_provider.provider_name,
            "embedding_model": self.embedding_provider.model,
            "embedding_version": self.embedding_provider.version,
            "vector_dimensions": self.dimensions,
            "embedding_cache_enabled": self.cache_enabled,
            "embedding_cache_hits": cache_hits,
            "embedding_cache_misses": cache_misses,
            "indexed_evidence_count": len(records),
            "scope_candidate_count": len(all_records),
            "scope_filtered_count": len(all_records) - len(records),
            "scope_domain": self._active_domain if self.scope_by_domain else None,
            "scope_max_age_days": self.scope_max_age_days,
            "scope_min_query_overlap": self.scope_min_query_overlap,
            "returned_docs_count": len(docs),
            "min_similarity": self.min_similarity,
            "ranking": ranking,
        }

    @staticmethod
    def _query_overlap(left: str, right: str) -> float:
        left_terms = tokenize_mixed_text(left)
        right_terms = tokenize_mixed_text(right)
        if not left_terms or not right_terms:
            return 0.0
        return len(left_terms.intersection(right_terms)) / len(left_terms.union(right_terms))

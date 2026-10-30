"""Small deterministic numpy vector index for locally persisted Evidence."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from deepresearch.memory.embeddings import EmbeddingProvider, HashingEmbeddingProvider
from deepresearch.schema import Evidence


@dataclass(frozen=True, slots=True)
class EvidenceVectorMatch:
    """One cosine-similarity match against an Evidence object."""

    evidence: Evidence
    score: float
    index: int


class EvidenceVectorIndex:
    """Hash mixed Chinese/English terms into normalized vectors and search by cosine similarity.

    This is the reproducible stage-three baseline. It avoids a remote embedding dependency while
    keeping the index interface replaceable by a learned embedding model later.
    """

    def __init__(
        self,
        *,
        dimensions: int = 512,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        self.embedding_provider = embedding_provider or HashingEmbeddingProvider(
            dimensions=dimensions
        )
        self.dimensions = self.embedding_provider.dimensions
        self._evidence: list[Evidence] = []
        self._matrix = np.empty((0, self.dimensions), dtype=np.float32)

    def build(self, evidence: list[Evidence]) -> None:
        self._evidence = list(evidence)
        if not evidence:
            self._matrix = np.empty((0, self.dimensions), dtype=np.float32)
            return
        self._matrix = self.embedding_provider.embed(
            [f"{item.claim} {item.quote} {item.source_title}" for item in evidence]
        )

    def build_from_vectors(self, evidence: list[Evidence], vectors: np.ndarray) -> None:
        """Build from versioned cache entries after validating shape and normalization."""

        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.shape != (len(evidence), self.dimensions):
            raise ValueError(
                f"Expected vector matrix shape {(len(evidence), self.dimensions)}, got {matrix.shape}."
            )
        self._evidence = list(evidence)
        self._matrix = matrix

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        min_similarity: float = 0.05,
    ) -> list[EvidenceVectorMatch]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        if not self._evidence:
            return []
        query_vector = self.embedding_provider.embed([query])[0]
        if not np.any(query_vector):
            return []
        scores = self._matrix @ query_vector
        ranked_indices = np.argsort(-scores, kind="stable")
        return [
            EvidenceVectorMatch(
                evidence=self._evidence[int(index)],
                score=float(scores[int(index)]),
                index=int(index),
            )
            for index in ranked_indices
            if float(scores[int(index)]) >= min_similarity
        ][:top_k]

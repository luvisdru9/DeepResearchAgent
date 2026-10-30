"""Shared state, evidence persistence, and context compression."""

from deepresearch.memory.compression import CompressionResult, EvidenceCompressor
from deepresearch.memory.store import SQLiteMemoryStore
from deepresearch.memory.vector_index import EvidenceVectorIndex, EvidenceVectorMatch
from deepresearch.memory.snapshot import EvidenceSnapshotStore
from deepresearch.memory.embeddings import (
    EmbeddingProvider,
    HashingEmbeddingProvider,
    OpenAICompatibleEmbeddingProvider,
    create_embedding_provider,
)

__all__ = [
    "CompressionResult",
    "EvidenceCompressor",
    "EvidenceVectorIndex",
    "EvidenceVectorMatch",
    "EvidenceSnapshotStore",
    "EmbeddingProvider",
    "HashingEmbeddingProvider",
    "OpenAICompatibleEmbeddingProvider",
    "create_embedding_provider",
    "SQLiteMemoryStore",
]

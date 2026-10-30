"""Deterministic offline retrieval over one validated benchmark snapshot."""

from __future__ import annotations

from pathlib import Path

from deepresearch.retrieval.base import SearchProvider
from deepresearch.retrieval.snapshot import load_frozen_snapshot
from deepresearch.schema import SourceDoc
from deepresearch.text import tokenize_mixed_text


class FrozenSearchProvider(SearchProvider):
    """Replay a fixed source universe without network or mutable corpus access."""

    def __init__(self, manifest_path: str | Path) -> None:
        snapshot = load_frozen_snapshot(manifest_path)
        self.manifest_path = snapshot.manifest_path
        self.manifest = snapshot.manifest
        self._documents = {document.id: document for document in snapshot.documents}

    def search(self, query: str, top_k: int = 5) -> list[SourceDoc]:
        docs, _ = self.search_with_diagnostics(query, top_k=top_k)
        return docs

    def search_with_diagnostics(
        self,
        query: str,
        top_k: int = 5,
    ) -> tuple[list[SourceDoc], dict[str, object]]:
        """Select the closest recorded query, then preserve its frozen result order."""

        query_terms = tokenize_mixed_text(query)
        ranked_queries = []
        for record in self.manifest.queries:
            matched = sorted(query_terms.intersection(tokenize_mixed_text(record.query)))
            ranked_queries.append((len(matched), record.id, matched, record))
        ranked_queries.sort(key=lambda item: (-item[0], item[1]))
        _, _, matched_terms, selected_query = ranked_queries[0]

        selected_ids = selected_query.result_source_ids[:top_k]
        docs = [
            self._documents[source_id].model_copy(
                update={
                    "score": float(len(selected_ids) - index),
                    "metadata": {
                        **self._documents[source_id].metadata,
                        "provider": "frozen",
                        "snapshot_id": self.manifest.snapshot_id,
                        "snapshot_content_hash": self.manifest.content_hash,
                        "frozen_query_id": selected_query.id,
                    },
                }
            )
            for index, source_id in enumerate(selected_ids)
        ]
        diagnostics = {
            "provider": "frozen",
            "offline": True,
            "snapshot_id": self.manifest.snapshot_id,
            "snapshot_content_hash": self.manifest.content_hash,
            "manifest_path": str(self.manifest_path),
            "frozen_query_id": selected_query.id,
            "matched_terms": matched_terms,
            "candidate_docs_count": len(self._documents),
            "returned_docs_count": len(docs),
            "ranking": [
                {
                    "rank": rank,
                    "source_id": source_id,
                    "source_title": self._documents[source_id].title,
                    "score": float(len(selected_ids) - rank + 1),
                    "matched_terms": matched_terms,
                    "returned": True,
                }
                for rank, source_id in enumerate(selected_ids, start=1)
            ],
        }
        return docs, diagnostics

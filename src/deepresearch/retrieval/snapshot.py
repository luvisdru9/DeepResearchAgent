"""Versioned, hash-verified source snapshots shared by retrieval and evaluation."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from deepresearch.schema import SourceDoc


class FrozenQueryRecord(BaseModel):
    """One manually selected query and its deterministic result order."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.-]+$")
    query: str = Field(min_length=1)
    result_source_ids: list[str] = Field(min_length=1)


class FrozenDocumentRecord(BaseModel):
    """Manifest pointer to one serialized SourceDoc and its exact-byte hash."""

    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(min_length=1)
    path: str = Field(min_length=1)
    content_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")


class FrozenSnapshotManifest(BaseModel):
    """Immutable contract binding queries, result order, and source bytes."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    snapshot_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.-]+$")
    sample_id: str = Field(min_length=1)
    benchmark_version: str = Field(min_length=1)
    captured_at: datetime
    provider: str = Field(min_length=1)
    queries: list[FrozenQueryRecord] = Field(min_length=1)
    documents: list[FrozenDocumentRecord] = Field(min_length=1)
    content_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")


class FrozenSnapshot:
    """Validated manifest plus the SourceDocs it freezes."""

    def __init__(
        self,
        manifest_path: Path,
        manifest: FrozenSnapshotManifest,
        documents: list[SourceDoc],
    ) -> None:
        self.manifest_path = manifest_path
        self.manifest = manifest
        self.documents = documents


def sha256_bytes(payload: bytes) -> str:
    """Return the canonical prefixed SHA-256 used by benchmark artifacts."""

    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def canonical_json_bytes(payload: Any) -> bytes:
    """Serialize hash inputs deterministically across runs and platforms."""

    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def snapshot_content_hash(manifest: FrozenSnapshotManifest | dict[str, Any]) -> str:
    """Hash all manifest semantics except the self-referential hash field."""

    if isinstance(manifest, FrozenSnapshotManifest):
        payload = manifest.model_dump(mode="json", exclude={"content_hash"})
    else:
        payload = {key: value for key, value in manifest.items() if key != "content_hash"}
    return sha256_bytes(canonical_json_bytes(payload))


def load_frozen_snapshot(manifest_path: str | Path) -> FrozenSnapshot:
    """Load a snapshot and reject manifest, path, or document tampering."""

    path = Path(manifest_path).resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    manifest = FrozenSnapshotManifest.model_validate(raw)
    expected_manifest_hash = snapshot_content_hash(manifest)
    if manifest.content_hash != expected_manifest_hash:
        raise ValueError(
            f"Frozen snapshot manifest hash mismatch for '{manifest.snapshot_id}'."
        )

    documents: list[SourceDoc] = []
    seen_source_ids: set[str] = set()
    snapshot_root = path.parent
    for record in manifest.documents:
        document_path = (snapshot_root / record.path).resolve()
        # manifest 中的相对路径不得逃逸快照目录，防止意外读取外部文件。
        if not document_path.is_relative_to(snapshot_root):
            raise ValueError(f"Frozen document path escapes snapshot root: {record.path}")
        payload = document_path.read_bytes()
        if sha256_bytes(payload) != record.content_hash:
            raise ValueError(f"Frozen document hash mismatch: {record.source_id}")
        document = SourceDoc.model_validate_json(payload)
        if document.id != record.source_id:
            raise ValueError(
                f"Frozen document id mismatch: expected {record.source_id}, got {document.id}."
            )
        if document.id in seen_source_ids:
            raise ValueError(f"Duplicate frozen source id: {document.id}")
        seen_source_ids.add(document.id)
        documents.append(document)

    referenced_ids = {
        source_id for query in manifest.queries for source_id in query.result_source_ids
    }
    unknown_ids = sorted(referenced_ids - seen_source_ids)
    if unknown_ids:
        raise ValueError(f"Frozen queries reference unknown source ids: {unknown_ids}")
    return FrozenSnapshot(path, manifest, documents)


def write_frozen_snapshot(
    destination: str | Path,
    *,
    snapshot_id: str,
    sample_id: str,
    benchmark_version: str,
    captured_at: datetime,
    provider: str,
    queries: list[FrozenQueryRecord],
    documents: list[SourceDoc],
) -> FrozenSnapshotManifest:
    """Create a new snapshot once; published snapshot directories are never overwritten."""

    root = Path(destination)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Frozen snapshot already exists and is immutable: {root}")
    documents_dir = root / "documents"
    documents_dir.mkdir(parents=True, exist_ok=True)

    records: list[FrozenDocumentRecord] = []
    seen_ids: set[str] = set()
    for document in documents:
        if document.id in seen_ids:
            raise ValueError(f"Duplicate frozen source id: {document.id}")
        seen_ids.add(document.id)
        relative_path = Path("documents") / f"{document.id}.json"
        payload = canonical_json_bytes(document.model_dump(mode="json"))
        (root / relative_path).write_bytes(payload)
        records.append(
            FrozenDocumentRecord(
                source_id=document.id,
                path=relative_path.as_posix(),
                content_hash=sha256_bytes(payload),
            )
        )

    draft = {
        "schema_version": "1.0",
        "snapshot_id": snapshot_id,
        "sample_id": sample_id,
        "benchmark_version": benchmark_version,
        "captured_at": captured_at,
        "provider": provider,
        "queries": queries,
        "documents": records,
        "content_hash": "sha256:" + "0" * 64,
    }
    # 先经 Pydantic 统一 datetime/子模型序列化，再计算稳定的整体哈希。
    provisional = FrozenSnapshotManifest.model_validate(draft)
    manifest = provisional.model_copy(update={"content_hash": snapshot_content_hash(provisional)})
    (root / "manifest.json").write_text(
        json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest

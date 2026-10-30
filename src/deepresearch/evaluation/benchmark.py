"""Versioned ResearchBench schemas with explicit legacy compatibility."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class BenchmarkSample(BaseModel):
    """Legacy runner-facing sample kept read-only for existing experiments."""

    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    language: str = "zh"
    domain: str | None = None
    task_type: Literal[
        "fact_qa", "comparative_research", "trend_analysis", "controversy_multihop"
    ] = "fact_qa"
    expected_claims: list[str] = Field(default_factory=list)
    required_sources: list[str] = Field(default_factory=list)
    forbidden_hallucinations: list[str] = Field(default_factory=list)
    judge_rubric: list[str] = Field(default_factory=list)
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    time_range: str | None = None


class AtomicClaimSatisfactionRoute(BaseModel):
    """One complete, self-contained way a report can satisfy an authored gold claim."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.-]+$")
    text: str = Field(min_length=1)


class AtomicClaim(BaseModel):
    """One weighted, independently judgeable answer requirement."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.-]+$")
    text: str = Field(min_length=1)
    satisfaction_routes: list[AtomicClaimSatisfactionRoute] = Field(default_factory=list)
    accepted_variants: list[str] = Field(default_factory=list)
    weight: float = Field(default=1.0, gt=0.0)
    required: bool = True

    @model_validator(mode="after")
    def validate_satisfaction_routes(self) -> "AtomicClaim":
        route_ids = [item.id for item in self.satisfaction_routes]
        if len(route_ids) != len(set(route_ids)):
            raise ValueError(f"Duplicate satisfaction route ids in claim '{self.id}'.")
        return self

    def effective_satisfaction_routes(self) -> list[AtomicClaimSatisfactionRoute]:
        """Return explicit routes or a stable route derived from legacy main claim text."""

        if self.satisfaction_routes:
            return list(self.satisfaction_routes)
        return [
            AtomicClaimSatisfactionRoute(id=f"{self.id}.main", text=self.text)
        ]


class EvidenceRequirement(BaseModel):
    """Source conditions required to support one or more atomic claims."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.-]+$")
    description: str = Field(min_length=1)
    claim_ids: list[str] = Field(min_length=1)
    # 冻结轨道用稳定 SourceDoc ID 计算 required-source recall；实时轨道可留空。
    required_source_ids: list[str] = Field(default_factory=list)
    min_independent_sources: int = Field(default=1, ge=1)
    primary_source_required: bool = False
    accepted_source_types: list[
        Literal[
            "primary",
            "research_paper",
            "official_document",
            "technical_document",
            "dataset",
            "independent_analysis",
        ]
    ] = Field(min_length=1)


class ForbiddenClaim(BaseModel):
    """Known error that the evaluated report must not assert as fact."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.-]+$")
    text: str = Field(min_length=1)
    severity: Literal["low", "medium", "high"] = "high"


class RubricAnchor(BaseModel):
    """Observable scoring anchor for one rubric criterion."""

    model_config = ConfigDict(extra="forbid")

    score: int = Field(ge=0, le=4)
    description: str = Field(min_length=1)


class RubricCriterion(BaseModel):
    """Weighted evaluation dimension with explicit low/high anchors."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.-]+$")
    dimension: Literal[
        "information_recall",
        "analysis_reasoning",
        "citation_quality",
        "task_following",
        "presentation",
    ]
    description: str = Field(min_length=1)
    weight: float = Field(default=1.0, gt=0.0)
    evaluation_mode: Literal["automatic", "human", "hybrid"] = "hybrid"
    anchors: list[RubricAnchor] = Field(min_length=2)

    @model_validator(mode="after")
    def validate_anchors(self) -> "RubricCriterion":
        scores = [item.score for item in self.anchors]
        if len(scores) != len(set(scores)):
            raise ValueError(f"Duplicate rubric anchor scores in '{self.id}'.")
        if 0 not in scores or 4 not in scores:
            raise ValueError(f"Rubric '{self.id}' must define score 0 and score 4 anchors.")
        return self


class SnapshotReference(BaseModel):
    """Immutable source snapshot required by the reproducible frozen track."""

    model_config = ConfigDict(extra="forbid")

    snapshot_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.-]+$")
    manifest_path: str = Field(min_length=1)
    content_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")


class BenchmarkSampleV2(BaseModel):
    """Strict v2 sample with atomic claims, evidence policy, and scoring anchors."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.-]+$")
    question: str = Field(min_length=1)
    language: str = Field(min_length=2)
    domain: str = Field(min_length=1)
    task_type: Literal[
        "fact_qa", "comparative_research", "trend_analysis", "controversy_multihop"
    ]
    difficulty: Literal["easy", "medium", "hard"]
    time_range: str | None = None
    claims: list[AtomicClaim] = Field(min_length=1)
    evidence_requirements: list[EvidenceRequirement] = Field(min_length=1)
    forbidden_claims: list[ForbiddenClaim] = Field(default_factory=list)
    rubric: list[RubricCriterion] = Field(min_length=1)
    snapshot: SnapshotReference | None = None
    annotator: str = Field(min_length=1)
    reviewer: str = Field(min_length=1)
    annotated_at: date
    disagreement_notes: list[str] = Field(default_factory=list)
    status: Literal["draft", "reviewed", "released"] = "draft"

    @model_validator(mode="after")
    def validate_internal_references(self) -> "BenchmarkSampleV2":
        _reject_duplicate_ids(self.claims, owner=f"sample '{self.id}' claims")
        _reject_duplicate_ids(
            self.evidence_requirements,
            owner=f"sample '{self.id}' evidence requirements",
        )
        _reject_duplicate_ids(self.forbidden_claims, owner=f"sample '{self.id}' forbidden claims")
        _reject_duplicate_ids(self.rubric, owner=f"sample '{self.id}' rubric")
        claim_ids = {item.id for item in self.claims}
        unknown = sorted(
            {
                claim_id
                for requirement in self.evidence_requirements
                for claim_id in requirement.claim_ids
                if claim_id not in claim_ids
            }
        )
        if unknown:
            raise ValueError(
                f"Sample '{self.id}' evidence requirements reference unknown claim ids: {unknown}"
            )
        return self

    def to_legacy(self) -> BenchmarkSample:
        """Normalize v2 data for existing runners without exposing reference answers to Pipeline."""

        return BenchmarkSample(
            id=self.id,
            question=self.question,
            language=self.language,
            domain=self.domain,
            task_type=self.task_type,
            expected_claims=[item.text for item in self.claims],
            required_sources=[item.description for item in self.evidence_requirements],
            forbidden_hallucinations=[item.text for item in self.forbidden_claims],
            judge_rubric=[item.description for item in self.rubric],
            difficulty=self.difficulty,
            time_range=self.time_range,
        )


class BenchmarkMetadata(BaseModel):
    """Version and provenance shared by every sample in one dataset release."""

    model_config = ConfigDict(extra="forbid")

    benchmark_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.-]+$")
    schema_version: Literal["2.0"] = "2.0"
    version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    # A metadata/rubric-only revision may reuse immutable snapshot bytes from an
    # earlier release. The explicit base version preserves that provenance.
    snapshot_base_version: str | None = Field(
        default=None,
        pattern=r"^\d+\.\d+\.\d+$",
    )
    track: Literal["frozen", "live_web"]
    released_at: date
    knowledge_cutoff_at: date | None = None
    language_mix: list[str] = Field(min_length=1)
    license: str = Field(min_length=1)
    dataset_card: str = Field(min_length=1)
    change_notes: list[str] = Field(default_factory=list)
    # 发布层级与人工校准状态显式分离。默认值保证历史 v2 数据集无需改写即可加载；
    # 自动构建器必须主动写入 engineering_automatic，不能借用 released 状态冒充正式集。
    release_tier: Literal[
        "development_draft",
        "engineering_automatic",
        "human_calibrated",
    ] = "development_draft"
    human_calibrated: bool = False
    formal_benchmark_eligible: bool = False

    @model_validator(mode="after")
    def validate_release_tier(self) -> "BenchmarkMetadata":
        if self.formal_benchmark_eligible and not self.human_calibrated:
            raise ValueError(
                "formal_benchmark_eligible requires human_calibrated=true."
            )
        if self.release_tier == "engineering_automatic" and (
            self.human_calibrated or self.formal_benchmark_eligible
        ):
            raise ValueError(
                "engineering_automatic datasets cannot claim human calibration or "
                "formal Benchmark eligibility."
            )
        return self


class BenchmarkDatasetV2(BaseModel):
    """Top-level release object; frozen datasets require a snapshot for every sample."""

    model_config = ConfigDict(extra="forbid")

    metadata: BenchmarkMetadata
    samples: list[BenchmarkSampleV2] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_release(self) -> "BenchmarkDatasetV2":
        _reject_duplicate_ids(self.samples, owner="benchmark samples")
        languages = {item.language for item in self.samples}
        declared = set(self.metadata.language_mix)
        if not languages.issubset(declared):
            raise ValueError(
                f"metadata.language_mix is missing sample languages: {sorted(languages - declared)}"
            )
        if self.metadata.track == "frozen":
            missing = [item.id for item in self.samples if item.snapshot is None]
            if missing:
                raise ValueError(f"Frozen benchmark samples require snapshots: {missing}")
            missing_source_annotations = [
                f"{sample.id}:{requirement.id}"
                for sample in self.samples
                for requirement in sample.evidence_requirements
                if not requirement.required_source_ids
            ]
            if missing_source_annotations:
                raise ValueError(
                    "Frozen benchmark evidence requirements need required_source_ids: "
                    f"{missing_source_annotations}"
                )
        return self


class BenchmarkLoadResult(BaseModel):
    """Explicitly identify legacy input while exposing normalized runner samples."""

    legacy: bool
    schema_version: str
    samples: list[BenchmarkSample]
    dataset_v2: BenchmarkDatasetV2 | None = None


def load_benchmark_bundle(path: str | Path) -> BenchmarkLoadResult:
    """Load legacy arrays or a strict v2 release without silently mixing their semantics."""

    benchmark_path = Path(path)
    raw: Any = json.loads(benchmark_path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        samples = [BenchmarkSample.model_validate(item) for item in raw]
        _reject_duplicate_ids(samples, owner="legacy benchmark samples")
        return BenchmarkLoadResult(legacy=True, schema_version="1.0", samples=samples)
    if not isinstance(raw, dict):
        raise ValueError("Benchmark root must be a legacy array or a v2 object.")
    dataset = BenchmarkDatasetV2.model_validate(raw)
    if dataset.metadata.track == "frozen":
        _validate_frozen_snapshot_references(benchmark_path, dataset)
    return BenchmarkLoadResult(
        legacy=False,
        schema_version=dataset.metadata.schema_version,
        samples=[item.to_legacy() for item in dataset.samples],
        dataset_v2=dataset,
    )


def load_benchmark(path: str | Path) -> list[BenchmarkSample]:
    """Return normalized samples for existing evaluation runners."""

    return load_benchmark_bundle(path).samples


def benchmark_v2_json_schema() -> dict[str, Any]:
    """Export the canonical JSON Schema used by authoring and CI validation tools."""

    return BenchmarkDatasetV2.model_json_schema()


def export_benchmark_v2_json_schema(destination: str | Path) -> Path:
    """Write the generated schema to a caller-selected path."""

    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(benchmark_v2_json_schema(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def resolve_snapshot_manifest_path(
    benchmark_path: str | Path,
    snapshot: SnapshotReference,
) -> Path:
    """Resolve a relative manifest while forbidding paths outside the dataset directory."""

    dataset_root = Path(benchmark_path).resolve().parent
    manifest_path = (dataset_root / snapshot.manifest_path).resolve()
    if not manifest_path.is_relative_to(dataset_root):
        raise ValueError(
            f"Frozen snapshot manifest escapes benchmark directory: {snapshot.manifest_path}"
        )
    return manifest_path


def _validate_frozen_snapshot_references(
    benchmark_path: Path,
    dataset: BenchmarkDatasetV2,
) -> None:
    """Verify every frozen sample against the bytes pinned by its manifest."""

    # 局部导入避免 benchmark schema 与 retrieval package 初始化互相依赖。
    from deepresearch.retrieval.snapshot import load_frozen_snapshot

    for sample in dataset.samples:
        assert sample.snapshot is not None  # 已由 BenchmarkDatasetV2 校验。
        manifest_path = resolve_snapshot_manifest_path(benchmark_path, sample.snapshot)
        frozen = load_frozen_snapshot(manifest_path)
        manifest = frozen.manifest
        if manifest.snapshot_id != sample.snapshot.snapshot_id:
            raise ValueError(f"Snapshot id mismatch for benchmark sample '{sample.id}'.")
        if manifest.sample_id != sample.id:
            raise ValueError(f"Snapshot sample id mismatch for benchmark sample '{sample.id}'.")
        expected_snapshot_version = (
            dataset.metadata.snapshot_base_version or dataset.metadata.version
        )
        if manifest.benchmark_version != expected_snapshot_version:
            raise ValueError(f"Snapshot benchmark version mismatch for sample '{sample.id}'.")
        if manifest.content_hash != sample.snapshot.content_hash:
            raise ValueError(f"Snapshot content hash mismatch for benchmark sample '{sample.id}'.")
        snapshot_source_ids = {item.id for item in frozen.documents}
        required_source_ids = {
            source_id
            for requirement in sample.evidence_requirements
            for source_id in requirement.required_source_ids
        }
        unknown_required_ids = sorted(required_source_ids - snapshot_source_ids)
        if unknown_required_ids:
            raise ValueError(
                f"Required source ids are absent from snapshot for sample '{sample.id}': "
                f"{unknown_required_ids}"
            )


def _reject_duplicate_ids(items: list[Any], *, owner: str) -> None:
    identifiers = [str(item.id) for item in items]
    duplicates = sorted({item for item in identifiers if identifiers.count(item) > 1})
    if duplicates:
        raise ValueError(f"Duplicate benchmark ids in {owner}: {duplicates}")

"""Resumable construction of the forty-sample engineering ResearchBench.

The builder deliberately remains separate from the fixed eight-sample Track A
builder.  It persists one validated sample at a time, so an interrupted network
build can resume without repeating already accepted LLM and retrieval calls.
"""

from __future__ import annotations

from collections import Counter
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, model_validator

from deepresearch.config import AppConfig
from deepresearch.agents.reader import _ground_quote
from deepresearch.evaluation.auto_draft import (
    AutoDraftBuilder,
    AutoDraftSampleResult,
    _sample_requires_primary,
)
from deepresearch.evaluation.benchmark import (
    BenchmarkDatasetV2,
    BenchmarkMetadata,
    BenchmarkSample,
    BenchmarkSampleV2,
    RubricAnchor,
    RubricCriterion,
    benchmark_v2_json_schema,
    load_benchmark,
)
from deepresearch.llm import LLMClient
from deepresearch.retrieval.base import SearchProvider
from deepresearch.retrieval.snapshot import load_frozen_snapshot
from deepresearch.schema import SourceDoc


RESEARCHBENCH40_TASK_TYPES = (
    "fact_qa",
    "comparative_research",
    "trend_analysis",
    "controversy_multihop",
)


class ResearchBench40Manifest(BaseModel):
    """Authored candidate manifest; existing development samples remain referenced."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    benchmark_id: Literal["researchbench_40_engineering"] = "researchbench_40_engineering"
    version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    source_benchmark: str = Field(min_length=1)
    existing_sample_ids: list[str] = Field(min_length=1)
    new_samples: list[BenchmarkSample] = Field(min_length=1)
    expected_count: Literal[40] = 40
    knowledge_cutoff_at: date

    @model_validator(mode="after")
    def validate_ids(self) -> "ResearchBench40Manifest":
        ids = [*self.existing_sample_ids, *(item.id for item in self.new_samples)]
        if len(ids) != len(set(ids)):
            raise ValueError("ResearchBench-40 candidate IDs must be unique.")
        if len(ids) != self.expected_count:
            raise ValueError(
                f"ResearchBench-40 requires exactly {self.expected_count} candidates, got {len(ids)}."
            )
        return self


class ResearchBench40SampleState(BaseModel):
    """Durable per-sample status used for safe build resumption."""

    model_config = ConfigDict(extra="forbid")

    sample_id: str
    status: Literal["passed", "failed"]
    candidate_manifest_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    error: str | None = None
    sample_path: str | None = None
    snapshot_manifest: str | None = None
    review_path: str | None = None
    attempt_path: str | None = None
    source_count: int = 0
    independent_domain_count: int = 0
    primary_source_count: int = 0
    review_profiles: list[str] = Field(default_factory=list)
    llm_calls: int = 0
    total_tokens: int = 0
    elapsed_seconds: float = 0.0
    resumed: bool = False
    updated_at: datetime


class ResearchBench40BuildResult(BaseModel):
    """One batch result plus the aggregate state of all forty fixed candidates."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    evaluation_mode: Literal["fully_automatic"] = "fully_automatic"
    release_tier: Literal["engineering_automatic"] = "engineering_automatic"
    human_calibrated: Literal[False] = False
    formal_benchmark_eligible: Literal[False] = False
    created_at: datetime
    candidate_manifest_hash: str
    requested_sample_ids: list[str]
    passed_count: int
    failed_count: int
    pending_count: int
    dataset_draft_path: str | None = None
    dataset_path: str | None = None
    final_validation_path: str | None = None
    samples: list[ResearchBench40SampleState]


class ResearchBench40Builder(AutoDraftBuilder):
    """Build the fixed, balanced forty-sample engineering dataset incrementally."""

    def __init__(
        self,
        config: AppConfig,
        *,
        search_provider: SearchProvider | None = None,
        review_clients: list[LLMClient] | None = None,
    ) -> None:
        # ResearchBench-40 freezes web evidence by contract.  Keep the application's
        # ordinary default (`local`) unchanged, while making this dedicated builder
        # deterministic and independent from whichever provider a user last used.
        builder_config = config
        if search_provider is None and config.retrieval.provider != "hybrid":
            builder_config = config.model_copy(
                update={
                    "retrieval": config.retrieval.model_copy(update={"provider": "hybrid"})
                }
            )
        super().__init__(
            builder_config,
            search_provider=search_provider,
            review_clients=review_clients,
        )

    def build(
        self,
        *,
        manifest_path: str | Path,
        output_dir: str | Path | None = None,
        sample_ids: list[str] | None = None,
        resume: bool = True,
        max_calls_per_sample: int = 12,
        max_tokens_per_sample: int = 40_000,
        max_seconds_per_sample: float = 300.0,
    ) -> ResearchBench40BuildResult:
        """Build a selected batch and materialize the full dataset only at 40/40."""

        manifest_file = Path(manifest_path).resolve()
        manifest = ResearchBench40Manifest.model_validate_json(
            manifest_file.read_text(encoding="utf-8")
        )
        manifest_hash = _sha256_file(manifest_file)
        candidates = self._resolve_candidates(manifest_file, manifest)
        self._validate_distribution(candidates)
        by_id = {item.id: item for item in candidates}
        requested = list(dict.fromkeys(sample_ids or [item.id for item in candidates]))
        unknown = sorted(set(requested) - set(by_id))
        if unknown:
            raise ValueError(f"Unknown ResearchBench-40 sample IDs: {unknown}")

        root = Path(output_dir).resolve() if output_dir else manifest_file.parent
        self._prepare_directories(root)
        batch_states: list[ResearchBench40SampleState] = []
        for sample_id in requested:
            if resume:
                resumed = self._load_passed_state(root, sample_id, manifest_hash)
                if resumed is not None:
                    batch_states.append(resumed.model_copy(update={"resumed": True}))
                    continue
            batch_states.append(
                self._build_and_persist_sample(
                    by_id[sample_id],
                    root=root,
                    manifest=manifest,
                    manifest_hash=manifest_hash,
                    max_calls=max_calls_per_sample,
                    max_tokens=max_tokens_per_sample,
                    max_seconds=max_seconds_per_sample,
                    replace_existing=not resume,
                )
            )

        all_states = [
            state
            for sample in candidates
            if (state := self._load_state(root, sample.id)) is not None
        ]
        accepted = self._load_all_passed_samples(root, candidates, manifest_hash)
        draft_path = self._write_dataset(
            root / "dataset_draft.json",
            manifest=manifest,
            samples=accepted,
            completed=False,
        ) if accepted else None
        dataset_path = None
        final_validation_path = None
        if len(accepted) == manifest.expected_count:
            dataset_path = self._write_dataset(
                root / "dataset.json",
                manifest=manifest,
                samples=accepted,
                completed=True,
            )
            final_validation_path = self._write_final_validation(
                root / "validation" / "final_validation.json",
                manifest=manifest,
                candidates=candidates,
                samples=accepted,
                states=all_states,
                manifest_hash=manifest_hash,
            )

        failed_count = sum(item.status == "failed" for item in all_states)
        passed_count = len(accepted)
        result = ResearchBench40BuildResult(
            created_at=datetime.now(timezone.utc),
            candidate_manifest_hash=manifest_hash,
            requested_sample_ids=requested,
            passed_count=passed_count,
            failed_count=failed_count,
            pending_count=manifest.expected_count - passed_count - failed_count,
            dataset_draft_path=str(draft_path) if draft_path else None,
            dataset_path=str(dataset_path) if dataset_path else None,
            final_validation_path=(
                str(final_validation_path) if final_validation_path else None
            ),
            samples=batch_states,
        )
        result_path = root / "validation" / "build_result.json"
        result_path.write_text(
            json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return result

    def _build_and_persist_sample(
        self,
        sample: BenchmarkSample,
        *,
        root: Path,
        manifest: ResearchBench40Manifest,
        manifest_hash: str,
        max_calls: int,
        max_tokens: int,
        max_seconds: float,
        replace_existing: bool,
    ) -> ResearchBench40SampleState:
        created_at = datetime.now(timezone.utc)
        existing_state = self._load_state(root, sample.id)
        attempt = (
            root
            / "validation"
            / "attempts"
            / sample.id
            / created_at.strftime("%Y%m%dT%H%M%S_%fZ")
        )
        attempt.mkdir(parents=True, exist_ok=False)
        review_path = attempt / "automatic_review.json"
        try:
            authored, result = self._build_sample(
                sample,
                build_dir=attempt,
                created_at=created_at,
                max_calls=max_calls,
                max_tokens=max_tokens,
                max_seconds=max_seconds,
                benchmark_version=manifest.version,
                snapshot_prefix="rb40",
                annotator="researchbench40_automatic_builder",
                reviewer="dual_flash_gold_support",
                min_supported_claims=3,
                review_output_path=review_path,
            )
            if not authored.forbidden_claims:
                raise ValueError("ResearchBench-40 samples require at least one forbidden claim.")
            final_snapshot = root / "snapshots" / sample.id
            final_review = root / "annotations" / "automatic_reviews" / f"{sample.id}.json"
            final_sample = root / "validation" / "samples" / f"{sample.id}.sample.json"
            if final_snapshot.exists() or final_review.exists() or final_sample.exists():
                if not replace_existing:
                    raise ValueError(
                        f"Immutable accepted artifacts already exist for '{sample.id}' "
                        "but could not resume."
                    )
                self._archive_existing_artifacts(
                    root,
                    sample.id,
                    created_at=created_at,
                    sample_path=final_sample,
                    snapshot_path=final_snapshot,
                    review_path=final_review,
                )
            shutil.move(str(attempt / "snapshots" / sample.id), str(final_snapshot))
            shutil.move(str(review_path), str(final_review))
            final_sample.write_text(
                json.dumps(authored.model_dump(mode="json"), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            state = self._passed_state(
                sample.id,
                root=root,
                manifest_hash=manifest_hash,
                result=result,
                sample_path=final_sample,
                review_path=final_review,
                attempt=attempt,
                updated_at=created_at,
            )
        except Exception as exc:  # 固定问题失败必须持久化，不能静默换题。
            state = ResearchBench40SampleState(
                sample_id=sample.id,
                status="failed",
                candidate_manifest_hash=manifest_hash,
                error=f"{type(exc).__name__}: {exc}",
                attempt_path=str(attempt.relative_to(root)),
                review_profiles=self.review_profiles,
                updated_at=created_at,
            )
        # A failed explicit rebuild must not invalidate a previously accepted
        # immutable sample.  The failed attempt remains available in its own dir.
        if not (
            state.status == "failed"
            and replace_existing
            and existing_state is not None
            and existing_state.status == "passed"
        ):
            self._write_state(root, state)
        else:
            (attempt / "failure.json").write_text(
                json.dumps(state.model_dump(mode="json"), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        return state

    @classmethod
    def _archive_existing_artifacts(
        cls,
        root: Path,
        sample_id: str,
        *,
        created_at: datetime,
        sample_path: Path,
        snapshot_path: Path,
        review_path: Path,
    ) -> None:
        """Archive an accepted version only after its replacement has passed."""

        archive = (
            root
            / "validation"
            / "superseded"
            / sample_id
            / created_at.strftime("%Y%m%dT%H%M%S_%fZ")
        )
        archive.mkdir(parents=True, exist_ok=False)
        if sample_path.exists():
            shutil.move(str(sample_path), str(archive / "sample.json"))
        if snapshot_path.exists():
            shutil.move(str(snapshot_path), str(archive / "snapshot"))
        if review_path.exists():
            shutil.move(str(review_path), str(archive / "automatic_review.json"))
        state_path = cls._state_path(root, sample_id)
        if state_path.exists():
            shutil.move(str(state_path), str(archive / "result.json"))

    @staticmethod
    def _passed_state(
        sample_id: str,
        *,
        root: Path,
        manifest_hash: str,
        result: AutoDraftSampleResult,
        sample_path: Path,
        review_path: Path,
        attempt: Path,
        updated_at: datetime,
    ) -> ResearchBench40SampleState:
        return ResearchBench40SampleState(
            sample_id=sample_id,
            status="passed",
            candidate_manifest_hash=manifest_hash,
            sample_path=str(sample_path.relative_to(root)),
            snapshot_manifest=f"snapshots/{sample_id}/manifest.json",
            review_path=str(review_path.relative_to(root)),
            attempt_path=str(attempt.relative_to(root)),
            source_count=result.source_count,
            independent_domain_count=result.independent_domain_count,
            primary_source_count=result.primary_source_count,
            review_profiles=result.review_profiles,
            llm_calls=result.llm_calls,
            total_tokens=result.total_tokens,
            elapsed_seconds=result.elapsed_seconds,
            updated_at=updated_at,
        )

    @staticmethod
    def _prepare_directories(root: Path) -> None:
        for relative in (
            "snapshots",
            "annotations/automatic_reviews",
            "validation/attempts",
            "validation/samples",
        ):
            (root / relative).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _state_path(root: Path, sample_id: str) -> Path:
        return root / "validation" / "samples" / f"{sample_id}.result.json"

    @classmethod
    def _write_state(cls, root: Path, state: ResearchBench40SampleState) -> None:
        cls._state_path(root, state.sample_id).write_text(
            json.dumps(state.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def _load_state(cls, root: Path, sample_id: str) -> ResearchBench40SampleState | None:
        path = cls._state_path(root, sample_id)
        if not path.exists():
            return None
        return ResearchBench40SampleState.model_validate_json(path.read_text(encoding="utf-8"))

    @classmethod
    def _load_passed_state(
        cls,
        root: Path,
        sample_id: str,
        manifest_hash: str,
    ) -> ResearchBench40SampleState | None:
        state = cls._load_state(root, sample_id)
        if state is None or state.status != "passed":
            return None
        if state.candidate_manifest_hash != manifest_hash:
            raise ValueError(
                f"Candidate manifest changed after '{sample_id}' was accepted; create a new version."
            )
        cls._validate_persisted_sample(root, state)
        return state

    @classmethod
    def _validate_persisted_sample(
        cls,
        root: Path,
        state: ResearchBench40SampleState,
    ) -> BenchmarkSampleV2:
        if not state.sample_path or not state.snapshot_manifest or not state.review_path:
            raise ValueError(f"Accepted state for '{state.sample_id}' is missing artifact paths.")
        sample = BenchmarkSampleV2.model_validate_json(
            (root / state.sample_path).read_text(encoding="utf-8")
        )
        frozen = load_frozen_snapshot(root / state.snapshot_manifest)
        if sample.snapshot is None or frozen.manifest.content_hash != sample.snapshot.content_hash:
            raise ValueError(f"Snapshot hash mismatch for accepted sample '{state.sample_id}'.")
        if not (root / state.review_path).is_file():
            raise ValueError(f"Automatic review is missing for '{state.sample_id}'.")
        domains = {
            (urlsplit(document.url).hostname or "").lower()
            for document in frozen.documents
            if urlsplit(document.url).hostname
        }
        if len(domains) < 2:
            raise ValueError(f"Accepted sample '{state.sample_id}' has fewer than two domains.")
        if len(sample.claims) < 3:
            raise ValueError(f"Accepted sample '{state.sample_id}' has fewer than three Gold claims.")
        if not sample.forbidden_claims:
            raise ValueError(f"Accepted sample '{state.sample_id}' has no forbidden claim.")
        expected_dimensions = {
            "information_recall",
            "analysis_reasoning",
            "citation_quality",
            "task_following",
            "presentation",
        }
        if {criterion.dimension for criterion in sample.rubric} != expected_dimensions:
            raise ValueError(f"Accepted sample '{state.sample_id}' has an incomplete rubric.")
        source_ids = {document.id for document in frozen.documents}
        claim_ids = {claim.id for claim in sample.claims}
        for requirement in sample.evidence_requirements:
            if not set(requirement.claim_ids).issubset(claim_ids):
                raise ValueError(
                    f"Evidence requirement '{requirement.id}' references an unknown claim."
                )
            if not set(requirement.required_source_ids).issubset(source_ids):
                raise ValueError(
                    f"Evidence requirement '{requirement.id}' references an unknown source."
                )
        cls._validate_review_artifact(
            root / state.review_path,
            sample=sample,
            documents={document.id: document for document in frozen.documents},
        )
        return sample

    @staticmethod
    def _validate_review_artifact(
        path: Path,
        *,
        sample: BenchmarkSampleV2,
        documents: dict[str, SourceDoc],
    ) -> None:
        review = json.loads(path.read_text(encoding="utf-8"))
        profiles = review.get("review_profiles")
        rounds = review.get("rounds")
        if not isinstance(profiles, list) or len(profiles) != 2 or len(set(profiles)) != 2:
            raise ValueError(f"Automatic review for '{sample.id}' needs two distinct profiles.")
        if not isinstance(rounds, list) or not rounds:
            raise ValueError(f"Automatic review for '{sample.id}' has no review round.")
        final_round = rounds[-1]
        accepted = final_round.get("accepted_source_ids", {})
        expected_claims = {claim.id for claim in sample.claims}
        if not expected_claims.issubset(set(accepted)):
            raise ValueError(f"Automatic review for '{sample.id}' does not accept every Gold claim.")
        for reviewer in final_round.get("reviews", []):
            by_claim: dict[str, dict[str, object]] = {}
            for item in reviewer.get("claims", []):
                raw_id = str(item.get("claim_id", ""))
                matches = [
                    claim_id
                    for claim_id in expected_claims
                    if claim_id == raw_id or claim_id.endswith(f".{raw_id}")
                ]
                if len(matches) == 1:
                    by_claim[matches[0]] = item
            if set(by_claim) != expected_claims:
                raise ValueError(
                    f"A final automatic review for '{sample.id}' omits a Gold claim."
                )
            for claim_id, item in by_claim.items():
                source_id = item.get("source_id")
                quote = item.get("quote")
                document = documents.get(source_id)
                if (
                    item.get("verdict") != "supported"
                    or document is None
                    or not isinstance(quote, str)
                    or _ground_quote(document.content, quote) is None
                ):
                    raise ValueError(
                        f"Final review quote for '{claim_id}' is not exactly grounded."
                    )

    @classmethod
    def _load_all_passed_samples(
        cls,
        root: Path,
        candidates: list[BenchmarkSample],
        manifest_hash: str,
    ) -> list[BenchmarkSampleV2]:
        accepted: list[BenchmarkSampleV2] = []
        for candidate in candidates:
            state = cls._load_state(root, candidate.id)
            if state is None or state.status != "passed":
                continue
            if state.candidate_manifest_hash != manifest_hash:
                raise ValueError(
                    f"Candidate manifest changed after '{candidate.id}' was accepted."
                )
            sample = cls._validate_persisted_sample(root, state)
            if _sample_requires_primary(candidate) and state.primary_source_count < 1:
                raise ValueError(
                    f"Accepted sample '{candidate.id}' requires a primary source but has none."
                )
            accepted.append(sample)
        return accepted

    @staticmethod
    def _resolve_candidates(
        manifest_path: Path,
        manifest: ResearchBench40Manifest,
    ) -> list[BenchmarkSample]:
        source_path = (manifest_path.parent / manifest.source_benchmark).resolve()
        source = {item.id: item for item in load_benchmark(source_path)}
        missing = sorted(set(manifest.existing_sample_ids) - set(source))
        if missing:
            raise ValueError(f"Referenced development samples are missing: {missing}")
        candidates = [source[item] for item in manifest.existing_sample_ids]
        candidates.extend(manifest.new_samples)
        return candidates

    @staticmethod
    def _validate_distribution(samples: list[BenchmarkSample]) -> None:
        if len(samples) != 40:
            raise ValueError(f"ResearchBench-40 requires 40 samples, got {len(samples)}.")
        languages = Counter(item.language for item in samples)
        tasks = Counter(item.task_type for item in samples)
        task_languages = Counter((item.task_type, item.language) for item in samples)
        if languages != {"zh": 20, "en": 20}:
            raise ValueError(f"Expected 20 zh and 20 en samples, got {dict(languages)}.")
        expected_tasks = {item: 10 for item in RESEARCHBENCH40_TASK_TYPES}
        if tasks != expected_tasks:
            raise ValueError(f"Expected 10 samples per task type, got {dict(tasks)}.")
        expected_pairs = {
            (task, language): 5
            for task in RESEARCHBENCH40_TASK_TYPES
            for language in ("zh", "en")
        }
        if task_languages != expected_pairs:
            raise ValueError(
                "Expected five samples for every task/language pair, got "
                f"{dict(task_languages)}."
            )
        for sample in samples:
            if len(sample.expected_claims) < 3:
                raise ValueError(f"Sample '{sample.id}' needs at least three answer requirements.")
            if not sample.forbidden_hallucinations:
                raise ValueError(f"Sample '{sample.id}' needs at least one forbidden claim.")

    @staticmethod
    def _write_dataset(
        path: Path,
        *,
        manifest: ResearchBench40Manifest,
        samples: list[BenchmarkSampleV2],
        completed: bool,
    ) -> Path:
        dataset = BenchmarkDatasetV2(
            metadata=BenchmarkMetadata(
                benchmark_id=manifest.benchmark_id,
                version=manifest.version,
                track="frozen",
                released_at=date.today(),
                knowledge_cutoff_at=manifest.knowledge_cutoff_at,
                language_mix=sorted({item.language for item in samples}),
                license="Engineering evaluation only; source licenses remain upstream.",
                dataset_card="DATASET_CARD.md",
                change_notes=[
                    "Balanced forty-sample engineering automatic frozen draft.",
                    "Gold claims require agreement from two configured Flash review profiles.",
                    "All forty fixed candidates passed." if completed else "Partial resumable draft.",
                ],
                release_tier="engineering_automatic",
                human_calibrated=False,
                formal_benchmark_eligible=False,
            ),
            samples=samples,
        )
        path.write_text(
            json.dumps(dataset.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path

    @staticmethod
    def _write_final_validation(
        path: Path,
        *,
        manifest: ResearchBench40Manifest,
        candidates: list[BenchmarkSample],
        samples: list[BenchmarkSampleV2],
        states: list[ResearchBench40SampleState],
        manifest_hash: str,
    ) -> Path:
        """Persist construction gates without presenting them as system evaluation."""

        state_by_id = {state.sample_id: state for state in states}
        task_counts = Counter(sample.task_type for sample in samples)
        language_counts = Counter(sample.language for sample in samples)
        primary_required_ids = [
            candidate.id for candidate in candidates if _sample_requires_primary(candidate)
        ]
        payload = {
            "schema_version": "1.0",
            "validation_scope": "benchmark_construction_only",
            "evaluation_executed": False,
            "release_tier": "engineering_automatic",
            "human_calibrated": False,
            "formal_benchmark_eligible": False,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "candidate_manifest_hash": manifest_hash,
            "dataset_version": manifest.version,
            "sample_count": len(samples),
            "status_counts": {"passed": len(samples), "failed": 0, "pending": 0},
            "language_counts": dict(sorted(language_counts.items())),
            "task_type_counts": dict(sorted(task_counts.items())),
            "gold_claim_count": sum(len(sample.claims) for sample in samples),
            "snapshot_count": len(samples),
            "automatic_review_count": len(samples),
            "primary_source_gate": {
                "applicable_sample_count": len(primary_required_ids),
                "satisfied_sample_count": sum(
                    state_by_id[sample_id].primary_source_count >= 1
                    for sample_id in primary_required_ids
                ),
            },
            "source_counts": {
                "minimum_per_sample": min(state.source_count for state in states),
                "maximum_per_sample": max(state.source_count for state in states),
                "minimum_independent_domains": min(
                    state.independent_domain_count for state in states
                ),
                "maximum_independent_domains": max(
                    state.independent_domain_count for state in states
                ),
            },
            "accepted_build_resources": {
                "llm_calls": sum(state.llm_calls for state in states),
                "total_tokens": sum(state.total_tokens for state in states),
                "elapsed_seconds": round(sum(state.elapsed_seconds for state in states), 3),
                "scope_note": "Excludes failed and superseded construction attempts.",
            },
            "checks": {
                "schema": "passed",
                "balanced_distribution": "passed",
                "snapshot_hashes": "passed",
                "source_and_claim_references": "passed",
                "exact_final_review_quotes": "passed",
                "five_dimension_rubric": "passed",
                "dual_profile_reviews": "passed",
            },
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    @staticmethod
    def _rubric(sample: BenchmarkSample) -> list[RubricCriterion]:
        """Guarantee one explicit criterion for every required evaluation dimension."""

        dimensions = (
            "information_recall",
            "analysis_reasoning",
            "citation_quality",
            "task_following",
            "presentation",
        )
        defaults = {
            "information_recall": "Covers the required answer claims and named subjects.",
            "analysis_reasoning": "Connects evidence to conclusions without unsupported leaps.",
            "citation_quality": "Uses directly supporting and traceable citations.",
            "task_following": "Follows the requested scope, comparison dimensions, and time boundary.",
            "presentation": "Presents a clear, structured, and appropriately qualified answer.",
        }
        descriptions = list(sample.judge_rubric[:5])
        descriptions.extend(
            defaults[dimension] for dimension in dimensions[len(descriptions) :]
        )
        return [
            RubricCriterion(
                id=f"{sample.id}.r{index}",
                dimension=dimension,
                description=description,
                weight=1.0,
                evaluation_mode="hybrid",
                anchors=[
                    RubricAnchor(score=0, description=f"Does not satisfy: {description}"),
                    RubricAnchor(score=4, description=f"Fully and accurately satisfies: {description}"),
                ],
            )
            for index, (dimension, description) in enumerate(
                zip(dimensions, descriptions, strict=True),
                start=1,
            )
        ]


def export_researchbench40_schema(destination: str | Path) -> Path:
    """Export the existing canonical Benchmark v2 schema into the dataset folder."""

    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(benchmark_v2_json_schema(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def _sha256_file(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"

"""Track A engineering-only Benchmark draft construction.

The builder converts fixed legacy development questions into strict Benchmark v2
drafts, freezes retrieved source bytes, and asks two configured Flash profiles to
independently ground every authored gold claim.  It intentionally fails closed:
failed fixed samples remain in the build record and are never replaced.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, model_validator

from deepresearch.config import AppConfig
from deepresearch.agents.reader import _ground_quote
from deepresearch.evaluation.benchmark import (
    AtomicClaim,
    AtomicClaimSatisfactionRoute,
    BenchmarkDatasetV2,
    BenchmarkMetadata,
    BenchmarkSample,
    BenchmarkSampleV2,
    EvidenceRequirement,
    ForbiddenClaim,
    RubricAnchor,
    RubricCriterion,
    SnapshotReference,
    load_benchmark,
)
from deepresearch.llm import ChatMessage, LLMClient, LLMTraceStore, create_llm_client
from deepresearch.quality.sources import assess_source_credibility
from deepresearch.retrieval import create_search_provider
from deepresearch.retrieval.base import SearchProvider
from deepresearch.retrieval.snapshot import FrozenQueryRecord, write_frozen_snapshot
from deepresearch.schema import SourceDoc


TRACK_A_DRAFT_SAMPLE_IDS = (
    "full_005",
    "full_006",
    "full_007",
    "full_015",
    "full_019",
    "full_024",
    "full_025",
    "full_026",
)

_PRIMARY_HINTS = (
    "原始",
    "官方",
    "标准",
    "监管",
    "论文",
    "研究",
    "报告",
    "model card",
    "technical report",
    "official",
    "original",
    "specification",
    "authoritative",
    "standards-body",
    "documentation",
    "report",
    "peer-reviewed",
    "field stud",
)
_OFFICIAL_HOSTS = {
    "acm.org",
    "dl.acm.org",
    "arxiv.org",
    "doi.org",
    "frontiersin.org",
    "ieee.org",
    "ietf.org",
    "rfc-editor.org",
    "openid.net",
    "ipcc.ch",
    "iea.org",
    "nrc.gov",
    "postgresql.org",
    "iso.org",
    "nist.gov",
    "oecd.org",
    "who.int",
    "nature.com",
    "sciencedirect.com",
    "springer.com",
    "plos.org",
    "openreview.net",
    "proceedings.mlr.press",
    "huggingface.co",
    "amazon.com",
    "aws.amazon.com",
    "ibm.com",
    "oracle.com",
    "microsoft.com",
    "postgresql.org",
    "berkeley.edu",
    "w3.org",
    "fidoalliance.org",
}


class DraftAuthoredClaim(BaseModel):
    """One evidence-first candidate gold claim produced from frozen text."""

    model_config = ConfigDict(extra="forbid")

    claim_id: str
    text: str = Field(min_length=1)
    source_id: str
    quote: str = Field(min_length=1)


class DraftClaimAuthoring(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claims: list[DraftAuthoredClaim]

    @model_validator(mode="before")
    @classmethod
    def normalize_common_json_envelope(cls, value):
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        if "claims" not in normalized and isinstance(normalized.get("results"), list):
            normalized["claims"] = normalized.pop("results")
        normalized.pop("type", None)
        return normalized


class DraftClaimSupport(BaseModel):
    """One profile's exact-text grounding decision for one authored claim."""

    model_config = ConfigDict(extra="forbid")

    claim_id: str
    verdict: Literal["supported", "unsupported"]
    source_id: str | None = None
    quote: str | None = None
    reason: str = Field(default="No reason supplied.", min_length=1, max_length=500)

    @model_validator(mode="before")
    @classmethod
    def normalize_supported_boolean(cls, value):
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        if "verdict" not in normalized and normalized.get("status") in {
            "supported",
            "unsupported",
        }:
            normalized["verdict"] = normalized.pop("status")
        if "verdict" not in normalized and isinstance(normalized.get("supported"), bool):
            normalized["verdict"] = (
                "supported" if normalized.pop("supported") else "unsupported"
            )
        # Some JSON-capable models use `note` for the requested short reason.
        # Normalize that one semantic alias while retaining extra="forbid" for
        # genuinely unknown fields and accidental schema drift.
        if "reason" not in normalized and isinstance(normalized.get("note"), str):
            normalized["reason"] = normalized.pop("note")
        else:
            normalized.pop("note", None)
        normalized.pop("text", None)
        return normalized


class DraftGoldSupportReview(BaseModel):
    """Compact response required from each independent support-review profile."""

    model_config = ConfigDict(extra="forbid")

    claims: list[DraftClaimSupport]

    @model_validator(mode="before")
    @classmethod
    def normalize_common_json_envelope(cls, value):
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        if "claims" not in normalized and isinstance(normalized.get("results"), list):
            normalized["claims"] = normalized.pop("results")
        if "claims" not in normalized:
            keyed_claims = []
            for key in sorted(normalized):
                item = normalized.get(key)
                if re.fullmatch(r"c\d+", str(key), flags=re.IGNORECASE) and isinstance(
                    item, dict
                ):
                    keyed_claims.append({"claim_id": str(key), **item})
            if keyed_claims:
                normalized = {"claims": keyed_claims}
        normalized.pop("type", None)
        return normalized


class AutoDraftSampleResult(BaseModel):
    sample_id: str
    status: Literal["passed", "failed"]
    error: str | None = None
    snapshot_manifest: str | None = None
    source_count: int = 0
    independent_domain_count: int = 0
    primary_source_count: int = 0
    review_profiles: list[str] = Field(default_factory=list)
    llm_calls: int = 0
    total_tokens: int = 0
    elapsed_seconds: float = 0.0


class AutoDraftBuildResult(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    evaluation_mode: Literal["fully_automatic"] = "fully_automatic"
    release_tier: Literal["engineering_automatic"] = "engineering_automatic"
    human_calibrated: Literal[False] = False
    formal_benchmark_eligible: Literal[False] = False
    created_at: datetime
    requested_sample_ids: list[str]
    dataset_path: str | None = None
    passed_count: int
    failed_count: int
    samples: list[AutoDraftSampleResult]


class AutoDraftBuilder:
    """Build immutable, engineering-only v2 drafts from a fixed legacy subset."""

    def __init__(
        self,
        config: AppConfig,
        *,
        search_provider: SearchProvider | None = None,
        review_clients: list[LLMClient] | None = None,
    ) -> None:
        self.config = config
        retrieval_config = config.retrieval.model_copy(
            update={
                "provider": "hybrid",
                "top_k": 6,
                "search_depth": "advanced",
                # Tavily 已返回清洗正文。自动草稿冻结该正文即可；逐 URL 补抓会让一个
                # 不可达页面反复消耗网络超时，破坏每样本 300 秒工程预算。
                "fetch_full_content": False,
            }
        )
        self.search_provider = search_provider or create_search_provider(retrieval_config)
        profiles = list(config.evaluation.judge_profiles[:2])
        if review_clients is None:
            if len(profiles) != 2:
                raise ValueError("Auto-draft construction requires exactly two Judge profiles.")
            review_clients = [
                create_llm_client(config.llm, profile_name=profile) for profile in profiles
            ]
        if len(review_clients) != 2:
            raise ValueError("Auto-draft construction requires exactly two review clients.")
        self.review_clients = review_clients
        self.review_profiles = [client.profile_name or client.model for client in review_clients]
        for client in self.review_clients:
            client.configure_raw_trace(
                enabled=config.runtime.save_llm_io,
                store=LLMTraceStore(
                    config.runtime.run_dir,
                    subdir=config.runtime.llm_trace_subdir,
                ),
            )

    def build(
        self,
        *,
        source_benchmark: str | Path = "data/researchbench_full.json",
        output_root: str | Path = "data/benchmarks/researchbench_v2/track_a_auto_draft",
        sample_ids: list[str] | None = None,
        max_calls_per_sample: int = 12,
        max_tokens_per_sample: int = 40_000,
        max_seconds_per_sample: float = 300.0,
    ) -> AutoDraftBuildResult:
        selected_ids = list(dict.fromkeys(sample_ids or TRACK_A_DRAFT_SAMPLE_IDS))
        unknown_fixed = sorted(set(selected_ids) - set(TRACK_A_DRAFT_SAMPLE_IDS))
        if unknown_fixed:
            raise ValueError(f"Track A auto-draft only accepts fixed sample ids: {unknown_fixed}")
        source_samples = {item.id: item for item in load_benchmark(source_benchmark)}
        missing = sorted(set(selected_ids) - set(source_samples))
        if missing:
            raise ValueError(f"Source benchmark is missing fixed samples: {missing}")

        created_at = datetime.now(timezone.utc)
        build_dir = Path(output_root) / created_at.strftime("%Y%m%dT%H%M%S_%fZ")
        build_dir.mkdir(parents=True, exist_ok=False)
        authored: list[BenchmarkSampleV2] = []
        results: list[AutoDraftSampleResult] = []
        for sample_id in selected_ids:
            sample = source_samples[sample_id]
            try:
                item, result = self._build_sample(
                    sample,
                    build_dir=build_dir,
                    created_at=created_at,
                    max_calls=max_calls_per_sample,
                    max_tokens=max_tokens_per_sample,
                    max_seconds=max_seconds_per_sample,
                )
                authored.append(item)
                results.append(result)
            except Exception as exc:  # 固定题失败必须留档，不能换题掩盖。
                results.append(
                    AutoDraftSampleResult(
                        sample_id=sample.id,
                        status="failed",
                        error=f"{type(exc).__name__}: {exc}",
                        review_profiles=self.review_profiles,
                    )
                )

        dataset_path: Path | None = None
        if len(authored) == len(selected_ids):
            dataset = BenchmarkDatasetV2(
                metadata=BenchmarkMetadata(
                    benchmark_id="researchbench_track_a_auto_draft",
                    version="0.1.0",
                    track="frozen",
                    released_at=created_at.date(),
                    knowledge_cutoff_at=created_at.date(),
                    language_mix=sorted({item.language for item in authored}),
                    license="Engineering evaluation only; source licenses remain upstream.",
                    dataset_card="docs/BENCHMARK_DESIGN.md",
                    change_notes=[
                        "Fixed eight-sample Track A engineering automatic draft.",
                        "Gold support reviewed once by each of two Flash profiles.",
                    ],
                    release_tier="engineering_automatic",
                    human_calibrated=False,
                    formal_benchmark_eligible=False,
                ),
                samples=authored,
            )
            dataset_path = build_dir / "dataset.json"
            dataset_path.write_text(
                json.dumps(dataset.model_dump(mode="json"), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        summary = AutoDraftBuildResult(
            created_at=created_at,
            requested_sample_ids=selected_ids,
            dataset_path=str(dataset_path) if dataset_path else None,
            passed_count=len(authored),
            failed_count=len(selected_ids) - len(authored),
            samples=results,
        )
        (build_dir / "build_result.json").write_text(
            json.dumps(summary.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return summary

    def _build_sample(
        self,
        sample: BenchmarkSample,
        *,
        build_dir: Path,
        created_at: datetime,
        max_calls: int,
        max_tokens: int,
        max_seconds: float,
        benchmark_version: str = "0.1.0",
        snapshot_prefix: str = "tracka",
        annotator: str = "track_a_automatic_builder",
        reviewer: str = "dual_flash_gold_support",
        min_supported_claims: int = 2,
        review_output_path: Path | None = None,
    ) -> tuple[BenchmarkSampleV2, AutoDraftSampleResult]:
        """Build one v2 sample while keeping Track A defaults backward compatible.

        The optional metadata and review path are used by larger resumable builds.
        Raw prompts and provider responses remain in the private trace store; the
        review artifact contains only normalized claims, verdicts, and source IDs.
        """

        started = time.perf_counter()
        queries = self._queries(sample)
        primary_required = _sample_requires_primary(sample)
        documents, query_records = self._retrieve(
            queries,
            reserve_primary=primary_required,
        )
        domains = {_domain(item.url) for item in documents if _domain(item.url)}
        if len(domains) < 2:
            raise ValueError("Fewer than two independent source domains were retrieved.")
        primary_count = sum(_is_primary_document(item) for item in documents)
        if primary_required and primary_count < 1:
            raise ValueError("No official or first-hand source was retrieved for a primary-source task.")

        snapshot_dir = build_dir / "snapshots" / sample.id
        manifest = write_frozen_snapshot(
            snapshot_dir,
            snapshot_id=f"{snapshot_prefix}_{sample.id}_{created_at.strftime('%Y%m%d')}",
            sample_id=sample.id,
            benchmark_version=benchmark_version,
            captured_at=created_at,
            provider="hybrid_tavily",
            queries=query_records,
            documents=documents,
        )
        run_id = f"autodraft_{sample.id}_{created_at.strftime('%H%M%S')}"
        before = [client.trace_collector.mark() for client in self.review_clients]
        for client in self.review_clients:
            client.bind_run(run_id, created_at=created_at)
        review_rounds: list[dict[str, object]] = []
        try:
            conservative_retry_used = False
            try:
                claims = asyncio.run(self._author_claims(sample, documents))
            except ValueError:
                author_traces = [
                    trace
                    for client, offset in zip(self.review_clients, before, strict=True)
                    for trace in client.trace_collector.since(offset)
                ]
                author_tokens = _trace_tokens(author_traces)
                # One failed author call is followed by one author retry and two
                # reviews. Use a conservative four-call estimate before spending.
                if author_tokens * 4 > max_tokens:
                    raise ValueError(
                        "Conservative author rewrite would exceed the per-sample token "
                        f"budget: estimated>{max_tokens}, initial={author_tokens}."
                    )
                claims = asyncio.run(
                    self._author_claims(sample, documents, conservative_retry=True)
                )
                conservative_retry_used = True
            reviews = asyncio.run(self._review_claims(sample, claims, documents))
            support_ids, dropped_claim_ids = self._validate_reviews(
                claims,
                documents,
                reviews,
            )
            review_rounds.append(
                _normalized_review_round(
                    round_name=(
                        "conservative_retry" if conservative_retry_used else "initial"
                    ),
                    claims=claims,
                    reviews=reviews,
                    support_ids=support_ids,
                    dropped_claim_ids=dropped_claim_ids,
                )
            )
            if len(support_ids) < min_supported_claims and not conservative_retry_used:
                initial_traces = [
                    trace
                    for client, offset in zip(self.review_clients, before, strict=True)
                    for trace in client.trace_collector.since(offset)
                ]
                initial_tokens = _trace_tokens(initial_traces)
                # The retry repeats one author call and two reviews with similarly
                # sized inputs. Fail before spending a second round when even that
                # conservative estimate cannot fit the declared per-sample budget.
                if initial_tokens * 2 > max_tokens:
                    raise ValueError(
                        "Conservative gold rewrite would exceed the per-sample token "
                        f"budget: estimated>{max_tokens}, initial={initial_tokens}."
                    )
                # 一次受限语义重试：重新生成更窄的 claim，再由两个 profile 独立复核。
                claims = asyncio.run(
                    self._author_claims(sample, documents, conservative_retry=True)
                )
                reviews = asyncio.run(self._review_claims(sample, claims, documents))
                support_ids, dropped_claim_ids = self._validate_reviews(
                    claims,
                    documents,
                    reviews,
                )
                review_rounds.append(
                    _normalized_review_round(
                        round_name="conservative_retry",
                        claims=claims,
                        reviews=reviews,
                        support_ids=support_ids,
                        dropped_claim_ids=dropped_claim_ids,
                    )
                )
        finally:
            for client in self.review_clients:
                client.unbind_run()
        traces = [
            trace
            for client, offset in zip(self.review_clients, before, strict=True)
            for trace in client.trace_collector.since(offset)
        ]
        calls = len(traces)
        total_tokens = _trace_tokens(traces)
        elapsed = time.perf_counter() - started
        if calls > max_calls:
            raise ValueError(f"Per-sample LLM call budget exceeded: {calls}/{max_calls}.")
        if total_tokens > max_tokens:
            raise ValueError(f"Per-sample token budget exceeded: {total_tokens}/{max_tokens}.")
        if elapsed > max_seconds:
            raise ValueError(f"Per-sample runtime budget exceeded: {elapsed:.1f}/{max_seconds}s.")
        if review_output_path is not None:
            review_output_path.parent.mkdir(parents=True, exist_ok=True)
            review_output_path.write_text(
                json.dumps(
                    {
                        "schema_version": "1.0",
                        "sample_id": sample.id,
                        "review_profiles": self.review_profiles,
                        "rounds": review_rounds,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        if len(support_ids) < min_supported_claims:
            # Preserve the original Track A error text because historical tests and
            # experiment records use it as a stable failure classifier.
            minimum_label = "two" if min_supported_claims == 2 else str(min_supported_claims)
            raise ValueError(
                f"Fewer than {minimum_label} candidate gold claims passed both "
                "support reviews after retry."
            )
        claims = [item for item in claims if item.id in support_ids]
        evidence_requirements = self._evidence_requirements(
            sample,
            claims,
            support_ids,
            documents,
            primary_required=primary_required,
        )
        authored = BenchmarkSampleV2(
            id=sample.id,
            question=sample.question,
            language=sample.language,
            domain=sample.domain or "general",
            task_type=sample.task_type,
            difficulty=sample.difficulty,
            time_range=sample.time_range,
            claims=claims,
            evidence_requirements=evidence_requirements,
            forbidden_claims=[
                ForbiddenClaim(id=f"{sample.id}.f{index}", text=text)
                for index, text in enumerate(sample.forbidden_hallucinations, start=1)
            ],
            rubric=self._rubric(sample),
            snapshot=SnapshotReference(
                snapshot_id=manifest.snapshot_id,
                manifest_path=(Path("snapshots") / sample.id / "manifest.json").as_posix(),
                content_hash=manifest.content_hash,
            ),
            annotator=annotator,
            reviewer=reviewer,
            annotated_at=date.today(),
            disagreement_notes=(
                [
                    "Automatic support review dropped candidate gold claims: "
                    + ", ".join(dropped_claim_ids)
                ]
                if dropped_claim_ids
                else []
            ),
            status="draft",
        )
        return authored, AutoDraftSampleResult(
            sample_id=sample.id,
            status="passed",
            snapshot_manifest=str(snapshot_dir / "manifest.json"),
            source_count=len(documents),
            independent_domain_count=len(domains),
            primary_source_count=primary_count,
            review_profiles=self.review_profiles,
            llm_calls=calls,
            total_tokens=total_tokens,
            elapsed_seconds=elapsed,
        )

    def _retrieve(
        self,
        queries: list[str],
        *,
        reserve_primary: bool,
    ) -> tuple[list[SourceDoc], list[FrozenQueryRecord]]:
        by_id: dict[str, SourceDoc] = {}
        query_ids: list[list[str]] = []
        for query in queries:
            # 12 个候选给官方/论文来源留出位置；最终快照仍只保留 12 个去重来源。
            docs, _ = self.search_provider.search_with_diagnostics(query, top_k=12)
            selected_ids: list[str] = []
            for doc in docs:
                if not doc.content.strip():
                    continue
                assessed = assess_source_credibility(doc)
                by_id.setdefault(assessed.id, assessed)
                selected_ids.append(assessed.id)
            if selected_ids:
                query_ids.append(selected_ids)
            else:
                query_ids.append([])
        if not by_id:
            raise ValueError("Retrieval returned no non-empty source documents.")
        # 先保留首次出现的一手来源，再按原始查询/结果顺序补齐。否则通用博客可能
        # 占满前 12 位，使“原始论文/官方规范”查询虽命中却永远进不了冻结快照。
        ordered = list(by_id.values())
        documents: list[SourceDoc] = []
        if reserve_primary:
            primary = next((item for item in ordered if _is_primary_document(item)), None)
            if primary is not None:
                documents.append(primary)
        documents.extend(item for item in ordered if item.id not in {doc.id for doc in documents})
        documents = documents[:12]
        allowed = {item.id for item in documents}
        records = [
            FrozenQueryRecord(
                id=f"q{index}",
                query=query,
                result_source_ids=[source_id for source_id in ids if source_id in allowed],
            )
            for index, (query, ids) in enumerate(zip(queries, query_ids, strict=True), start=1)
            if any(source_id in allowed for source_id in ids)
        ]
        if not records:
            raise ValueError("No frozen query retained a source document.")
        return documents, records

    async def _review_claims(
        self,
        sample: BenchmarkSample,
        claims: list[AtomicClaim],
        documents: list[SourceDoc],
    ) -> list[DraftGoldSupportReview]:
        source_payload = _compact_source_payload(sample, documents)
        user_payload = {
            "question": sample.question,
            "claims": [{"claim_id": item.id, "text": item.text} for item in claims],
            "sources": source_payload,
        }
        messages = [
            ChatMessage(
                role="system",
                content=(
                    "You are an evidence-grounding reviewer. Return JSON only. Review every claim "
                    "exactly once. Mark supported only when one supplied source directly supports the "
                    "whole claim. For supported claims, source_id must exist and quote must be an exact "
                    "contiguous copy from that source content. Otherwise mark unsupported with null "
                    "source_id and quote. Keep reason under 40 words."
                ),
            ),
            ChatMessage(
                role="user",
                content=json.dumps(user_payload, ensure_ascii=False),
            ),
        ]
        return list(
            await asyncio.gather(
                *(
                    client.json_chat(
                        messages,
                        DraftGoldSupportReview,
                        agent="auto_draft_gold_support",
                    )
                    for client in self.review_clients
                )
            )
        )

    async def _author_claims(
        self,
        sample: BenchmarkSample,
        documents: list[SourceDoc],
        *,
        conservative_retry: bool = False,
    ) -> list[AtomicClaim]:
        """Author answer-relevant atomic claims before independent support review."""

        source_payload = _compact_source_payload(sample, documents)
        claim_ids = [
            f"{sample.id}.c{index}"
            for index in range(1, len(sample.expected_claims) + 1)
        ]
        payload = {
            "question": sample.question,
            "candidate_answer_requirements": sample.expected_claims,
            "claim_ids_to_use_once_each": claim_ids,
            "sources": source_payload,
        }
        messages = [
            ChatMessage(
                role="system",
                content=(
                    "Author an engineering Benchmark gold draft from supplied frozen sources. "
                    "Return JSON exactly as {\"claims\":[{\"claim_id\":\"...\","
                    "\"text\":\"...\",\"source_id\":\"...\",\"quote\":\"...\"}]}. "
                    "Use every requested claim_id exactly once. Each text must be one complete, "
                    "answer-relevant factual proposition, stay within what its one source directly "
                    "supports, and address the candidate requirements when evidence permits. quote "
                    "must be an exact contiguous copy from source content. Do not add other keys."
                    + (
                        " This is a retry after support rejection: make each claim narrower, use "
                        "different direct source passages when possible, and do not combine facts."
                        if conservative_retry
                        else ""
                    )
                ),
            ),
            ChatMessage(role="user", content=json.dumps(payload, ensure_ascii=False)),
        ]
        authored = await self.review_clients[0].json_chat(
            messages,
            DraftClaimAuthoring,
            agent="auto_draft_claim_author",
        )
        by_id = {item.claim_id: item for item in authored.claims}
        if set(by_id) != set(claim_ids) or len(by_id) != len(authored.claims):
            raise ValueError("Claim authoring did not return each stable claim ID once.")
        documents_by_id = {item.id: item for item in documents}
        claims: list[AtomicClaim] = []
        for claim_id in claim_ids:
            item = by_id[claim_id]
            document = documents_by_id.get(item.source_id)
            if document is None:
                raise ValueError(f"Claim authoring used unknown source {item.source_id}.")
            grounded_quote = _recover_exact_quote(document.content, item.quote)
            if grounded_quote is None:
                raise ValueError(f"Claim authoring quote is not an exact span for {claim_id}.")
            claims.append(
                AtomicClaim(
                    id=claim_id,
                    text=item.text.strip(),
                    satisfaction_routes=[
                        AtomicClaimSatisfactionRoute(
                            id=f"{claim_id}.main",
                            text=item.text.strip(),
                        )
                    ],
                    weight=1.0,
                    required=True,
                )
            )
        return claims

    @staticmethod
    def _validate_reviews(
        claims: list[AtomicClaim],
        documents: list[SourceDoc],
        reviews: list[DraftGoldSupportReview],
    ) -> tuple[dict[str, list[str]], list[str]]:
        expected = {item.id for item in claims}
        docs = {item.id: item for item in documents}
        support_ids: dict[str, list[str]] = {item.id: [] for item in claims}
        invalid_claim_ids: set[str] = set()
        for review_index, review in enumerate(reviews, start=1):
            by_claim: dict[str, DraftClaimSupport] = {}
            for item in review.claims:
                claim_id = item.claim_id
                if claim_id not in expected:
                    matches = [
                        expected_id
                        for expected_id in expected
                        if expected_id.endswith(f".{claim_id}")
                    ]
                    if len(matches) == 1:
                        claim_id = matches[0]
                by_claim[claim_id] = item
            if set(by_claim) != expected or len(by_claim) != len(review.claims):
                raise ValueError(f"Review {review_index} did not return each stable claim ID once.")
            for claim_id in sorted(expected):
                item = by_claim[claim_id]
                if item.verdict != "supported" or not item.source_id or not item.quote:
                    invalid_claim_ids.add(claim_id)
                    continue
                document = docs.get(item.source_id)
                if document is None:
                    invalid_claim_ids.add(claim_id)
                    continue
                grounded_quote = _recover_exact_quote(document.content, item.quote)
                if grounded_quote is None:
                    invalid_claim_ids.add(claim_id)
                    continue
                # Persist the recovered source bytes, not the model's trimmed or
                # prefixed copy, so every review artifact remains exactly grounded.
                item.quote = grounded_quote
                if item.source_id not in support_ids[claim_id]:
                    support_ids[claim_id].append(item.source_id)
        accepted = {
            claim_id: source_ids
            for claim_id, source_ids in support_ids.items()
            if claim_id not in invalid_claim_ids and len(source_ids) >= 1
        }
        dropped = sorted(expected - set(accepted))
        return accepted, dropped

    @staticmethod
    def _queries(sample: BenchmarkSample) -> list[str]:
        queries = [sample.question]
        queries.extend(
            f"{sample.question} {requirement}" for requirement in sample.required_sources
        )
        if _sample_requires_primary(sample):
            queries.append(
                " ".join(
                    [
                        sample.question,
                        "original paper official specification technical report arXiv RFC PDF",
                    ]
                )
            )
        return list(dict.fromkeys(queries))

    @staticmethod
    def _evidence_requirements(
        sample: BenchmarkSample,
        claims: list[AtomicClaim],
        support_ids: dict[str, list[str]],
        documents: list[SourceDoc],
        *,
        primary_required: bool,
    ) -> list[EvidenceRequirement]:
        documents_by_id = {item.id: item for item in documents}
        primary_claim_ids = {
            claim.id
            for claim in claims
            if any(
                source_id in documents_by_id
                and _is_primary_document(documents_by_id[source_id])
                for source_id in support_ids[claim.id]
            )
        }
        # 快照级门禁已确保适用题含一手来源；只有当该来源也直接支持某条 gold claim 时，
        # 才把更强的 primary_source_required 绑定到 requirement，不能凭来源存在硬绑定。
        primary_requirement_claim = next(iter(sorted(primary_claim_ids)), None)
        requirements: list[EvidenceRequirement] = []
        for index, claim in enumerate(claims, start=1):
            source_ids = support_ids[claim.id]
            requirements.append(
                EvidenceRequirement(
                    id=f"{sample.id}.e{index}",
                    description=f"Frozen evidence directly supporting: {claim.text}",
                    claim_ids=[claim.id],
                    required_source_ids=source_ids,
                    min_independent_sources=1,
                    primary_source_required=(
                        primary_required and claim.id == primary_requirement_claim
                    ),
                    accepted_source_types=[
                        "primary",
                        "research_paper",
                        "official_document",
                        "technical_document",
                        "independent_analysis",
                    ],
                )
            )
        return requirements

    @staticmethod
    def _rubric(sample: BenchmarkSample) -> list[RubricCriterion]:
        dimensions = (
            "information_recall",
            "analysis_reasoning",
            "citation_quality",
            "task_following",
            "presentation",
        )
        return [
            RubricCriterion(
                id=f"{sample.id}.r{index}",
                dimension=dimensions[(index - 1) % len(dimensions)],
                description=text,
                weight=1.0,
                evaluation_mode="hybrid",
                anchors=[
                    RubricAnchor(score=0, description=f"Does not satisfy: {text}"),
                    RubricAnchor(score=4, description=f"Fully and accurately satisfies: {text}"),
                ],
            )
            for index, text in enumerate(sample.judge_rubric, start=1)
        ]


def _sample_requires_primary(sample: BenchmarkSample) -> bool:
    text = " ".join(sample.required_sources).lower()
    return any(hint in text for hint in _PRIMARY_HINTS)


def _normalized_review_round(
    *,
    round_name: str,
    claims: list[AtomicClaim],
    reviews: list[DraftGoldSupportReview],
    support_ids: dict[str, list[str]],
    dropped_claim_ids: list[str],
) -> dict[str, object]:
    """Return the compact, auditable review record persisted beside a draft.

    This intentionally excludes prompts, full source bodies, request IDs, and raw
    responses so the Benchmark artifact cannot become a second LLM trace store.
    """

    return {
        "round": round_name,
        "claims": [item.model_dump(mode="json") for item in claims],
        "reviews": [item.model_dump(mode="json") for item in reviews],
        "accepted_source_ids": support_ids,
        "dropped_claim_ids": dropped_claim_ids,
    }


def _domain(url: str) -> str:
    return (urlsplit(url).hostname or "").lower()


def _is_primary_document(document: SourceDoc) -> bool:
    host = _domain(document.url)
    signals = set(document.credibility.signals if document.credibility else [])
    if {"institutional_domain", "primary_research_repository"}.intersection(signals):
        return True
    return any(host == item or host.endswith(f".{item}") for item in _OFFICIAL_HOSTS)


def _trace_tokens(traces: list[object]) -> int:
    """Sum provider-reported usage without depending on a concrete trace class."""

    return sum(
        int(getattr(trace, "input_tokens", 0) or 0)
        + int(getattr(trace, "output_tokens", 0) or 0)
        for trace in traces
    )


def _recover_exact_quote(content: str, candidate: str) -> str | None:
    """Recover an exact span only by conservatively trimming quote edges.

    Models occasionally prepend "It is" or append commentary to an otherwise
    verbatim quote.  This helper never changes an interior character: at least
    80% and 40 characters of the candidate must survive as an exact source span.
    """

    grounded = _ground_quote(content, candidate)
    if grounded is not None:
        return grounded
    quote = candidate.strip()
    minimum = max(40, int(len(quote) * 0.8))
    if len(quote) <= minimum:
        return None
    max_trim = len(quote) - minimum
    for total_trim in range(1, max_trim + 1):
        for left_trim in range(total_trim + 1):
            right_trim = total_trim - left_trim
            end = len(quote) - right_trim if right_trim else len(quote)
            trimmed = quote[left_trim:end].strip(" \t\r\n,;:-")
            if len(trimmed) < minimum:
                continue
            grounded = _ground_quote(content, trimmed)
            if grounded is not None:
                return grounded
    return None


def _compact_source_payload(
    sample: BenchmarkSample,
    documents: list[SourceDoc],
    *,
    max_documents: int = 6,
    max_chars_per_document: int = 1200,
) -> list[dict[str, str]]:
    """Select short, contiguous source windows for budgeted Gold review.

    The frozen snapshot still retains the entire SourceDoc.  Returning exact
    slices here keeps author/reviewer inputs bounded while allowing every quote
    to be verified later against the untouched full text.
    """

    query_text = " ".join([sample.question, *sample.expected_claims])
    terms = _relevance_terms(query_text)
    selected_documents = _select_review_documents(documents, limit=max_documents)
    return [
        {
            "source_id": document.id,
            "title": document.title,
            "url": document.url,
            "content": _best_contiguous_window(
                document.content,
                terms,
                max_chars=max_chars_per_document,
            ),
        }
        for document in selected_documents
    ]


def _select_review_documents(documents: list[SourceDoc], *, limit: int) -> list[SourceDoc]:
    """Prefer primary/high-authority evidence without collapsing domain diversity."""

    ranked = sorted(
        enumerate(documents),
        key=lambda item: (
            -int(_is_primary_document(item[1])),
            -float(item[1].credibility.score if item[1].credibility else 0.0),
            -float(item[1].score or 0.0),
            item[0],
        ),
    )
    selected: list[SourceDoc] = []
    selected_ids: set[str] = set()
    domains: set[str] = set()
    # First pass maximizes independent domains among the strongest sources.
    for _, document in ranked:
        domain = _domain(document.url)
        if domain and domain in domains:
            continue
        selected.append(document)
        selected_ids.add(document.id)
        if domain:
            domains.add(domain)
        if len(selected) == limit:
            return selected
    # A sparse domain set should not reduce the number of review sources.
    for _, document in ranked:
        if document.id in selected_ids:
            continue
        selected.append(document)
        if len(selected) == limit:
            break
    return selected


def _relevance_terms(text: str) -> set[str]:
    normalized = text.lower()
    terms = {
        item
        for item in re.findall(r"[a-z0-9][a-z0-9_.+-]{2,}", normalized)
        if item not in {"what", "which", "with", "from", "into", "about"}
    }
    # Chinese bigrams make relevance selection useful even when the exact long
    # authored requirement does not occur verbatim in an English or Chinese page.
    for run in re.findall(r"[\u4e00-\u9fff]{2,}", normalized):
        terms.update(run[index : index + 2] for index in range(len(run) - 1))
    return terms


def _best_contiguous_window(content: str, terms: set[str], *, max_chars: int) -> str:
    if len(content) <= max_chars:
        return content
    stride = max(1, max_chars // 2)
    candidates = [
        content[start : start + max_chars]
        for start in range(0, len(content), stride)
    ]
    # Prefer the earliest window on equal scores to keep the selection stable.
    return max(
        enumerate(candidates),
        key=lambda item: (
            sum(term in item[1].lower() for term in terms),
            -item[0],
        ),
    )[1]

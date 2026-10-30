"""Rule fallback and LLM-as-Judge evaluation over reports plus their Evidence table."""

from __future__ import annotations

import asyncio
import random
from typing import TYPE_CHECKING, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from deepresearch.llm import ChatMessage, LLMClient
from deepresearch.schema import EvalResult, ResearchReport, ResearchTask
from deepresearch.judge_protocol import (
    JUDGE_PROTOCOL_VERSION,
    judge_protocol_fingerprint,
    judge_system_prompt,
)

if TYPE_CHECKING:
    from deepresearch.evaluation.benchmark import AtomicClaim
    from deepresearch.evaluation.claims import AtomicReportClaim


class JudgeClaimAssessment(BaseModel):
    """Evidence-grounded judgment for one report claim."""

    # 1-based 索引便于多 Judge 对齐同一条关键发现；旧模型输出可省略并按文本对齐。
    claim_index: int | None = Field(default=None, ge=1)
    claim_id: str | None = None
    claim: str
    status: Literal[
        "supported",
        "partially_supported",
        "unsupported",
        "contradicted",
        "miscited",
        "uncited",
    ]
    evidence_ids: list[str] = Field(default_factory=list)
    reason: str = ""


class JudgeGoldClaimAssessment(BaseModel):
    """Judge decision about whether one authored benchmark claim is covered."""

    gold_claim_id: str
    status: Literal["covered", "partially_covered", "not_covered", "contradicted"]
    report_claim_ids: list[str] = Field(default_factory=list)
    # Judge v5 requires a concrete complete route for covered decisions.
    matched_route_id: str | None = None
    reason: str = ""


class LLMJudgeOutput(BaseModel):
    """Five-dimensional judge output required by PLAN.md."""

    factuality: int = Field(ge=1, le=5)
    completeness: int = Field(ge=1, le=5)
    citation_quality: int = Field(ge=1, le=5)
    logical_structure: int = Field(ge=1, le=5)
    writing_quality: int = Field(ge=1, le=5)
    overall_reason: str
    # 至少逐条核验一个关键结论；只返回总分无法支撑引用精度与幻觉率指标。
    # 该约束会让统一 LLM 适配层重试无效 JSON，重试仍失败时再走规则回退。
    claim_assessments: list[JudgeClaimAssessment] = Field(min_length=1)
    # Legacy benchmark calls have no authored claim IDs, so this list may be empty.
    gold_claim_assessments: list[JudgeGoldClaimAssessment] = Field(default_factory=list)


class LLMEvaluatorAgent:
    """Judge report quality using only the question, rubric, report, and supplied Evidence."""

    def __init__(
        self,
        llm: LLMClient,
        *,
        fallback_to_rule: bool = True,
        protocol_version: str = JUDGE_PROTOCOL_VERSION,
        shuffle_claims: bool = True,
    ) -> None:
        self.llm = llm
        self.fallback_to_rule = fallback_to_rule
        self.protocol_version = protocol_version
        self.protocol_fingerprint = judge_protocol_fingerprint(protocol_version)
        self.system_prompt = judge_system_prompt(protocol_version)
        self.shuffle_claims = shuffle_claims

    def evaluate(
        self,
        sample_id: str,
        task: ResearchTask,
        report: ResearchReport,
        expected_claims: list[str],
        latency_seconds: float,
        *,
        gold_claims: list[AtomicClaim] | None = None,
        claim_order_seed: int | None = None,
    ) -> EvalResult:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.evaluate_async(
                    sample_id,
                    task,
                    report,
                    expected_claims,
                    latency_seconds,
                    gold_claims=gold_claims,
                    claim_order_seed=claim_order_seed,
                )
            )
        raise RuntimeError("LLMEvaluatorAgent.evaluate() cannot run inside an event loop.")

    async def evaluate_async(
        self,
        sample_id: str,
        task: ResearchTask,
        report: ResearchReport,
        expected_claims: list[str],
        latency_seconds: float,
        *,
        gold_claims: list[AtomicClaim] | None = None,
        claim_order_seed: int | None = None,
    ) -> EvalResult:
        # 记录本次调用前的 trace 游标，便于把安全的资源摘要写入评测产物；
        # 完整 prompt/response 仍只保存在独立 llm_traces 中。
        trace_offset = self.llm.trace_collector.mark()
        alignment_retry_used = False
        try:
            # 延迟导入，避免 agents 包与 evaluation facade 初始化时形成循环依赖。
            from deepresearch.evaluation.claims import extract_atomic_claims

            atomic_claims = extract_atomic_claims(report, require_citations=True)
            presented_claims = list(atomic_claims)
            if self.shuffle_claims and claim_order_seed is not None:
                random.Random(claim_order_seed).shuffle(presented_claims)
            messages = self._build_messages(
                task,
                report,
                expected_claims,
                presented_claims,
                gold_claims or [],
                system_prompt=self.system_prompt,
            )
            output = await self.llm.json_chat(
                messages,
                LLMJudgeOutput,
                agent="evaluator",
            )
            try:
                assessed, effective_assessments, judged_gold = self._validate_and_align_output(
                    output,
                    report,
                    atomic_claims,
                    presented_claims,
                    gold_claims or [],
                    require_ids=(claim_order_seed is not None and self.shuffle_claims),
                    require_route_match=self.protocol_version == "5.0",
                )
            except ValueError as alignment_error:
                # JSON schema validation cannot detect omitted or misaligned stable IDs. Give the
                # same Judge one bounded correction turn before degrading to deterministic rules.
                alignment_retry_used = True
                retry_messages = self._build_alignment_retry_messages(
                    messages,
                    output,
                    atomic_claims,
                    gold_claims or [],
                    alignment_error,
                )
                try:
                    output = await self.llm.json_chat(
                        retry_messages,
                        LLMJudgeOutput,
                        agent="evaluator_alignment_retry",
                    )
                    assessed, effective_assessments, judged_gold = (
                        self._validate_and_align_output(
                            output,
                            report,
                            atomic_claims,
                            presented_claims,
                            gold_claims or [],
                            require_ids=(
                                claim_order_seed is not None and self.shuffle_claims
                            ),
                            require_route_match=self.protocol_version == "5.0",
                        )
                    )
                except Exception as retry_error:
                    # Preserve the first precise alignment error in rule-fallback metadata; the
                    # second exception remains available as its chained cause in strict mode.
                    raise alignment_error from retry_error
            deterministic = self._rule_score(
                sample_id,
                report,
                expected_claims,
                latency_seconds,
                gold_claims=gold_claims,
            )
            domains = {
                urlparse(item.source_url).netloc or item.source_url for item in report.evidence
            }
            cited_count = sum(bool(item.citation_numbers) for item in atomic_claims)
            support_weight = sum(
                1.0
                if item.status == "supported"
                else 0.5
                if item.status == "partially_supported"
                else 0.0
                for item in effective_assessments
            )
            citation_correctness = support_weight / cited_count if cited_count else 0.0
            citation_completeness = (
                support_weight / len(atomic_claims) if atomic_claims else 0.0
            )
            hallucination_weight = sum(
                0.5
                if item.status == "partially_supported"
                else 1.0
                if item.status in {"unsupported", "contradicted", "miscited", "uncited"}
                else 0.0
                for item in effective_assessments
            )
            hallucination_rate = (
                hallucination_weight / len(atomic_claims) if atomic_claims else 1.0
            )
            gold_by_id = {item.id: item for item in gold_claims or []}
            judged_weighted_recall = self._weighted_judge_gold_recall(
                judged_gold,
                gold_claims or [],
            )
            return EvalResult(
                sample_id=sample_id,
                run_id=report.run_id,
                fact_accuracy=output.factuality / 5.0,
                weighted_claim_recall=(
                    judged_weighted_recall
                    if gold_claims is not None
                    else deterministic.weighted_claim_recall
                ),
                citation_coverage=deterministic.citation_coverage,
                citation_precision=citation_correctness,
                citation_existence=deterministic.citation_existence,
                citation_correctness=citation_correctness,
                citation_completeness=citation_completeness,
                hallucination_rate=hallucination_rate,
                answer_completeness=output.completeness / 5.0,
                latency_seconds=latency_seconds,
                metadata={
                    "source_diversity": len(domains),
                    "evaluator": "llm_judge",
                    "metric_protocol": "claim_citation_v2",
                    "support_method": "llm_judge",
                    "judge_protocol_version": self.protocol_version,
                    "judge_protocol_fingerprint": self.protocol_fingerprint,
                    "claim_order_seed": claim_order_seed,
                    "judge_profile": self.llm.profile_name,
                    "judge_model": self.llm.model,
                    "judge_provider": self.llm.provider_name,
                    "judge_call_resource": self._judge_call_resource(trace_offset),
                    "judge_alignment_retry_used": alignment_retry_used,
                    "judge_scores": {
                        "factuality": output.factuality,
                        "completeness": output.completeness,
                        "citation_quality": output.citation_quality,
                        "logical_structure": output.logical_structure,
                        "writing_quality": output.writing_quality,
                    },
                    "judge_reason": output.overall_reason,
                    "claim_assessments": [
                        item.model_dump(mode="json") for item in effective_assessments
                    ],
                    "raw_claim_assessments": [
                        item.model_dump(mode="json") for item in assessed
                    ],
                    "gold_claim_assessments": [
                        {
                            **item.model_dump(mode="json"),
                            "required": gold_by_id[item.gold_claim_id].required,
                            "weight": gold_by_id[item.gold_claim_id].weight,
                        }
                        for item in judged_gold
                    ],
                },
            )
        except Exception as exc:
            if not self.fallback_to_rule:
                raise
            result = self._rule_score(
                sample_id,
                report,
                expected_claims,
                latency_seconds,
                gold_claims=gold_claims,
            )
            result.metadata.update(
                {
                    "evaluator": "rule_fallback",
                    # 聚合层必须知道失败来自哪个 profile，才能排除该票并判断
                    # 某个独立模型是否仍达到最少成功重复数。
                    "judge_profile": self.llm.profile_name,
                    "judge_model": self.llm.model,
                    "judge_provider": self.llm.provider_name,
                    "judge_protocol_version": self.protocol_version,
                    "judge_protocol_fingerprint": self.protocol_fingerprint,
                    "claim_order_seed": claim_order_seed,
                    "judge_call_resource": self._judge_call_resource(trace_offset),
                    "judge_alignment_retry_used": alignment_retry_used,
                    "judge_error": f"{type(exc).__name__}: {exc}",
                }
            )
            return result

    def _judge_call_resource(self, trace_offset: int) -> dict[str, object]:
        """Return a credential-free resource summary for the current Judge call."""

        traces = self.llm.trace_collector.since(trace_offset)
        if not traces:
            return {}
        trace = traces[-1]
        return {
            "call_id": trace.call_id,
            "status": trace.status,
            "attempt_count": trace.attempt_count,
            "input_tokens": trace.input_tokens,
            "output_tokens": trace.output_tokens,
            "latency_seconds": trace.latency_seconds,
            "cost_usd": trace.cost_usd,
            "finish_reason": trace.finish_reason,
            "trace_path": trace.trace_path,
        }

    @staticmethod
    def _rule_score(
        sample_id: str,
        report: ResearchReport,
        expected_claims: list[str],
        latency_seconds: float,
        *,
        gold_claims: list[AtomicClaim] | None = None,
    ) -> EvalResult:
        # 延迟导入避免 agents 包初始化时与 evaluation.runner 形成循环依赖。
        from deepresearch.evaluation.metrics import score_report

        return score_report(
            sample_id,
            report,
            expected_claims,
            latency_seconds,
            gold_claims=gold_claims,
        )

    @staticmethod
    def _build_messages(
        task: ResearchTask,
        report: ResearchReport,
        expected_claims: list[str],
        atomic_claims: list[AtomicReportClaim],
        gold_claims: list[AtomicClaim],
        *,
        system_prompt: str,
    ) -> list[ChatMessage]:
        evidence_text = "\n".join(
            (
                f"- id={item.id}; claim={item.claim}; quote={item.quote}; "
                f"source={item.source_title}; url={item.source_url}"
            )
            for item in report.evidence
        ) or "- no evidence"
        claims_text = "\n".join(
            (
                f"{index}. id={item.id}; claim={item.text}; citations={item.citation_numbers}; "
                f"mapped_evidence_ids={item.evidence_ids}; invalid_citations={item.invalid_citation_numbers}"
            )
            for index, item in enumerate(atomic_claims, start=1)
        ) or "- no atomic claims"
        gold_claims_text = "\n".join(
            (
                f"- id={item.id}; text={item.text}; "
                f"satisfaction_routes={[route.model_dump(mode='json') for route in item.effective_satisfaction_routes()]}; "
                f"accepted_variants={item.accepted_variants}; "
                f"weight={item.weight}; required={item.required}"
            )
            for item in gold_claims
        ) or "- no authored gold claims"
        user = (
            f"Question: {task.query}\n"
            f"Expected rubric points: {expected_claims or ['not supplied']}\n\n"
            f"Report:\n{report.markdown}\n\n"
            f"Atomic claims to assess:\n{claims_text}\n\n"
            f"Gold claims to assess for coverage:\n{gold_claims_text}\n\n"
            f"Evidence:\n{evidence_text}"
        )
        return [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=user),
        ]

    @classmethod
    def _validate_and_align_output(
        cls,
        output: LLMJudgeOutput,
        report: ResearchReport,
        atomic_claims: list[AtomicReportClaim],
        presented_claims: list[AtomicReportClaim],
        gold_claims: list[AtomicClaim],
        *,
        require_ids: bool,
        require_route_match: bool,
    ) -> tuple[
        list[JudgeClaimAssessment],
        list[JudgeClaimAssessment],
        list[JudgeGoldClaimAssessment],
    ]:
        """Validate semantic Judge pointers and align shuffled rows by stable IDs."""

        known_ids = {item.id for item in report.evidence}
        for assessment in output.claim_assessments:
            unknown = set(assessment.evidence_ids) - known_ids
            if unknown:
                raise ValueError(f"Judge referenced unknown Evidence IDs: {sorted(unknown)}")
        assessed = cls._align_assessments(
            output.claim_assessments,
            atomic_claims,
            presented_claims=presented_claims,
            require_ids=require_ids,
        )
        effective = cls._effective_assessments(assessed, atomic_claims)
        judged_gold = cls._align_gold_assessments(
            output.gold_claim_assessments,
            gold_claims,
            atomic_claims,
            require_route_match=require_route_match,
        )
        return assessed, effective, judged_gold

    @staticmethod
    def _build_alignment_retry_messages(
        original_messages: list[ChatMessage],
        output: LLMJudgeOutput,
        atomic_claims: list[AtomicReportClaim],
        gold_claims: list[AtomicClaim],
        error: ValueError,
    ) -> list[ChatMessage]:
        """Request one complete replacement after a schema-valid but misaligned response."""

        report_ids = [item.id for item in atomic_claims]
        gold_ids = [item.id for item in gold_claims]
        route_ids = {
            item.id: [route.id for route in item.effective_satisfaction_routes()]
            for item in gold_claims
        }
        allowed_evidence = {
            item.id: item.evidence_ids for item in atomic_claims
        }
        correction = (
            "Your previous JSON passed schema validation but failed stable-ID alignment: "
            f"{error}. Return a complete replacement JSON object, not a patch. "
            "Assess every report claim exactly once, with no missing, duplicate, or extra IDs. "
            "Assess every gold claim exactly once, with no missing, duplicate, or extra IDs. "
            f"Required report claim IDs: {report_ids}. Required gold claim IDs: {gold_ids}. "
            f"Allowed satisfaction route IDs per gold claim: {route_ids}. "
            f"Allowed Evidence IDs per report claim: {allowed_evidence}. "
            "Do not cite Evidence outside the corresponding report claim mapping. "
            "Under Judge v5, covered requires one valid matched_route_id and at least one "
            "report_claim_id; every other gold status requires matched_route_id=null."
        )
        return [
            *original_messages,
            ChatMessage(role="assistant", content=output.model_dump_json()),
            ChatMessage(role="user", content=correction),
        ]

    @staticmethod
    def _align_assessments(
        assessments: list[JudgeClaimAssessment],
        claims: list[AtomicReportClaim],
        *,
        presented_claims: list[AtomicReportClaim] | None = None,
        require_ids: bool = False,
    ) -> list[JudgeClaimAssessment]:
        """Require one auditable Judge decision for every extracted atomic claim."""

        if len(assessments) != len(claims):
            raise ValueError(
                "Judge must assess every atomic claim exactly once: "
                f"expected {len(claims)}, got {len(assessments)}."
            )
        presented = presented_claims or claims
        presented_by_id = {item.id: item for item in presented}
        if require_ids and any(not item.claim_id for item in assessments):
            raise ValueError("Shuffled Judge protocol requires claim_id on every assessment.")
        by_id = {item.claim_id: item for item in assessments if item.claim_id}
        if by_id and (
            len(by_id) != len(assessments) or set(by_id) != set(presented_by_id)
        ):
            raise ValueError("Judge claim IDs do not exactly match supplied report claim IDs.")
        aligned: list[JudgeClaimAssessment] = []
        positional = dict(zip((item.id for item in presented), assessments, strict=True))
        for index, claim in enumerate(claims, start=1):
            assessment = by_id.get(claim.id) if by_id else positional[claim.id]
            assert assessment is not None
            if assessment.claim_index not in {None, index}:
                # claim_id 是随机顺序协议下的唯一对齐依据，展示索引不再作为报告顺序。
                if not by_id:
                    raise ValueError(
                        f"Judge claim_index mismatch: expected {index}, got {assessment.claim_index}."
                    )
            if assessment.claim_id not in {None, claim.id}:
                raise ValueError(
                    f"Judge claim_id mismatch: expected {claim.id}, got {assessment.claim_id}."
                )
            non_cited_ids = sorted(set(assessment.evidence_ids) - set(claim.evidence_ids))
            if non_cited_ids:
                raise ValueError(
                    f"Judge used Evidence not cited by claim '{claim.id}': {non_cited_ids}"
                )
            aligned.append(
                assessment.model_copy(
                    update={"claim_index": index, "claim_id": claim.id, "claim": claim.text}
                )
            )
        return aligned

    @staticmethod
    def _align_gold_assessments(
        assessments: list[JudgeGoldClaimAssessment],
        gold_claims: list[AtomicClaim],
        report_claims: list[AtomicReportClaim],
        *,
        require_route_match: bool = False,
    ) -> list[JudgeGoldClaimAssessment]:
        if not gold_claims:
            return []
        expected_ids = {item.id for item in gold_claims}
        by_id = {item.gold_claim_id: item for item in assessments}
        if len(assessments) != len(by_id) or set(by_id) != expected_ids:
            raise ValueError("Judge gold claim IDs do not exactly match supplied gold claims.")
        report_ids = {item.id for item in report_claims}
        aligned: list[JudgeGoldClaimAssessment] = []
        for gold in gold_claims:
            item = by_id[gold.id]
            unknown = sorted(set(item.report_claim_ids) - report_ids)
            if unknown:
                raise ValueError(
                    f"Judge gold claim '{gold.id}' referenced unknown report claims: {unknown}"
                )
            if require_route_match:
                allowed_routes = {
                    route.id for route in gold.effective_satisfaction_routes()
                }
                if item.status == "covered":
                    if item.matched_route_id not in allowed_routes:
                        raise ValueError(
                            f"Judge gold claim '{gold.id}' must match one valid route when covered."
                        )
                    if not item.report_claim_ids:
                        raise ValueError(
                            f"Judge gold claim '{gold.id}' needs report_claim_ids when covered."
                        )
                elif item.matched_route_id is not None:
                    raise ValueError(
                        f"Judge gold claim '{gold.id}' cannot match a route when status is "
                        f"'{item.status}'."
                    )
            aligned.append(item)
        return aligned

    @staticmethod
    def _weighted_judge_gold_recall(
        assessments: list[JudgeGoldClaimAssessment],
        gold_claims: list[AtomicClaim],
    ) -> float | None:
        if not gold_claims:
            return None
        assessment_by_id = {item.gold_claim_id: item for item in assessments}
        active = [item for item in gold_claims if item.required] or gold_claims
        total_weight = sum(item.weight for item in active)
        if total_weight == 0.0:
            return 0.0
        status_score = {
            "covered": 1.0,
            "partially_covered": 0.5,
            "not_covered": 0.0,
            "contradicted": 0.0,
        }
        return sum(
            item.weight * status_score[assessment_by_id[item.id].status]
            for item in active
        ) / total_weight

    @staticmethod
    def _effective_assessments(
        assessments: list[JudgeClaimAssessment],
        claims: list[AtomicReportClaim],
    ) -> list[JudgeClaimAssessment]:
        """Apply deterministic pointer failures before semantic Judge labels."""

        effective: list[JudgeClaimAssessment] = []
        for assessment, claim in zip(assessments, claims, strict=True):
            status = assessment.status
            reason = assessment.reason
            if not claim.citation_numbers:
                status = "uncited"
                reason = f"No citation marker. Judge note: {reason}"
            elif not claim.evidence_ids:
                status = "miscited"
                reason = f"Citation does not resolve to Evidence. Judge note: {reason}"
            effective.append(assessment.model_copy(update={"status": status, "reason": reason}))
        return effective

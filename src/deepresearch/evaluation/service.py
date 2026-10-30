"""Select deterministic or LLM-based report evaluation from AppConfig."""

from __future__ import annotations

from collections import Counter
from itertools import combinations
from statistics import mean, pstdev
from typing import Any

from deepresearch.agents.evaluator import LLMEvaluatorAgent
from deepresearch.config import AppConfig
from deepresearch.evaluation.benchmark import AtomicClaim
from deepresearch.evaluation.metrics import score_report
from deepresearch.judge_protocol import JudgeProtocolManifest, judge_protocol_fingerprint
from deepresearch.llm import LLMTraceStore, create_llm_client
from deepresearch.schema import EvalResult, ResearchReport, ResearchTask


class ReportEvaluationService:
    """Stable evaluation facade used by benchmark and comparison runners."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.agents: list[LLMEvaluatorAgent] = []
        if config.evaluation.provider == "llm":
            profiles = config.evaluation.judge_profiles or [config.evaluation.llm_profile]
            for profile in profiles:
                client = create_llm_client(config.llm, profile_name=profile)
                client.configure_raw_trace(
                    enabled=config.runtime.save_llm_io,
                    store=LLMTraceStore(
                        config.runtime.run_dir,
                        subdir=config.runtime.llm_trace_subdir,
                    ),
                )
                self.agents.append(
                    LLMEvaluatorAgent(
                        client,
                        fallback_to_rule=config.evaluation.fallback_to_rule,
                        protocol_version=config.evaluation.judge_protocol_version,
                        shuffle_claims=config.evaluation.judge_shuffle_claims,
                    )
                )
        elif config.evaluation.provider != "rule":
            raise ValueError(
                f"Unsupported evaluation provider '{config.evaluation.provider}'. "
                "Supported providers: rule, llm."
            )

    def evaluate(
        self,
        sample_id: str,
        task: ResearchTask,
        report: ResearchReport,
        expected_claims: list[str],
        latency_seconds: float,
        process_metrics: dict[str, Any] | None = None,
        gold_claims: list[AtomicClaim] | None = None,
    ) -> EvalResult:
        if not self.agents:
            result = score_report(
                sample_id,
                report,
                expected_claims,
                latency_seconds,
                gold_claims=gold_claims,
            )
        else:
            for agent in self.agents:
                agent.llm.bind_run(report.run_id, created_at=report.created_at)
            try:
                results = []
                for agent_index, agent in enumerate(self.agents):
                    for repetition in range(self.config.evaluation.judge_repetitions):
                        seed = (
                            self.config.evaluation.judge_random_seed
                            + agent_index * 10_000
                            + repetition
                        )
                        results.append(
                            agent.evaluate(
                                sample_id,
                                task,
                                report,
                                expected_claims,
                                latency_seconds,
                                gold_claims=gold_claims,
                                claim_order_seed=seed,
                            )
                        )
                result = aggregate_judge_results(
                    results,
                    aggregation_policy=self.config.evaluation.judge_aggregation_policy,
                    expected_profiles=[agent.llm.profile_name for agent in self.agents],
                    repetitions_per_profile=self.config.evaluation.judge_repetitions,
                    min_successful_per_profile=(
                        self.config.evaluation.judge_min_successful_repetitions_per_profile
                    ),
                    min_resolved_rate=(
                        self.config.evaluation.automatic_consensus_min_resolved_rate
                    ),
                    min_cross_model_agreement=(
                        self.config.evaluation.automatic_consensus_min_cross_model_agreement
                    ),
                    min_cohen_kappa=(
                        self.config.evaluation.automatic_consensus_min_cohen_kappa
                    ),
                )
            finally:
                for agent in self.agents:
                    agent.llm.unbind_run()
        if process_metrics is not None:
            resources = process_metrics.get("resources", {})
            result = result.model_copy(
                update={
                    "latency_seconds": float(
                        resources.get("latency_seconds", result.latency_seconds)
                    ),
                    "cost_usd": float(resources.get("cost_usd", result.cost_usd)),
                    "metadata": {**result.metadata, "process_metrics": process_metrics},
                }
            )
        return result

    def protocol_manifest(self) -> JudgeProtocolManifest:
        """Return the frozen, secret-free Judge settings written to benchmark artifacts."""

        profiles = (
            [agent.llm.profile_name for agent in self.agents]
            if self.agents
            else []
        )
        return JudgeProtocolManifest(
            version=self.config.evaluation.judge_protocol_version,
            fingerprint=judge_protocol_fingerprint(
                self.config.evaluation.judge_protocol_version
            ),
            judge_profiles=profiles,
            judge_models=[agent.llm.model for agent in self.agents],
            independent_model_versions=(
                len({agent.llm.model for agent in self.agents}) >= 2
            ),
            repetitions=self.config.evaluation.judge_repetitions,
            shuffle_claims=self.config.evaluation.judge_shuffle_claims,
            random_seed=self.config.evaluation.judge_random_seed,
            raw_trace_enabled=self.config.runtime.save_llm_io,
            label_aggregation_policy=self.config.evaluation.judge_aggregation_policy,
            min_successful_repetitions_per_profile=(
                self.config.evaluation.judge_min_successful_repetitions_per_profile
            ),
            automatic_gate_thresholds={
                "min_resolved_rate": (
                    self.config.evaluation.automatic_consensus_min_resolved_rate
                ),
                "min_cross_model_agreement": (
                    self.config.evaluation.automatic_consensus_min_cross_model_agreement
                ),
                "min_cohen_kappa": (
                    self.config.evaluation.automatic_consensus_min_cohen_kappa
                ),
            },
        )


def aggregate_judge_results(
    results: list[EvalResult],
    *,
    aggregation_policy: str = "majority_with_conservative_tie",
    expected_profiles: list[str | None] | None = None,
    repetitions_per_profile: int | None = None,
    min_successful_per_profile: int = 1,
    min_resolved_rate: float = 0.85,
    min_cross_model_agreement: float = 0.80,
    min_cohen_kappa: float = 0.60,
) -> EvalResult:
    """Aggregate repeated/model-diverse judges while preserving disagreement diagnostics."""

    if not results:
        raise ValueError("At least one Judge result is required.")
    if len(results) == 1 and aggregation_policy != "hierarchical_model_consensus":
        return results[0]

    if aggregation_policy == "hierarchical_model_consensus":
        return _aggregate_hierarchical_consensus(
            results,
            expected_profiles=expected_profiles,
            repetitions_per_profile=repetitions_per_profile,
            min_successful_per_profile=min_successful_per_profile,
            min_resolved_rate=min_resolved_rate,
            min_cross_model_agreement=min_cross_model_agreement,
            min_cohen_kappa=min_cohen_kappa,
        )
    if aggregation_policy != "majority_with_conservative_tie":
        raise ValueError(f"Unsupported Judge aggregation policy: {aggregation_policy}")

    metric_names = (
        "fact_accuracy",
        "weighted_claim_recall",
        "citation_coverage",
        "citation_precision",
        "citation_existence",
        "citation_correctness",
        "citation_completeness",
        "hallucination_rate",
        "answer_completeness",
    )
    averages: dict[str, float | None] = {}
    deviations: dict[str, float | None] = {}
    for name in metric_names:
        raw_values = [getattr(item, name) for item in results]
        if all(value is None for value in raw_values):
            averages[name] = None
            deviations[name] = None
            continue
        values = [float(value or 0.0) for value in raw_values]
        averages[name] = mean(values)
        deviations[name] = pstdev(values)
    agreement, kappa, comparable_claims = _claim_agreement(results)
    gold_agreement, gold_kappa, comparable_gold_claims = _gold_claim_agreement(results)
    aggregate_claims = _aggregate_label_assessments(
        results,
        field="claim_assessments",
        id_field="claim_id",
        conservative_order=(
            "supported",
            "partially_supported",
            "unsupported",
            "uncited",
            "miscited",
            "contradicted",
        ),
    )
    aggregate_gold = _aggregate_label_assessments(
        results,
        field="gold_claim_assessments",
        id_field="gold_claim_id",
        conservative_order=(
            "covered",
            "partially_covered",
            "not_covered",
            "contradicted",
        ),
    )
    return EvalResult(
        sample_id=results[0].sample_id,
        run_id=results[0].run_id,
        **averages,
        latency_seconds=results[0].latency_seconds,
        cost_usd=sum(item.cost_usd for item in results),
        metadata={
            "source_diversity": mean(
                float(item.metadata.get("source_diversity", 0)) for item in results
            ),
            "evaluator": "llm_judge_ensemble",
            "judge_count": len(results),
            "judge_profiles": [item.metadata.get("judge_profile") for item in results],
            "judge_protocol_version": results[0].metadata.get("judge_protocol_version"),
            "judge_protocol_fingerprint": results[0].metadata.get(
                "judge_protocol_fingerprint"
            ),
            "claim_order_seeds": [
                item.metadata.get("claim_order_seed") for item in results
            ],
            "judge_fallback_count": sum(
                item.metadata.get("evaluator") == "rule_fallback" for item in results
            ),
            "metric_stddev": deviations,
            "claim_status_agreement": agreement,
            "cohen_kappa": kappa,
            "comparable_claim_count": comparable_claims,
            "gold_claim_status_agreement": gold_agreement,
            "gold_claim_cohen_kappa": gold_kappa,
            "comparable_gold_claim_count": comparable_gold_claims,
            "label_aggregation_policy": "majority_with_conservative_tie",
            "aggregate_claim_assessments": aggregate_claims,
            "aggregate_gold_claim_assessments": aggregate_gold,
            "individual_judges": [item.metadata for item in results],
        },
    )


def _aggregate_hierarchical_consensus(
    results: list[EvalResult],
    *,
    expected_profiles: list[str | None] | None,
    repetitions_per_profile: int | None,
    min_successful_per_profile: int,
    min_resolved_rate: float,
    min_cross_model_agreement: float,
    min_cohen_kappa: float,
) -> EvalResult:
    """Build fail-closed profile-majority then cross-model consensus.

    Rule fallbacks remain visible for diagnostics and compatibility, but they are
    deliberately excluded from semantic voting.  This prevents a transport or
    JSON failure from silently becoming an extra model opinion.
    """

    profiles = list(
        dict.fromkeys(
            profile
            for profile in (
                expected_profiles
                or [item.metadata.get("judge_profile") for item in results]
            )
            if profile
        )
    )
    successful = [
        item for item in results if item.metadata.get("evaluator") == "llm_judge"
    ]
    fallbacks = [
        item for item in results if item.metadata.get("evaluator") == "rule_fallback"
    ]
    grouped = {
        profile: [
            item
            for item in successful
            if item.metadata.get("judge_profile") == profile
        ]
        for profile in profiles
    }
    profile_status = {
        profile: {
            "model": next(
                (
                    item.metadata.get("judge_model")
                    for item in results
                    if item.metadata.get("judge_profile") == profile
                ),
                None,
            ),
            "provider": next(
                (
                    item.metadata.get("judge_provider")
                    for item in results
                    if item.metadata.get("judge_profile") == profile
                ),
                None,
            ),
            "successful_repetitions": len(grouped[profile]),
            "fallback_repetitions": sum(
                item.metadata.get("judge_profile") == profile for item in fallbacks
            ),
            "required_successful_repetitions": min_successful_per_profile,
            "available": len(grouped[profile]) >= min_successful_per_profile,
        }
        for profile in profiles
    }
    active_models = {
        str(status["model"])
        for status in profile_status.values()
        if status.get("model")
    }
    independent_model_versions = len(active_models) >= 2
    reference_source = (
        "multi_model_automatic_consensus"
        if independent_model_versions
        else "multi_profile_same_model_automatic_consensus"
    )
    report_consensus = _hierarchical_label_consensus(
        results,
        grouped=grouped,
        profiles=profiles,
        field="claim_assessments",
        id_field="claim_id",
        conservative_order=(
            "supported",
            "partially_supported",
            "unsupported",
            "uncited",
            "miscited",
            "contradicted",
        ),
        repetitions_per_profile=repetitions_per_profile,
        min_successful_per_profile=min_successful_per_profile,
    )
    gold_consensus = _hierarchical_label_consensus(
        results,
        grouped=grouped,
        profiles=profiles,
        field="gold_claim_assessments",
        id_field="gold_claim_id",
        conservative_order=(
            "covered",
            "partially_covered",
            "not_covered",
            "contradicted",
        ),
        repetitions_per_profile=repetitions_per_profile,
        min_successful_per_profile=min_successful_per_profile,
    )
    report_diagnostics = _consensus_diagnostics(report_consensus, profiles)
    gold_diagnostics = _consensus_diagnostics(gold_consensus, profiles)
    gate = _automatic_consensus_gate(
        profile_status=profile_status,
        fallback_count=len(fallbacks),
        report=report_diagnostics,
        gold=gold_diagnostics,
        gold_items=gold_consensus,
        min_resolved_rate=min_resolved_rate,
        min_cross_model_agreement=min_cross_model_agreement,
        min_cohen_kappa=min_cohen_kappa,
    )

    metric_results = successful or results
    averages, deviations = _average_eval_metrics(metric_results)
    citation_metrics = _consensus_citation_metrics(
        report_consensus,
        fallback=averages,
    )
    gold_recall = _consensus_gold_recall(gold_consensus)
    if gold_recall is not None:
        averages["weighted_claim_recall"] = gold_recall
    averages.update(citation_metrics)

    source_diversity_values = [
        float(item.metadata.get("source_diversity", 0)) for item in metric_results
    ]
    return EvalResult(
        sample_id=results[0].sample_id,
        run_id=results[0].run_id,
        **averages,
        latency_seconds=results[0].latency_seconds,
        cost_usd=sum(item.cost_usd for item in results),
        metadata={
            "source_diversity": (
                mean(source_diversity_values) if source_diversity_values else 0.0
            ),
            "evaluator": "llm_judge_hierarchical_ensemble",
            "evaluation_mode": "fully_automatic",
            "reference_source": reference_source,
            "release_tier": "engineering_automatic",
            "human_calibrated": False,
            "formal_benchmark_eligible": False,
            "judge_count": len(successful),
            "judge_attempt_count": len(results),
            "judge_profiles": profiles,
            "judge_models": sorted(active_models),
            "independent_model_versions": independent_model_versions,
            "judge_protocol_version": results[0].metadata.get(
                "judge_protocol_version"
            ),
            "judge_protocol_fingerprint": results[0].metadata.get(
                "judge_protocol_fingerprint"
            ),
            "claim_order_seeds": [
                item.metadata.get("claim_order_seed") for item in results
            ],
            "judge_fallback_count": len(fallbacks),
            "metric_stddev": deviations,
            "claim_status_agreement": report_diagnostics["cross_model_agreement"],
            "cohen_kappa": report_diagnostics["cohen_kappa"],
            "comparable_claim_count": report_diagnostics["comparable_count"],
            "gold_claim_status_agreement": gold_diagnostics[
                "cross_model_agreement"
            ],
            "gold_claim_cohen_kappa": gold_diagnostics["cohen_kappa"],
            "comparable_gold_claim_count": gold_diagnostics["comparable_count"],
            "label_aggregation_policy": "hierarchical_model_consensus",
            "aggregate_claim_assessments": report_consensus,
            "aggregate_gold_claim_assessments": gold_consensus,
            "automatic_consensus": {
                "schema_version": "1.0",
                "reference_source": reference_source,
                "independent_model_versions": independent_model_versions,
                "profile_status": profile_status,
                "report_claims": report_diagnostics,
                "gold_claims": gold_diagnostics,
                "gate": gate,
            },
            "automatic_quality_status": (
                "pass" if gate["passed"] else "automatic_quality_unresolved"
            ),
            "individual_judges": [item.metadata for item in results],
        },
    )


def _average_eval_metrics(
    results: list[EvalResult],
) -> tuple[dict[str, float | None], dict[str, float | None]]:
    """Average compatible metrics without allowing fallback votes into consensus."""

    metric_names = (
        "fact_accuracy",
        "weighted_claim_recall",
        "citation_coverage",
        "citation_precision",
        "citation_existence",
        "citation_correctness",
        "citation_completeness",
        "hallucination_rate",
        "answer_completeness",
    )
    averages: dict[str, float | None] = {}
    deviations: dict[str, float | None] = {}
    for name in metric_names:
        raw_values = [getattr(item, name) for item in results]
        if all(value is None for value in raw_values):
            averages[name] = None
            deviations[name] = None
            continue
        values = [float(value or 0.0) for value in raw_values]
        averages[name] = mean(values)
        deviations[name] = pstdev(values)
    return averages, deviations


def _hierarchical_label_consensus(
    results: list[EvalResult],
    *,
    grouped: dict[str, list[EvalResult]],
    profiles: list[str],
    field: str,
    id_field: str,
    conservative_order: tuple[str, ...],
    repetitions_per_profile: int | None,
    min_successful_per_profile: int,
) -> list[dict[str, Any]]:
    """Aggregate one label family without treating deterministic fallback as a vote."""

    all_items: dict[str, list[tuple[EvalResult, dict[str, Any]]]] = {}
    for result in results:
        for item in result.metadata.get(field, []):
            raw_key = item.get(id_field)
            if raw_key is None and id_field == "claim_id":
                raw_key = item.get("claim_index")
            if raw_key is not None:
                all_items.setdefault(str(raw_key), []).append((result, item))

    severity = {label: index for index, label in enumerate(conservative_order)}
    aggregated: list[dict[str, Any]] = []
    for key in sorted(all_items):
        entries = all_items[key]
        model_majorities: dict[str, dict[str, Any]] = {}
        semantic_statuses: list[str] = []
        for profile in profiles:
            votes = [
                str(item.get("status", ""))
                for result in grouped.get(profile, [])
                for item in result.metadata.get(field, [])
                if str(item.get(id_field) or item.get("claim_index")) == key
            ]
            semantic_statuses.extend(votes)
            counts = Counter(votes)
            selected: str | None = None
            majority_count = 0
            tied: list[str] = []
            if len(votes) >= min_successful_per_profile and counts:
                majority_count = max(counts.values())
                tied = [label for label, count in counts.items() if count == majority_count]
                if len(tied) == 1:
                    selected = tied[0]
            model_majorities[profile] = {
                "status": selected,
                "votes": votes,
                "vote_counts": dict(sorted(counts.items())),
                "successful_repetitions": len(votes),
                "majority_count": majority_count,
                "unanimous": bool(votes) and len(set(votes)) == 1,
                "available": len(votes) >= min_successful_per_profile,
                "tie": len(tied) > 1,
            }

        available_statuses = [
            value["status"]
            for value in model_majorities.values()
            if value["status"] is not None
        ]
        all_profiles_resolved = (
            len(model_majorities) >= 2
            and len(available_statuses) == len(model_majorities)
        )
        conflict = all_profiles_resolved and len(set(available_statuses)) > 1
        if all_profiles_resolved and not conflict:
            resolution_status = "resolved"
            selected = available_statuses[0]
            exact_repetitions = repetitions_per_profile or max(
                value["successful_repetitions"]
                for value in model_majorities.values()
            )
            fully_unanimous = all(
                value["unanimous"]
                and value["successful_repetitions"] == exact_repetitions
                for value in model_majorities.values()
            )
            confidence = "high" if fully_unanimous else "medium"
            aggregation_reason = "cross_model_consensus"
        else:
            resolution_status = (
                "degraded_single_model"
                if len(available_statuses) == 1
                else "unresolved"
            )
            confidence = "low"
            candidates = available_statuses or semantic_statuses
            if not candidates:
                # Fallback labels are retained only as a compatibility output;
                # the row stays unresolved and cannot pass the automatic gate.
                candidates = [
                    str(item.get("status", "")) for _, item in entries
                ]
            selected = max(
                candidates,
                key=lambda label: severity.get(label, len(severity)),
            )
            aggregation_reason = (
                "cross_model_conflict"
                if conflict
                else "insufficient_profile_consensus"
            )

        selected_item = next(
            (
                item
                for result, item in entries
                if result.metadata.get("evaluator") == "llm_judge"
                and item.get("status") == selected
            ),
            entries[0][1],
        )
        base = dict(selected_item)
        base[id_field] = base.get(id_field, key)
        base["status"] = selected
        base["confidence"] = confidence
        base["resolution_status"] = resolution_status
        base["conflict"] = conflict
        base["aggregation_reason"] = aggregation_reason
        base["model_majorities"] = model_majorities
        base["individual_statuses"] = semantic_statuses
        base["fallback_count"] = sum(
            result.metadata.get("evaluator") == "rule_fallback"
            for result, _ in entries
        )
        if id_field == "gold_claim_id":
            report_claim_ids = sorted(
                {
                    str(claim_id)
                    for result, item in entries
                    if result.metadata.get("evaluator") == "llm_judge"
                    and item.get("status") == selected
                    for claim_id in item.get("report_claim_ids", [])
                }
            )
            base["report_claim_ids"] = (
                [] if selected == "not_covered" else report_claim_ids
            )
        aggregated.append(base)
    return aggregated


def _consensus_diagnostics(
    rows: list[dict[str, Any]], profiles: list[str]
) -> dict[str, Any]:
    """Summarize cross-model agreement over profile-level majority labels."""

    resolved_count = sum(item.get("resolution_status") == "resolved" for item in rows)
    low_count = sum(item.get("confidence") == "low" for item in rows)
    agreements: list[float] = []
    kappas: list[float] = []
    expected_agreements: list[float] = []
    comparable_count = 0
    label_distributions: dict[str, dict[str, int]] = {
        profile: dict(
            sorted(
                Counter(
                    str(item["model_majorities"][profile]["status"])
                    for item in rows
                    if profile in item.get("model_majorities", {})
                    and item["model_majorities"][profile]["status"] is not None
                ).items()
            )
        )
        for profile in profiles
    }
    for left, right in combinations(profiles, 2):
        paired = [
            (
                item["model_majorities"][left]["status"],
                item["model_majorities"][right]["status"],
            )
            for item in rows
            if left in item.get("model_majorities", {})
            and right in item.get("model_majorities", {})
            and item["model_majorities"][left]["status"] is not None
            and item["model_majorities"][right]["status"] is not None
        ]
        if not paired:
            continue
        left_values = [str(value[0]) for value in paired]
        right_values = [str(value[1]) for value in paired]
        comparable_count += len(paired)
        agreements.append(
            sum(a == b for a, b in zip(left_values, right_values, strict=True))
            / len(paired)
        )
        labels = set(left_values).union(right_values)
        expected_agreements.append(
            sum(
                (left_values.count(label) / len(left_values))
                * (right_values.count(label) / len(right_values))
                for label in labels
            )
        )
        kappas.append(_cohen_kappa(left_values, right_values))
    agreement = mean(agreements) if agreements else 0.0
    kappa = mean(kappas) if kappas else 0.0
    dominant_label_share = {
        profile: (
            max(counts.values()) / sum(counts.values()) if counts else 0.0
        )
        for profile, counts in label_distributions.items()
    }
    # Kappa can collapse when nearly every item belongs to one label.  Record
    # that prevalence effect for diagnosis, but never bypass the configured gate.
    prevalence_warning = bool(
        comparable_count
        and agreement >= 0.8
        and kappa < 0.6
        and any(share >= 0.8 for share in dominant_label_share.values())
    )
    return {
        "total_count": len(rows),
        "resolved_count": resolved_count,
        "unresolved_count": len(rows) - resolved_count,
        "resolved_rate": resolved_count / len(rows) if rows else 1.0,
        "low_confidence_count": low_count,
        "low_confidence_rate": low_count / len(rows) if rows else 0.0,
        "cross_model_agreement": agreement,
        "expected_chance_agreement": (
            mean(expected_agreements) if expected_agreements else 0.0
        ),
        "cohen_kappa": kappa,
        "comparable_count": comparable_count,
        "label_distributions": label_distributions,
        "dominant_label_share": dominant_label_share,
        "prevalence_warning": prevalence_warning,
        "prevalence_note": (
            "High observed agreement coexists with low kappa under strongly imbalanced labels; "
            "the configured kappa gate remains active."
            if prevalence_warning
            else None
        ),
    }


def _automatic_consensus_gate(
    *,
    profile_status: dict[str, dict[str, Any]],
    fallback_count: int,
    report: dict[str, Any],
    gold: dict[str, Any],
    gold_items: list[dict[str, Any]],
    min_resolved_rate: float,
    min_cross_model_agreement: float,
    min_cohen_kappa: float,
) -> dict[str, Any]:
    """Apply engineering-only automatic acceptance thresholds."""

    failures: list[str] = []
    if len(profile_status) < 2:
        failures.append("fewer_than_two_judge_profiles")
    for profile, status in profile_status.items():
        if not status["available"]:
            failures.append(f"profile_unavailable:{profile}")
    if fallback_count:
        failures.append("llm_fallback_present")
    for name, diagnostics in (("report_claims", report), ("gold_claims", gold)):
        if diagnostics["resolved_rate"] < min_resolved_rate:
            failures.append(f"{name}:resolved_rate_below_threshold")
        if diagnostics["cross_model_agreement"] < min_cross_model_agreement:
            failures.append(f"{name}:cross_model_agreement_below_threshold")
        if diagnostics["cohen_kappa"] < min_cohen_kappa:
            failures.append(f"{name}:cohen_kappa_below_threshold")
    required_unresolved = [
        str(item.get("gold_claim_id"))
        for item in gold_items
        if item.get("required") is True
        and item.get("resolution_status") != "resolved"
    ]
    if required_unresolved:
        failures.append("required_gold_claim_unresolved")
    required_incomplete = [
        str(item.get("gold_claim_id"))
        for item in gold_items
        if item.get("required") is True
        and item.get("resolution_status") == "resolved"
        and item.get("status") != "covered"
    ]
    if required_incomplete:
        # Agreement is not report quality: two Judges may reliably agree that a
        # required answer component is missing.  Such consensus must fail closed.
        failures.append("required_gold_claim_not_fully_covered")
    # Successful LLM parsing/alignment is the only source of semantic rows;
    # malformed IDs cause fallback before this point and are already fail-closed.
    structure_error_count = 0
    return {
        "passed": not failures,
        "status": "pass" if not failures else "automatic_quality_unresolved",
        "failure_reasons": failures,
        "thresholds": {
            "min_resolved_rate": min_resolved_rate,
            "min_cross_model_agreement": min_cross_model_agreement,
            "min_cohen_kappa": min_cohen_kappa,
            "max_llm_fallbacks": 0,
            "min_judge_profiles": 2,
        },
        "llm_fallback_count": fallback_count,
        "structure_error_count": structure_error_count,
        "required_gold_claim_unresolved_ids": required_unresolved,
        "required_gold_claim_incomplete_ids": required_incomplete,
        "b4_automatic_draft_allowed": not failures,
        "formal_benchmark_eligible": False,
    }


def _consensus_citation_metrics(
    rows: list[dict[str, Any]],
    *,
    fallback: dict[str, float | None],
) -> dict[str, float | None]:
    """Derive conservative citation metrics from selected consensus statuses."""

    if not rows:
        return {
            "citation_precision": fallback["citation_precision"],
            "citation_correctness": fallback["citation_correctness"],
            "citation_completeness": fallback["citation_completeness"],
            "hallucination_rate": fallback["hallucination_rate"],
        }
    support_weight = sum(
        1.0
        if item.get("status") == "supported"
        else 0.5
        if item.get("status") == "partially_supported"
        else 0.0
        for item in rows
    )
    cited_count = sum(
        item.get("status") not in {"uncited", "miscited"} for item in rows
    )
    hallucination_weight = sum(
        0.5
        if item.get("status") == "partially_supported"
        else 1.0
        if item.get("status")
        in {"unsupported", "contradicted", "miscited", "uncited"}
        else 0.0
        for item in rows
    )
    correctness = support_weight / cited_count if cited_count else 0.0
    return {
        "citation_precision": correctness,
        "citation_correctness": correctness,
        "citation_completeness": support_weight / len(rows),
        "hallucination_rate": hallucination_weight / len(rows),
    }


def _consensus_gold_recall(rows: list[dict[str, Any]]) -> float | None:
    """Calculate weighted authored-claim recall from conservative consensus labels."""

    if not rows:
        return None
    total_weight = sum(float(item.get("weight", 1.0)) for item in rows)
    if total_weight <= 0:
        return None
    covered = sum(
        float(item.get("weight", 1.0))
        * (
            1.0
            if item.get("status") == "covered"
            else 0.5
            if item.get("status") == "partially_covered"
            else 0.0
        )
        for item in rows
    )
    return covered / total_weight


def summarize_automatic_consensus(results: list[EvalResult]) -> dict[str, Any] | None:
    """Summarize per-sample automatic gates without inventing a human reference."""

    samples: list[dict[str, Any]] = []
    for result in results:
        consensus = result.metadata.get("automatic_consensus")
        if not isinstance(consensus, dict):
            continue
        gate = consensus.get("gate", {})
        samples.append(
            {
                "sample_id": result.sample_id,
                "run_id": result.run_id,
                "status": gate.get("status", "automatic_quality_unresolved"),
                "passed": bool(gate.get("passed", False)),
                "failure_reasons": list(gate.get("failure_reasons", [])),
                "reference_source": consensus.get("reference_source"),
                # Historical automatic artifacts predate the explicit marker and were
                # produced by the then-configured multi-model path.
                "independent_model_versions": bool(
                    consensus.get("independent_model_versions", True)
                ),
                "report_resolved_rate": consensus.get("report_claims", {}).get(
                    "resolved_rate"
                ),
                "gold_resolved_rate": consensus.get("gold_claims", {}).get(
                    "resolved_rate"
                ),
                "report_cross_model_agreement": consensus.get(
                    "report_claims", {}
                ).get("cross_model_agreement"),
                "gold_cross_model_agreement": consensus.get("gold_claims", {}).get(
                    "cross_model_agreement"
                ),
            }
        )
    if not samples:
        return None
    independent_model_versions = all(
        sample["independent_model_versions"] for sample in samples
    )
    reference_source = (
        "multi_model_automatic_consensus"
        if independent_model_versions
        else "multi_profile_same_model_automatic_consensus"
    )
    return {
        "schema_version": "1.0",
        "evaluation_mode": "fully_automatic",
        "reference_source": reference_source,
        "independent_model_versions": independent_model_versions,
        "release_tier": "engineering_automatic",
        "human_calibrated": False,
        "formal_benchmark_eligible": False,
        "passed": len(samples) == len(results) and all(
            sample["passed"] for sample in samples
        ),
        "b4_automatic_draft_allowed": len(samples) == len(results)
        and all(sample["passed"] for sample in samples),
        "samples": samples,
    }


def _aggregate_label_assessments(
    results: list[EvalResult],
    *,
    field: str,
    id_field: str,
    conservative_order: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Aggregate stable-ID labels by majority, resolving ties conservatively."""

    per_key: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        for item in result.metadata.get(field, []):
            raw_key = item.get(id_field)
            if raw_key is None and id_field == "claim_id":
                raw_key = item.get("claim_index")
            if raw_key is None:
                continue
            per_key.setdefault(str(raw_key), []).append(item)

    severity = {label: index for index, label in enumerate(conservative_order)}
    aggregated: list[dict[str, Any]] = []
    for key in sorted(per_key):
        items = per_key[key]
        statuses = [str(item.get("status", "")) for item in items]
        counts = Counter(statuses)
        largest = max(counts.values())
        tied = [label for label, count in counts.items() if count == largest]
        selected = max(tied, key=lambda label: severity.get(label, len(severity)))
        base = dict(items[0])
        base[id_field] = base.get(id_field, key)
        base["status"] = selected
        base["individual_statuses"] = statuses
        base["aggregation_reason"] = (
            "majority_vote" if len(tied) == 1 else "conservative_tie_break"
        )
        if id_field == "gold_claim_id":
            report_claim_ids = sorted(
                {
                    str(claim_id)
                    for item in items
                    for claim_id in item.get("report_claim_ids", [])
                }
            )
            base["report_claim_ids"] = (
                [] if selected == "not_covered" else report_claim_ids
            )
        aggregated.append(base)
    return aggregated


def _claim_agreement(results: list[EvalResult]) -> tuple[float | None, float | None, int]:
    """Compute pairwise simple agreement and Cohen's kappa over aligned claims."""

    mappings = [_assessment_mapping(item.metadata.get("claim_assessments", [])) for item in results]
    agreements: list[float] = []
    kappas: list[float] = []
    comparable = 0
    for left, right in combinations(mappings, 2):
        keys = sorted(set(left).intersection(right), key=str)
        if not keys:
            continue
        left_statuses = [left[key] for key in keys]
        right_statuses = [right[key] for key in keys]
        comparable += len(keys)
        agreements.append(
            sum(a == b for a, b in zip(left_statuses, right_statuses, strict=True)) / len(keys)
        )
        kappas.append(_cohen_kappa(left_statuses, right_statuses))
    return (
        mean(agreements) if agreements else None,
        mean(kappas) if kappas else None,
        comparable,
    )


def _gold_claim_agreement(
    results: list[EvalResult],
) -> tuple[float | None, float | None, int]:
    """Compute repeated-Judge agreement over authored gold claim coverage labels."""

    mappings = [
        {
            str(item.get("gold_claim_id")): str(item.get("status", ""))
            for item in result.metadata.get("gold_claim_assessments", [])
            if item.get("gold_claim_id")
        }
        for result in results
    ]
    agreements: list[float] = []
    kappas: list[float] = []
    comparable = 0
    for left, right in combinations(mappings, 2):
        keys = sorted(set(left).intersection(right))
        if not keys:
            continue
        left_statuses = [left[key] for key in keys]
        right_statuses = [right[key] for key in keys]
        comparable += len(keys)
        agreements.append(
            sum(a == b for a, b in zip(left_statuses, right_statuses, strict=True))
            / len(keys)
        )
        kappas.append(_cohen_kappa(left_statuses, right_statuses))
    return (
        mean(agreements) if agreements else None,
        mean(kappas) if kappas else None,
        comparable,
    )


def _assessment_mapping(assessments: list[dict[str, Any]]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in assessments:
        claim_id = item.get("claim_id")
        index = item.get("claim_index")
        key = (
            f"id:{claim_id}"
            if claim_id
            else f"index:{index}"
            if index is not None
            else f"claim:{str(item.get('claim', '')).strip().lower()}"
        )
        if key not in {"claim:", "index:None"}:
            mapping[key] = str(item.get("status", ""))
    return mapping


def _cohen_kappa(left: list[str], right: list[str]) -> float:
    if not left:
        return 0.0
    observed = sum(a == b for a, b in zip(left, right, strict=True)) / len(left)
    labels = set(left).union(right)
    expected = sum(
        (left.count(label) / len(left)) * (right.count(label) / len(right))
        for label in labels
    )
    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1.0 - expected)

"""B3 human-consensus calibration and auditable mismatch diagnostics."""

from __future__ import annotations

from collections import Counter
from typing import Any, Literal

from deepresearch.evaluation.annotations import (
    HumanAnnotationBundle,
    HumanAnnotationSet,
    HumanGoldClaimAnnotation,
    HumanReportClaimAnnotation,
    annotation_agreement,
    bundle_labels,
    calibrate_labels,
)
from deepresearch.quality import assess_finding_substance
from deepresearch.schema import Evidence

_THRESHOLDS = {
    "human_simple_agreement_min": 0.80,
    "human_cohen_kappa_min": 0.60,
    "automatic_macro_precision_min": 0.75,
    "automatic_macro_recall_min": 0.75,
}
_UNCERTAINTY_MARKERS = (
    "不确定",
    "不清楚",
    "拿不准",
    "可能",
    "unsure",
    "uncertain",
    "unclear",
    "maybe",
)


def build_b3_calibration(
    automatic: dict[str, Any],
    annotator_a: HumanAnnotationBundle,
    annotator_b: HumanAnnotationBundle,
    *,
    adjudication: HumanAnnotationBundle | None = None,
) -> dict[str, Any]:
    """Build calibration output using unanimous labels or explicit adjudication."""

    _validate_roles(annotator_a, annotator_b, adjudication)
    predicted_report, predicted_gold = automatic_labels(automatic)
    a_report = bundle_labels(annotator_a, target="report_claims")
    b_report = bundle_labels(annotator_b, target="report_claims")
    a_gold = bundle_labels(annotator_a, target="gold_claims")
    b_gold = bundle_labels(annotator_b, target="gold_claims")
    report_disagreements = _disagreements(a_report, b_report)
    gold_disagreements = _disagreements(a_gold, b_gold)
    if report_disagreements or gold_disagreements:
        if adjudication is None:
            raise ValueError(
                "A/B annotations disagree; a completed adjudication bundle is required."
            )
        reference_source = "human_adjudication"
        adjudicated_report = bundle_labels(adjudication, target="report_claims")
        adjudicated_gold = bundle_labels(adjudication, target="gold_claims")
        _require_adjudicated_disagreements(
            adjudicated_report,
            report_disagreements,
            "report claims",
        )
        _require_adjudicated_disagreements(
            adjudicated_gold,
            gold_disagreements,
            "gold claims",
        )
        # Unanimous A/B decisions remain the reference.  Only disputed keys are
        # replaced by the third human's decision, so a disagreement-only packet
        # is sufficient and the adjudicator need not relabel all 34 items.
        reference_report = dict(a_report)
        reference_report.update(
            {
                row["key"]: adjudicated_report[row["key"]]
                for row in report_disagreements
            }
        )
        reference_gold = dict(a_gold)
        reference_gold.update(
            {row["key"]: adjudicated_gold[row["key"]] for row in gold_disagreements}
        )
        reference_bundle = _merge_reference_bundle(
            annotator_a,
            adjudication,
            report_disagreements=report_disagreements,
            gold_disagreements=gold_disagreements,
        )
    else:
        reference_source = "unanimous_consensus"
        reference_report = a_report
        reference_gold = a_gold
        reference_bundle = annotator_a

    _require_matching_keys(predicted_report, reference_report, "report claims")
    _require_matching_keys(predicted_gold, reference_gold, "gold claims")
    report_agreement = annotation_agreement(
        annotator_a, annotator_b, target="report_claims"
    )
    gold_agreement = annotation_agreement(
        annotator_a, annotator_b, target="gold_claims"
    )
    report_calibration = calibrate_labels(predicted_report, reference_report)
    gold_calibration = calibrate_labels(predicted_gold, reference_gold)
    reason_stats = _reason_statistics(annotator_a, annotator_b)
    provenance_warnings: list[str] = []
    if reason_stats["identical_reason_count"] == reason_stats["total_reason_count"]:
        provenance_warnings.append(
            "A/B labels, linked report claims, and every free-text reason are identical; "
            "independence is user-attested but cannot be verified from the artifacts."
        )
        independence_attestation = "user_attested_artifacts_content_identical"
    else:
        independence_attestation = "distinct_annotator_ids_artifacts_not_identical"
    if reason_stats["uncertainty_marker_count"]:
        provenance_warnings.append(
            "Some reference reasons contain uncertainty markers; labels remain unchanged "
            "and are surfaced in mismatch diagnostics."
        )

    mismatches = {
        "report_claims": _mismatch_rows(
            predicted_report,
            reference_report,
            reference_bundle,
            target="report_claims",
        ),
        "gold_claims": _mismatch_rows(
            predicted_gold,
            reference_gold,
            reference_bundle,
            target="gold_claims",
        ),
    }
    gate = _calibration_gate(
        report_agreement.model_dump(mode="json"),
        gold_agreement.model_dump(mode="json"),
        report_calibration.model_dump(mode="json"),
        gold_calibration.model_dump(mode="json"),
    )
    substance_rows = _reference_substance_rows(reference_bundle)
    substance_counts = Counter(item["status"] for item in substance_rows)
    return {
        "schema_version": "1.1",
        "reference_source": reference_source,
        "independence_attestation": independence_attestation,
        "provenance_warnings": provenance_warnings,
        "identical_reason_count": reason_stats["identical_reason_count"],
        "total_reason_count": reason_stats["total_reason_count"],
        "uncertainty_marker_count": reason_stats["uncertainty_marker_count"],
        "uncertainty_items": reason_stats["uncertainty_items"],
        "disagreements": {
            "report_claims": report_disagreements,
            "gold_claims": gold_disagreements,
        },
        "pilot_thresholds": _THRESHOLDS,
        "human_agreement": {
            "report_claims": report_agreement.model_dump(mode="json"),
            "gold_claims": gold_agreement.model_dump(mode="json"),
        },
        "automatic_vs_reference": {
            "report_claims": report_calibration.model_dump(mode="json"),
            "gold_claims": gold_calibration.model_dump(mode="json"),
        },
        "reference_report_substance": {
            "status_counts": dict(sorted(substance_counts.items())),
            "invalid_count": sum(
                count
                for status, count in substance_counts.items()
                if status != "substantive"
            ),
            "items": substance_rows,
        },
        "mismatches": mismatches,
        "gate": gate,
    }


def build_disagreement_adjudication_bundle(
    annotator_a: HumanAnnotationBundle,
    annotator_b: HumanAnnotationBundle,
    *,
    adjudicator_id: str = "adjudicator_pending",
) -> HumanAnnotationBundle:
    """Create a blank adjudication packet containing only A/B disagreements.

    The packet intentionally omits both annotators' labels and reasons.  The
    adjudicator receives the same claim/Evidence context, but makes an
    independent decision only for disputed items.
    """

    _validate_roles(annotator_a, annotator_b, None)
    a_report = bundle_labels(annotator_a, target="report_claims")
    b_report = bundle_labels(annotator_b, target="report_claims")
    a_gold = bundle_labels(annotator_a, target="gold_claims")
    b_gold = bundle_labels(annotator_b, target="gold_claims")
    report_keys = {
        row["key"] for row in _disagreements(a_report, b_report)
    }
    gold_keys = {row["key"] for row in _disagreements(a_gold, b_gold)}
    if not report_keys and not gold_keys:
        raise ValueError("A/B annotations are unanimous; adjudication is not required.")

    samples: list[HumanAnnotationSet] = []
    for sample in annotator_a.samples:
        report_claims = []
        for item in sample.report_claims:
            key = f"{sample.sample_id}:{sample.run_id}:{item.claim_id}"
            if key in report_keys:
                report_claims.append(
                    item.model_copy(
                        deep=True,
                        update={"label": None, "reason": ""},
                    )
                )
        gold_claims = []
        for item in sample.gold_claims:
            key = f"{sample.sample_id}:{sample.run_id}:{item.gold_claim_id}"
            if key in gold_keys:
                gold_claims.append(
                    item.model_copy(
                        deep=True,
                        update={
                            "label": None,
                            "reason": "",
                            # A/B selected links are judgments too, so they must
                            # not leak into the adjudicator's blank packet.
                            "report_claim_ids": [],
                        },
                    )
                )
        if not report_claims and not gold_claims:
            continue
        samples.append(
            HumanAnnotationSet(
                benchmark_id=sample.benchmark_id,
                benchmark_version=sample.benchmark_version,
                sample_id=sample.sample_id,
                run_id=sample.run_id,
                annotator_id=adjudicator_id,
                role="adjudicator",
                status="pending",
                report_claims=report_claims,
                gold_claims=gold_claims,
                notes=[
                    "This packet contains only A/B disagreements.",
                    "A/B labels and reasons are intentionally withheld.",
                ],
            )
        )
    return HumanAnnotationBundle(
        annotator_id=adjudicator_id,
        role="adjudicator",
        samples=samples,
    )


def compare_b3_protocol_results(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    annotator_a: HumanAnnotationBundle,
    annotator_b: HumanAnnotationBundle,
) -> dict[str, Any]:
    """Compare two Judge protocols against one unchanged unanimous reference.

    This comparison deliberately keeps low-confidence human labels in the main
    metrics while preventing them from silently authorizing B4 expansion.
    """

    _validate_roles(annotator_a, annotator_b, None)
    baseline_report, baseline_gold = automatic_labels(baseline)
    candidate_report, candidate_gold = automatic_labels(candidate)
    reference_report = bundle_labels(annotator_a, target="report_claims")
    reference_gold = bundle_labels(annotator_a, target="gold_claims")
    if reference_report != bundle_labels(annotator_b, target="report_claims"):
        raise ValueError(
            "A/B report-claim labels must be unanimous for protocol comparison."
        )
    if reference_gold != bundle_labels(annotator_b, target="gold_claims"):
        raise ValueError("A/B gold-claim labels must be unanimous for protocol comparison.")
    for name, labels, reference in (
        ("baseline report claims", baseline_report, reference_report),
        ("baseline gold claims", baseline_gold, reference_gold),
        ("candidate report claims", candidate_report, reference_report),
        ("candidate gold claims", candidate_gold, reference_gold),
    ):
        _require_matching_keys(labels, reference, name)

    baseline_metrics = {
        "report_claims": calibrate_labels(
            baseline_report, reference_report
        ).model_dump(mode="json"),
        "gold_claims": calibrate_labels(
            baseline_gold, reference_gold
        ).model_dump(mode="json"),
    }
    candidate_metrics = {
        "report_claims": calibrate_labels(
            candidate_report, reference_report
        ).model_dump(mode="json"),
        "gold_claims": calibrate_labels(
            candidate_gold, reference_gold
        ).model_dump(mode="json"),
    }
    changes = _protocol_label_changes(
        baseline_report,
        candidate_report,
        reference_report,
        annotator_a,
        target="report_claims",
    ) + _protocol_label_changes(
        baseline_gold,
        candidate_gold,
        reference_gold,
        annotator_a,
        target="gold_claims",
    )
    reason_stats = _reason_statistics(annotator_a, annotator_b)
    decision_reasons: list[str] = []
    for target, metrics in candidate_metrics.items():
        if metrics["macro_precision"] < _THRESHOLDS["automatic_macro_precision_min"]:
            decision_reasons.append(f"{target}:macro_precision_below_threshold")
        if metrics["macro_recall"] < _THRESHOLDS["automatic_macro_recall_min"]:
            decision_reasons.append(f"{target}:macro_recall_below_threshold")
    if reason_stats["uncertainty_marker_count"]:
        decision_reasons.append("reference_contains_low_confidence_labels")

    return {
        "schema_version": "1.0",
        "reference_source": "unanimous_consensus",
        "baseline_protocol": _protocol_manifest(baseline),
        "candidate_protocol": _protocol_manifest(candidate),
        "baseline_metrics": baseline_metrics,
        "candidate_metrics": candidate_metrics,
        "metric_deltas": _metric_deltas(baseline_metrics, candidate_metrics),
        "changed_labels": changes,
        "change_summary": {
            "changed_count": len(changes),
            "became_reference_match": sum(
                row["candidate_matches_reference"]
                and not row["baseline_matches_reference"]
                for row in changes
            ),
            "lost_reference_match": sum(
                row["baseline_matches_reference"]
                and not row["candidate_matches_reference"]
                for row in changes
            ),
            "changed_between_non_reference_labels": sum(
                not row["baseline_matches_reference"]
                and not row["candidate_matches_reference"]
                for row in changes
            ),
        },
        "repetition_diagnostics": {
            "baseline": _repetition_diagnostics(baseline),
            "candidate": _repetition_diagnostics(candidate),
        },
        "reference_uncertainty": {
            "annotator_item_count": reason_stats["uncertainty_marker_count"],
            "items": reason_stats["uncertainty_items"],
        },
        "b4_decision": {
            "allowed": not decision_reasons,
            "reasons": decision_reasons,
        },
    }


def _protocol_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    protocol = payload.get("judge_protocol")
    if not isinstance(protocol, dict):
        raise ValueError("Judge result artifact is missing judge_protocol metadata.")
    return dict(protocol)


def _protocol_label_changes(
    baseline: dict[str, str],
    candidate: dict[str, str],
    reference: dict[str, str],
    bundle: HumanAnnotationBundle,
    *,
    target: Literal["report_claims", "gold_claims"],
) -> list[dict[str, Any]]:
    items = _annotation_items(bundle)
    rows: list[dict[str, Any]] = []
    for key in sorted(reference):
        if baseline[key] == candidate[key]:
            continue
        item = items[key]
        rows.append(
            {
                "key": key,
                "target": target,
                "baseline_label": baseline[key],
                "candidate_label": candidate[key],
                "reference_label": reference[key],
                "baseline_matches_reference": baseline[key] == reference[key],
                "candidate_matches_reference": candidate[key] == reference[key],
                "reference_reason": item.reason,
                "uncertainty_markers": _uncertainty_markers(item.reason),
            }
        )
    return rows


def _metric_deltas(
    baseline: dict[str, dict[str, Any]],
    candidate: dict[str, dict[str, Any]],
) -> dict[str, dict[str, float]]:
    return {
        target: {
            name: float(candidate[target][name]) - float(baseline[target][name])
            for name in ("accuracy", "macro_precision", "macro_recall")
        }
        for target in ("report_claims", "gold_claims")
    }


def _repetition_diagnostics(payload: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for result in payload.get("results", []):
        metadata = result.get("metadata", {})
        rows.append(
            {
                "sample_id": result.get("sample_id"),
                "judge_count": metadata.get("judge_count", 1),
                "fallback_count": metadata.get("judge_fallback_count", 0),
                "report_claim_agreement": metadata.get("claim_status_agreement"),
                "gold_claim_agreement": metadata.get("gold_claim_status_agreement"),
            }
        )
    agreement_values = [
        float(row[name])
        for row in rows
        for name in ("report_claim_agreement", "gold_claim_agreement")
        if row[name] is not None
    ]
    return {
        "samples": rows,
        "fallback_count": sum(int(row["fallback_count"] or 0) for row in rows),
        "minimum_recorded_agreement": min(agreement_values) if agreement_values else None,
    }


def automatic_labels(payload: dict[str, Any]) -> tuple[dict[str, str], dict[str, str]]:
    """Extract first-Judge labels using canonical sample/run/claim keys."""

    report_labels: dict[str, str] = {}
    gold_labels: dict[str, str] = {}
    for result in payload.get("results", []):
        sample_id = str(result["sample_id"])
        run_id = str(result["run_id"])
        metadata = result.get("metadata", {})
        aggregate_report = metadata.get("aggregate_claim_assessments")
        aggregate_gold = metadata.get("aggregate_gold_claim_assessments")
        if aggregate_report is not None or aggregate_gold is not None:
            report_assessments = aggregate_report or []
            gold_assessments = aggregate_gold or []
        elif metadata.get("individual_judges"):
            metadata = metadata["individual_judges"][0]
            report_assessments = metadata.get("claim_assessments", [])
            gold_assessments = metadata.get("gold_claim_assessments", [])
        else:
            report_assessments = metadata.get("claim_assessments", [])
            gold_assessments = metadata.get("gold_claim_assessments", [])
        for item in report_assessments:
            report_labels[f"{sample_id}:{run_id}:{item['claim_id']}"] = str(
                item["status"]
            )
        for item in gold_assessments:
            gold_labels[f"{sample_id}:{run_id}:{item['gold_claim_id']}"] = str(
                item["status"]
            )
    return report_labels, gold_labels


def _validate_roles(
    annotator_a: HumanAnnotationBundle,
    annotator_b: HumanAnnotationBundle,
    adjudication: HumanAnnotationBundle | None,
) -> None:
    if annotator_a.role != "annotator" or annotator_b.role != "annotator":
        raise ValueError("Both independent bundles must use role=annotator.")
    if annotator_a.annotator_id == annotator_b.annotator_id:
        raise ValueError("Independent annotation bundles must use different annotator IDs.")
    if adjudication is not None and adjudication.role != "adjudicator":
        raise ValueError("The adjudication bundle must use role=adjudicator.")


def _disagreements(left: dict[str, str], right: dict[str, str]) -> list[dict[str, str]]:
    if set(left) != set(right):
        raise ValueError("A/B annotation keys must match exactly.")
    return [
        {"key": key, "annotator_a": left[key], "annotator_b": right[key]}
        for key in sorted(left)
        if left[key] != right[key]
    ]


def _require_adjudicated_disagreements(
    adjudicated: dict[str, str],
    disagreements: list[dict[str, str]],
    label: str,
) -> None:
    """Require every disputed key while accepting legacy full adjudication files."""

    required = {row["key"] for row in disagreements}
    missing = sorted(required - set(adjudicated))
    if missing:
        raise ValueError(f"Adjudication is missing disputed {label}: {missing}")


def _merge_reference_bundle(
    unanimous_base: HumanAnnotationBundle,
    adjudication: HumanAnnotationBundle,
    *,
    report_disagreements: list[dict[str, str]],
    gold_disagreements: list[dict[str, str]],
) -> HumanAnnotationBundle:
    """Merge third-human decisions into a complete bundle used for diagnostics."""

    adjudicated_items = _annotation_items(adjudication)
    report_keys = {row["key"] for row in report_disagreements}
    gold_keys = {row["key"] for row in gold_disagreements}
    samples: list[HumanAnnotationSet] = []
    for sample in unanimous_base.samples:
        report_claims = []
        for item in sample.report_claims:
            key = f"{sample.sample_id}:{sample.run_id}:{item.claim_id}"
            source = adjudicated_items[key] if key in report_keys else item
            if not isinstance(source, HumanReportClaimAnnotation):
                raise ValueError(f"Adjudication item type mismatch for '{key}'.")
            report_claims.append(source.model_copy(deep=True))
        gold_claims = []
        for item in sample.gold_claims:
            key = f"{sample.sample_id}:{sample.run_id}:{item.gold_claim_id}"
            source = adjudicated_items[key] if key in gold_keys else item
            if not isinstance(source, HumanGoldClaimAnnotation):
                raise ValueError(f"Adjudication item type mismatch for '{key}'.")
            gold_claims.append(source.model_copy(deep=True))
        samples.append(
            HumanAnnotationSet(
                benchmark_id=sample.benchmark_id,
                benchmark_version=sample.benchmark_version,
                sample_id=sample.sample_id,
                run_id=sample.run_id,
                annotator_id=adjudication.annotator_id,
                role="adjudicator",
                status="complete",
                report_claims=report_claims,
                gold_claims=gold_claims,
                notes=["Unanimous A/B labels merged with disputed adjudicator labels."],
            )
        )
    return HumanAnnotationBundle(
        annotator_id=adjudication.annotator_id,
        role="adjudicator",
        samples=samples,
    )


def _require_matching_keys(
    predicted: dict[str, str], reference: dict[str, str], label: str
) -> None:
    if set(predicted) != set(reference):
        missing_predicted = sorted(set(reference) - set(predicted))
        missing_reference = sorted(set(predicted) - set(reference))
        raise ValueError(
            f"Automatic/reference {label} keys differ; "
            f"missing_predicted={missing_predicted}, missing_reference={missing_reference}"
        )


def _reason_statistics(
    left: HumanAnnotationBundle,
    right: HumanAnnotationBundle,
) -> dict[str, Any]:
    left_items = _annotation_items(left)
    right_items = _annotation_items(right)
    if set(left_items) != set(right_items):
        raise ValueError("A/B annotation item keys must match exactly.")
    identical = 0
    uncertainty_items: list[dict[str, str]] = []
    for key in sorted(left_items):
        left_item = left_items[key]
        right_item = right_items[key]
        if (
            left_item.reason.strip() == right_item.reason.strip()
            and left_item.label == right_item.label
            and getattr(left_item, "report_claim_ids", [])
            == getattr(right_item, "report_claim_ids", [])
        ):
            identical += 1
        for annotator, reason in (
            (left.annotator_id, left_item.reason),
            (right.annotator_id, right_item.reason),
        ):
            markers = _uncertainty_markers(reason)
            if markers:
                uncertainty_items.append(
                    {
                        "key": key,
                        "annotator_id": annotator,
                        "markers": ",".join(markers),
                        "reason": reason,
                    }
                )
    return {
        "identical_reason_count": identical,
        "total_reason_count": len(left_items),
        "uncertainty_marker_count": len(uncertainty_items),
        "uncertainty_items": uncertainty_items,
    }


def _annotation_items(
    bundle: HumanAnnotationBundle,
) -> dict[str, HumanReportClaimAnnotation | HumanGoldClaimAnnotation]:
    result: dict[str, HumanReportClaimAnnotation | HumanGoldClaimAnnotation] = {}
    for sample in bundle.samples:
        for item in sample.report_claims:
            result[f"{sample.sample_id}:{sample.run_id}:{item.claim_id}"] = item
        for item in sample.gold_claims:
            result[f"{sample.sample_id}:{sample.run_id}:{item.gold_claim_id}"] = item
    return result


def _uncertainty_markers(reason: str) -> list[str]:
    lowered = reason.casefold()
    return [marker for marker in _UNCERTAINTY_MARKERS if marker.casefold() in lowered]


def _mismatch_rows(
    predicted: dict[str, str],
    reference: dict[str, str],
    bundle: HumanAnnotationBundle,
    *,
    target: Literal["report_claims", "gold_claims"],
) -> list[dict[str, Any]]:
    item_by_key = _annotation_items(bundle)
    sample_report_items = {
        (sample.sample_id, sample.run_id): {item.claim_id: item for item in sample.report_claims}
        for sample in bundle.samples
    }
    rows: list[dict[str, Any]] = []
    for key in sorted(reference):
        if predicted[key] == reference[key]:
            continue
        item = item_by_key[key]
        markers = _uncertainty_markers(item.reason)
        if markers:
            category = "human_low_confidence"
        elif target == "gold_claims" and isinstance(item, HumanGoldClaimAnnotation):
            sample_id, run_id, _ = key.split(":", 2)
            linked = [
                sample_report_items[(sample_id, run_id)][claim_id]
                for claim_id in item.report_claim_ids
                if claim_id in sample_report_items[(sample_id, run_id)]
            ]
            category = (
                "report_missing"
                if not linked or all(_report_item_non_substantive(value) for value in linked)
                else "judge_error_candidate"
            )
        else:
            category = (
                "report_missing"
                if isinstance(item, HumanReportClaimAnnotation)
                and _report_item_non_substantive(item)
                else "judge_error_candidate"
            )
        rows.append(
            {
                "key": key,
                "target": target,
                "automatic_label": predicted[key],
                "reference_label": reference[key],
                "category": category,
                "reference_reason": item.reason,
                "uncertainty_markers": markers,
                "report_claim_ids": getattr(item, "report_claim_ids", []),
            }
        )
    return rows


def _report_item_non_substantive(item: HumanReportClaimAnnotation) -> bool:
    evidence = [
        Evidence(
            id=value.evidence_id,
            claim=value.claim,
            quote=value.quote,
            source_id=value.evidence_id,
            source_title=value.source_title,
            source_url=value.source_url,
        )
        for value in item.evidence
    ]
    return assess_finding_substance(
        "general research question",
        item.claim,
        finding_index=1,
        evidence=evidence,
    ).status != "substantive"


def _reference_substance_rows(bundle: HumanAnnotationBundle) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in bundle.samples:
        for item in sample.report_claims:
            evidence = [
                Evidence(
                    id=value.evidence_id,
                    claim=value.claim,
                    quote=value.quote,
                    source_id=value.evidence_id,
                    source_title=value.source_title,
                    source_url=value.source_url,
                )
                for value in item.evidence
            ]
            diagnostic = assess_finding_substance(
                "general research question",
                item.claim,
                finding_index=1,
                evidence=evidence,
            )
            rows.append(
                {
                    "key": f"{sample.sample_id}:{sample.run_id}:{item.claim_id}",
                    "status": diagnostic.status,
                    "reason": diagnostic.reason,
                }
            )
    return rows


def _calibration_gate(
    report_agreement: dict[str, Any],
    gold_agreement: dict[str, Any],
    report_calibration: dict[str, Any],
    gold_calibration: dict[str, Any],
) -> dict[str, Any]:
    failures: list[str] = []
    for name, metrics in (
        ("report_claim_human_agreement", report_agreement),
        ("gold_claim_human_agreement", gold_agreement),
    ):
        if (metrics.get("simple_agreement") or 0.0) < _THRESHOLDS[
            "human_simple_agreement_min"
        ]:
            failures.append(f"{name}:simple_agreement_below_threshold")
        if (metrics.get("cohen_kappa") or 0.0) < _THRESHOLDS["human_cohen_kappa_min"]:
            failures.append(f"{name}:cohen_kappa_below_threshold")
    for name, metrics in (
        ("report_claim_automatic", report_calibration),
        ("gold_claim_automatic", gold_calibration),
    ):
        if (metrics.get("macro_precision") or 0.0) < _THRESHOLDS[
            "automatic_macro_precision_min"
        ]:
            failures.append(f"{name}:macro_precision_below_threshold")
        if (metrics.get("macro_recall") or 0.0) < _THRESHOLDS[
            "automatic_macro_recall_min"
        ]:
            failures.append(f"{name}:macro_recall_below_threshold")
    return {
        "passed": not failures,
        "failure_reasons": failures,
        "next_stage": "B4" if not failures else "revise_protocol_or_report_quality",
    }

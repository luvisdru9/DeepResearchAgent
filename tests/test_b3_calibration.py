from __future__ import annotations

import pytest

from deepresearch.evaluation.annotations import (
    AnnotationEvidence,
    HumanAnnotationBundle,
    HumanAnnotationSet,
    HumanGoldClaimAnnotation,
    HumanReportClaimAnnotation,
)
from deepresearch.evaluation.calibration import (
    build_b3_calibration,
    build_disagreement_adjudication_bundle,
    compare_b3_protocol_results,
)


def _bundle(
    annotator_id: str,
    *,
    report_label: str = "supported",
    gold_label: str = "covered",
    report_claim: str = "证据支持该结论。",
    reason: str = "证据直接支持。",
) -> HumanAnnotationBundle:
    report = HumanReportClaimAnnotation(
        claim_id="claim_1",
        claim=report_claim,
        citation_numbers=[1],
        evidence_ids=["ev_1"],
        evidence=[
            AnnotationEvidence(
                evidence_id="ev_1",
                claim="证据支持该结论。",
                quote="证据支持该结论。",
                source_title="研究来源",
                source_url="https://example.com/source",
            )
        ],
        label=report_label,
        reason=reason,
    )
    gold = HumanGoldClaimAnnotation(
        gold_claim_id="gold_1",
        gold_claim="报告需要覆盖该结论",
        weight=1.0,
        required=True,
        report_claim_ids=["claim_1"],
        label=gold_label,
        reason=reason,
    )
    sample = HumanAnnotationSet(
        benchmark_id="bench",
        benchmark_version="1",
        sample_id="sample_1",
        run_id="run_1",
        annotator_id=annotator_id,
        status="complete",
        report_claims=[report],
        gold_claims=[gold],
    )
    return HumanAnnotationBundle(annotator_id=annotator_id, samples=[sample])


def _automatic(*, report_label: str = "supported", gold_label: str = "covered") -> dict:
    return {
        "judge_protocol": {
            "version": "1.0",
            "fingerprint": "sha256:baseline",
            "repetitions": 2,
        },
        "results": [
            {
                "sample_id": "sample_1",
                "run_id": "run_1",
                "metadata": {
                    "claim_assessments": [
                        {"claim_id": "claim_1", "status": report_label}
                    ],
                    "gold_claim_assessments": [
                        {"gold_claim_id": "gold_1", "status": gold_label}
                    ],
                },
            }
        ]
    }


def test_unanimous_completed_bundles_do_not_require_adjudication():
    payload = build_b3_calibration(
        _automatic(),
        _bundle("annotator_a"),
        _bundle("annotator_b"),
    )

    assert payload["reference_source"] == "unanimous_consensus"
    assert payload["human_agreement"]["report_claims"]["simple_agreement"] == 1.0
    assert payload["identical_reason_count"] == 2
    assert payload["independence_attestation"] == (
        "user_attested_artifacts_content_identical"
    )
    assert payload["gate"]["passed"] is True
    assert "automatic_vs_adjudicated" not in payload
    assert payload["reference_report_substance"]["invalid_count"] == 0


def test_disagreement_without_adjudication_fails():
    with pytest.raises(ValueError, match="adjudication bundle is required"):
        build_b3_calibration(
            _automatic(),
            _bundle("annotator_a"),
            _bundle("annotator_b", gold_label="not_covered"),
        )


def test_disagreement_only_packet_can_supply_third_human_reference():
    annotator_a = _bundle("annotator_a", gold_label="covered")
    annotator_b = _bundle("annotator_b", gold_label="not_covered")

    adjudication = build_disagreement_adjudication_bundle(
        annotator_a,
        annotator_b,
        adjudicator_id="adjudicator_1",
    )

    assert adjudication.role == "adjudicator"
    assert len(adjudication.samples) == 1
    assert adjudication.samples[0].report_claims == []
    assert len(adjudication.samples[0].gold_claims) == 1
    disputed = adjudication.samples[0].gold_claims[0]
    assert disputed.label is None
    assert disputed.reason == ""
    assert disputed.report_claim_ids == []

    # Simulate the third human completing only the disputed item.
    disputed.label = "partially_covered"
    disputed.reason = "报告只表达了目标要点的一部分。"
    disputed.report_claim_ids = ["claim_1"]
    adjudication.samples[0].status = "complete"
    payload = build_b3_calibration(
        _automatic(gold_label="partially_covered"),
        annotator_a,
        annotator_b,
        adjudication=adjudication,
    )

    assert payload["reference_source"] == "human_adjudication"
    assert payload["automatic_vs_reference"]["gold_claims"]["accuracy"] == 1.0
    assert payload["reference_report_substance"]["invalid_count"] == 0


def test_unanimous_annotations_do_not_create_adjudication_packet():
    with pytest.raises(ValueError, match="adjudication is not required"):
        build_disagreement_adjudication_bundle(
            _bundle("annotator_a"),
            _bundle("annotator_b"),
        )


def test_uncertain_reference_reason_is_visible_in_mismatch():
    payload = build_b3_calibration(
        _automatic(gold_label="not_covered"),
        _bundle("annotator_a", reason="这里我不确定。"),
        _bundle("annotator_b", reason="这里我不确定。"),
    )

    assert payload["uncertainty_marker_count"] == 4
    mismatch = payload["mismatches"]["gold_claims"][0]
    assert mismatch["category"] == "human_low_confidence"
    assert mismatch["uncertainty_markers"] == ["不确定"]
    assert payload["gate"]["passed"] is False


def test_truncated_linked_claim_marks_gold_mismatch_as_report_missing():
    payload = build_b3_calibration(
        _automatic(gold_label="not_covered"),
        _bundle(
            "annotator_a",
            report_claim="Research paper abstract...",
            reason="报告已覆盖。",
        ),
        _bundle(
            "annotator_b",
            report_claim="Research paper abstract...",
            reason="报告已覆盖。",
        ),
    )

    assert payload["mismatches"]["gold_claims"][0]["category"] == "report_missing"
    assert payload["reference_report_substance"]["status_counts"] == {
        "truncated_fragment": 1
    }


def test_protocol_comparison_records_label_changes_and_blocks_uncertain_reference():
    baseline = _automatic(gold_label="not_covered")
    candidate = _automatic(gold_label="covered")
    candidate["judge_protocol"] = {
        "version": "2.0",
        "fingerprint": "sha256:candidate",
        "repetitions": 2,
    }

    payload = compare_b3_protocol_results(
        baseline,
        candidate,
        _bundle("annotator_a", reason="这里我不确定。"),
        _bundle("annotator_b", reason="这里我不确定。"),
    )

    assert payload["baseline_protocol"]["version"] == "1.0"
    assert payload["candidate_protocol"]["version"] == "2.0"
    assert payload["change_summary"] == {
        "changed_count": 1,
        "became_reference_match": 1,
        "lost_reference_match": 0,
        "changed_between_non_reference_labels": 0,
    }
    assert payload["metric_deltas"]["gold_claims"]["accuracy"] == 1.0
    assert payload["b4_decision"]["allowed"] is False
    assert "reference_contains_low_confidence_labels" in payload["b4_decision"]["reasons"]


def test_protocol_comparison_requires_unanimous_reference():
    with pytest.raises(ValueError, match="gold-claim labels must be unanimous"):
        compare_b3_protocol_results(
            _automatic(),
            _automatic(),
            _bundle("annotator_a"),
            _bundle("annotator_b", gold_label="not_covered"),
        )


def test_calibration_prefers_explicit_aggregate_labels_over_first_judge():
    automatic = _automatic(gold_label="not_covered")
    automatic["results"][0]["metadata"] = {
        "aggregate_claim_assessments": [
            {"claim_id": "claim_1", "status": "supported"}
        ],
        "aggregate_gold_claim_assessments": [
            {"gold_claim_id": "gold_1", "status": "covered"}
        ],
        "individual_judges": [
            {
                "claim_assessments": [
                    {"claim_id": "claim_1", "status": "unsupported"}
                ],
                "gold_claim_assessments": [
                    {"gold_claim_id": "gold_1", "status": "not_covered"}
                ],
            }
        ],
    }

    payload = build_b3_calibration(
        automatic,
        _bundle("annotator_a"),
        _bundle("annotator_b"),
    )

    assert payload["automatic_vs_reference"]["report_claims"]["accuracy"] == 1.0
    assert payload["automatic_vs_reference"]["gold_claims"]["accuracy"] == 1.0

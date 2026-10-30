import pytest
from pydantic import ValidationError

from deepresearch.evaluation.annotations import (
    HumanAnnotationBundle,
    annotation_agreement,
    build_annotation_set,
    calibrate_labels,
)
from deepresearch.evaluation.benchmark import AtomicClaim
from deepresearch.schema import Evidence, ResearchReport


def _report() -> ResearchReport:
    evidence = Evidence(
        id="ev_1",
        claim="引用质量需要检查证据支持关系。",
        quote="引用是否真正支持对应结论决定引用精度。",
        source_id="source_1",
        source_title="Citation Guide",
        source_url="https://example.com/citation",
    )
    return ResearchReport(
        run_id="annotation_run",
        query="如何评测引用质量？",
        title="Report",
        summary="summary",
        key_findings=["引用精度需要检查证据支持关系。 [1]"],
        evidence=[evidence],
        markdown=(
            "# Report\n\n## 关键发现\n- 引用精度需要检查证据支持关系。 [1]\n\n"
            "## 证据表\n1. 引用是否真正支持对应结论决定引用精度。\n"
        ),
    )


def _completed_bundle(annotator: str, labels: tuple[str, str]) -> HumanAnnotationBundle:
    annotation = build_annotation_set(
        benchmark_id="bench",
        benchmark_version="0.1.0",
        sample_id="sample",
        report=_report(),
        gold_claims=[AtomicClaim(id="gold.c1", text="引用精度")],
        annotator_id=annotator,
    )
    payload = annotation.model_dump()
    payload["report_claims"][0]["label"] = labels[0]
    payload["gold_claims"][0]["label"] = labels[1]
    payload["status"] = "complete"
    completed = annotation.__class__.model_validate(payload)
    return HumanAnnotationBundle(annotator_id=annotator, samples=[completed])


def test_pending_packet_keeps_labels_empty_and_complete_packet_rejects_them():
    annotation = build_annotation_set(
        benchmark_id="bench",
        benchmark_version="0.1.0",
        sample_id="sample",
        report=_report(),
        gold_claims=[AtomicClaim(id="gold.c1", text="引用精度")],
        annotator_id="annotator_a",
    )

    assert annotation.status == "pending"
    assert annotation.report_claims[0].label is None
    with pytest.raises(ValidationError, match="every report claim label"):
        annotation.__class__.model_validate({**annotation.model_dump(), "status": "complete"})


def test_human_agreement_reports_simple_agreement_and_kappa():
    left = _completed_bundle("a", ("supported", "covered"))
    right = _completed_bundle("b", ("unsupported", "covered"))

    report_agreement = annotation_agreement(left, right, target="report_claims")
    gold_agreement = annotation_agreement(left, right, target="gold_claims")

    assert report_agreement.simple_agreement == 0.0
    assert report_agreement.cohen_kappa == 0.0
    assert gold_agreement.simple_agreement == 1.0
    assert gold_agreement.cohen_kappa == 1.0


def test_calibration_reports_per_label_precision_and_recall():
    predicted = {"c1": "supported", "c2": "supported", "c3": "unsupported"}
    reference = {"c1": "supported", "c2": "unsupported", "c3": "unsupported"}

    result = calibrate_labels(predicted, reference)

    assert result.accuracy == pytest.approx(2 / 3)
    assert result.per_label["supported"]["precision"] == pytest.approx(0.5)
    assert result.per_label["supported"]["recall"] == 1.0
    assert result.per_label["unsupported"]["precision"] == 1.0
    assert result.per_label["unsupported"]["recall"] == pytest.approx(0.5)

"""Human annotation packets and calibration metrics for Benchmark B3."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from deepresearch.evaluation.benchmark import AtomicClaim
from deepresearch.evaluation.claims import extract_atomic_claims
from deepresearch.schema import ResearchReport, utc_now

ClaimLabel = Literal[
    "supported",
    "partially_supported",
    "unsupported",
    "contradicted",
    "miscited",
    "uncited",
]
GoldClaimLabel = Literal[
    "covered",
    "partially_covered",
    "not_covered",
    "contradicted",
]


class AnnotationEvidence(BaseModel):
    """Minimal source-bound Evidence shown to human annotators."""

    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    claim: str
    quote: str
    source_title: str
    source_url: str


class HumanReportClaimAnnotation(BaseModel):
    """One annotator's judgment of report-claim evidence grounding."""

    model_config = ConfigDict(extra="forbid")

    claim_id: str
    claim: str
    citation_numbers: list[int] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    evidence: list[AnnotationEvidence] = Field(default_factory=list)
    label: ClaimLabel | None = None
    reason: str = ""


class HumanGoldClaimAnnotation(BaseModel):
    """One annotator's judgment of authored gold-claim coverage."""

    model_config = ConfigDict(extra="forbid")

    gold_claim_id: str
    gold_claim: str
    weight: float = Field(gt=0.0)
    required: bool
    report_claim_ids: list[str] = Field(default_factory=list)
    label: GoldClaimLabel | None = None
    reason: str = ""


class HumanAnnotationSet(BaseModel):
    """Independent labels for one benchmark sample and one generated report."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    benchmark_id: str
    benchmark_version: str
    sample_id: str
    run_id: str
    annotator_id: str
    role: Literal["annotator", "adjudicator"] = "annotator"
    status: Literal["pending", "complete"] = "pending"
    report_claims: list[HumanReportClaimAnnotation]
    gold_claims: list[HumanGoldClaimAnnotation]
    created_at: datetime = Field(default_factory=utc_now)
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_completion(self) -> "HumanAnnotationSet":
        if self.status == "complete":
            if any(item.label is None for item in self.report_claims):
                raise ValueError("Complete annotation sets require every report claim label.")
            if any(item.label is None for item in self.gold_claims):
                raise ValueError("Complete annotation sets require every gold claim label.")
        return self


class HumanAnnotationBundle(BaseModel):
    """All pilot samples assigned to one independent annotator."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    annotator_id: str
    role: Literal["annotator", "adjudicator"] = "annotator"
    instructions_version: Literal["b3.1"] = "b3.1"
    samples: list[HumanAnnotationSet]


class AgreementMetrics(BaseModel):
    """Pairwise agreement over one aligned label collection."""

    compared_count: int
    simple_agreement: float | None
    cohen_kappa: float | None
    confusion: dict[str, int] = Field(default_factory=dict)


class LabelCalibrationMetrics(BaseModel):
    """Automatic-label performance against adjudicated human labels."""

    compared_count: int
    accuracy: float | None
    macro_precision: float | None
    macro_recall: float | None
    per_label: dict[str, dict[str, float | int]] = Field(default_factory=dict)
    confusion: dict[str, int] = Field(default_factory=dict)


def build_annotation_set(
    *,
    benchmark_id: str,
    benchmark_version: str,
    sample_id: str,
    report: ResearchReport,
    gold_claims: list[AtomicClaim],
    annotator_id: str,
    role: Literal["annotator", "adjudicator"] = "annotator",
) -> HumanAnnotationSet:
    """Create a pending, evidence-linked packet without inventing human labels."""

    report_claims = extract_atomic_claims(report, require_citations=True)
    evidence_by_id = {item.id: item for item in report.evidence}
    return HumanAnnotationSet(
        benchmark_id=benchmark_id,
        benchmark_version=benchmark_version,
        sample_id=sample_id,
        run_id=report.run_id,
        annotator_id=annotator_id,
        role=role,
        report_claims=[
            HumanReportClaimAnnotation(
                claim_id=item.id,
                claim=item.text,
                citation_numbers=item.citation_numbers,
                evidence_ids=item.evidence_ids,
                evidence=[
                    AnnotationEvidence(
                        evidence_id=evidence_id,
                        claim=evidence_by_id[evidence_id].claim,
                        quote=evidence_by_id[evidence_id].quote,
                        source_title=evidence_by_id[evidence_id].source_title,
                        source_url=evidence_by_id[evidence_id].source_url,
                    )
                    for evidence_id in item.evidence_ids
                ],
            )
            for item in report_claims
        ],
        gold_claims=[
            HumanGoldClaimAnnotation(
                gold_claim_id=item.id,
                gold_claim=item.text,
                weight=item.weight,
                required=item.required,
            )
            for item in gold_claims
        ],
        notes=[
            "Fill labels independently; do not inspect another annotator's file.",
            "Use only the report and supplied Evidence when judging report-claim grounding.",
        ],
    )


def annotation_agreement(
    left: HumanAnnotationBundle,
    right: HumanAnnotationBundle,
    *,
    target: Literal["report_claims", "gold_claims"],
) -> AgreementMetrics:
    """Compare two completed independent bundles using stable sample/run/claim IDs."""

    left_labels = _bundle_labels(left, target=target)
    right_labels = _bundle_labels(right, target=target)
    if set(left_labels) != set(right_labels):
        missing_left = sorted(set(right_labels) - set(left_labels))
        missing_right = sorted(set(left_labels) - set(right_labels))
        raise ValueError(
            f"Annotation keys differ; missing_left={missing_left}, missing_right={missing_right}"
        )
    keys = sorted(left_labels)
    if not keys:
        return AgreementMetrics(compared_count=0, simple_agreement=None, cohen_kappa=None)
    left_values = [left_labels[key] for key in keys]
    right_values = [right_labels[key] for key in keys]
    agreement = sum(
        left_value == right_value
        for left_value, right_value in zip(left_values, right_values, strict=True)
    ) / len(keys)
    return AgreementMetrics(
        compared_count=len(keys),
        simple_agreement=agreement,
        cohen_kappa=_cohen_kappa(left_values, right_values),
        confusion=_confusion(left_values, right_values),
    )


def calibrate_labels(
    predicted: dict[str, str],
    reference: dict[str, str],
) -> LabelCalibrationMetrics:
    """Calculate per-label precision/recall and macro scores on aligned decisions."""

    if set(predicted) != set(reference):
        raise ValueError("Predicted and reference label keys must match exactly.")
    keys = sorted(reference)
    if not keys:
        return LabelCalibrationMetrics(
            compared_count=0,
            accuracy=None,
            macro_precision=None,
            macro_recall=None,
        )
    predicted_values = [predicted[key] for key in keys]
    reference_values = [reference[key] for key in keys]
    labels = sorted(set(predicted_values).union(reference_values))
    per_label: dict[str, dict[str, float | int]] = {}
    precisions: list[float] = []
    recalls: list[float] = []
    for label in labels:
        true_positive = sum(
            predicted_value == label and reference_value == label
            for predicted_value, reference_value in zip(
                predicted_values, reference_values, strict=True
            )
        )
        false_positive = sum(
            predicted_value == label and reference_value != label
            for predicted_value, reference_value in zip(
                predicted_values, reference_values, strict=True
            )
        )
        false_negative = sum(
            predicted_value != label and reference_value == label
            for predicted_value, reference_value in zip(
                predicted_values, reference_values, strict=True
            )
        )
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        precisions.append(precision)
        recalls.append(recall)
        per_label[label] = {
            "precision": precision,
            "recall": recall,
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
        }
    return LabelCalibrationMetrics(
        compared_count=len(keys),
        accuracy=sum(
            predicted_value == reference_value
            for predicted_value, reference_value in zip(
                predicted_values, reference_values, strict=True
            )
        )
        / len(keys),
        macro_precision=mean(precisions),
        macro_recall=mean(recalls),
        per_label=per_label,
        confusion=_confusion(reference_values, predicted_values),
    )


def bundle_labels(
    bundle: HumanAnnotationBundle,
    *,
    target: Literal["report_claims", "gold_claims"],
) -> dict[str, str]:
    """Expose complete labels with the canonical sample/run/claim key."""

    return _bundle_labels(bundle, target=target)


def write_annotation_bundle(bundle: HumanAnnotationBundle, destination: str | Path) -> Path:
    """Persist one independent packet; callers choose a non-shared annotator path."""

    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(bundle.model_dump_json(indent=2), encoding="utf-8")
    return path


def _bundle_labels(
    bundle: HumanAnnotationBundle,
    *,
    target: Literal["report_claims", "gold_claims"],
) -> dict[str, str]:
    labels: dict[str, str] = {}
    for sample in bundle.samples:
        if sample.status != "complete":
            raise ValueError(f"Annotation sample '{sample.sample_id}' is not complete.")
        items = sample.report_claims if target == "report_claims" else sample.gold_claims
        for item in items:
            identifier = item.claim_id if target == "report_claims" else item.gold_claim_id
            if item.label is None:
                raise ValueError(f"Missing label for '{identifier}'.")
            key = f"{sample.sample_id}:{sample.run_id}:{identifier}"
            if key in labels:
                raise ValueError(f"Duplicate annotation key: {key}")
            labels[key] = item.label
    return labels


def _cohen_kappa(left: list[str], right: list[str]) -> float:
    observed = sum(
        left_value == right_value
        for left_value, right_value in zip(left, right, strict=True)
    ) / len(left)
    labels = set(left).union(right)
    expected = sum(
        (left.count(label) / len(left)) * (right.count(label) / len(right))
        for label in labels
    )
    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1.0 - expected)


def _confusion(reference: list[str], predicted: list[str]) -> dict[str, int]:
    counts = Counter(
        f"reference={reference_value}|predicted={predicted_value}"
        for reference_value, predicted_value in zip(reference, predicted, strict=True)
    )
    return dict(sorted(counts.items()))

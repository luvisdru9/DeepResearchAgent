"""Evaluation entrypoints for ResearchBench-style smoke tests."""

from deepresearch.evaluation.runner import EvaluationRunner
from deepresearch.evaluation.service import summarize_automatic_consensus
from deepresearch.evaluation.process_metrics import score_research_process
from deepresearch.evaluation.annotations import (
    AgreementMetrics,
    HumanAnnotationBundle,
    HumanAnnotationSet,
    LabelCalibrationMetrics,
    annotation_agreement,
    bundle_labels,
    build_annotation_set,
    calibrate_labels,
)
from deepresearch.evaluation.calibration import (
    build_disagreement_adjudication_bundle,
    compare_b3_protocol_results,
)
from deepresearch.judge_protocol import (
    JUDGE_PROTOCOL_VERSION,
    JudgeProtocolManifest,
    judge_protocol_fingerprint,
    judge_system_prompt,
)
from deepresearch.evaluation.claims import (
    AtomicReportClaim,
    ClaimCitationAssessment,
    GoldClaimCoverage,
    assess_claim_citations,
    assess_gold_claim_coverage,
    extract_atomic_claims,
    weighted_gold_claim_recall,
)
from deepresearch.evaluation.benchmark import (
    AtomicClaim,
    AtomicClaimSatisfactionRoute,
    BenchmarkDatasetV2,
    BenchmarkLoadResult,
    BenchmarkSample,
    BenchmarkSampleV2,
    benchmark_v2_json_schema,
    export_benchmark_v2_json_schema,
    load_benchmark,
    load_benchmark_bundle,
    resolve_snapshot_manifest_path,
)
from deepresearch.evaluation.compression_comparison import (
    CompressionComparisonRunner,
    CompressionVariantResult,
)
from deepresearch.evaluation.retrieval_comparison import (
    RetrievalComparisonRunner,
    RetrievalVariantResult,
)
from deepresearch.evaluation.system_comparison import (
    SYSTEM_STRATEGIES,
    SystemComparisonRunner,
    SystemVariantResult,
)
from deepresearch.evaluation.auto_draft import (
    TRACK_A_DRAFT_SAMPLE_IDS,
    AutoDraftBuildResult,
    AutoDraftBuilder,
)
from deepresearch.evaluation.researchbench40 import (
    RESEARCHBENCH40_TASK_TYPES,
    ResearchBench40Builder,
    ResearchBench40BuildResult,
    ResearchBench40Manifest,
    ResearchBench40SampleState,
    export_researchbench40_schema,
)
from deepresearch.evaluation.example_export import export_comparison_examples, export_run_examples
from deepresearch.evaluation.frozen_snapshot import (
    FrozenDocumentRecord,
    FrozenQueryRecord,
    FrozenSnapshotManifest,
    load_frozen_snapshot,
    write_frozen_snapshot,
)

__all__ = [
    "AgreementMetrics",
    "AutoDraftBuildResult",
    "AutoDraftBuilder",
    "CompressionComparisonRunner",
    "CompressionVariantResult",
    "EvaluationRunner",
    "JUDGE_PROTOCOL_VERSION",
    "JudgeProtocolManifest",
    "HumanAnnotationBundle",
    "HumanAnnotationSet",
    "LabelCalibrationMetrics",
    "FrozenDocumentRecord",
    "FrozenQueryRecord",
    "FrozenSnapshotManifest",
    "AtomicClaim",
    "AtomicClaimSatisfactionRoute",
    "AtomicReportClaim",
    "ClaimCitationAssessment",
    "GoldClaimCoverage",
    "BenchmarkDatasetV2",
    "BenchmarkLoadResult",
    "BenchmarkSample",
    "BenchmarkSampleV2",
    "benchmark_v2_json_schema",
    "assess_claim_citations",
    "assess_gold_claim_coverage",
    "annotation_agreement",
    "bundle_labels",
    "build_annotation_set",
    "build_disagreement_adjudication_bundle",
    "calibrate_labels",
    "compare_b3_protocol_results",
    "export_benchmark_v2_json_schema",
    "extract_atomic_claims",
    "weighted_gold_claim_recall",
    "load_benchmark",
    "load_benchmark_bundle",
    "resolve_snapshot_manifest_path",
    "score_research_process",
    "summarize_automatic_consensus",
    "judge_protocol_fingerprint",
    "judge_system_prompt",
    "load_frozen_snapshot",
    "RetrievalComparisonRunner",
    "RetrievalVariantResult",
    "RESEARCHBENCH40_TASK_TYPES",
    "ResearchBench40Builder",
    "ResearchBench40BuildResult",
    "ResearchBench40Manifest",
    "ResearchBench40SampleState",
    "SYSTEM_STRATEGIES",
    "TRACK_A_DRAFT_SAMPLE_IDS",
    "SystemComparisonRunner",
    "SystemVariantResult",
    "export_comparison_examples",
    "export_researchbench40_schema",
    "export_run_examples",
    "write_frozen_snapshot",
]

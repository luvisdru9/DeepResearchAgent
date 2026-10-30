"""Frozen protocol metadata for reproducible LLM-as-Judge experiments."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import BaseModel, Field

JUDGE_PROTOCOL_VERSION = "5.0"

# 修改该提示词必须同步升级 JUDGE_PROTOCOL_VERSION；指纹会进入每个评测产物。
JUDGE_SYSTEM_PROMPT_V1 = (
    "You are an evidence-grounded evaluator for a deep research system. "
    "Judge only against the supplied report, evidence, question, and rubric. "
    "Do not infer which system produced the report. Do not reward style as factuality and do not "
    "assume a citation supports a claim merely because it exists. Return one JSON object with "
    "integer scores from 1 to 5 for factuality, completeness, citation_quality, logical_structure, "
    "and writing_quality; include overall_reason, claim_assessments, and gold_claim_assessments. "
    "Assess every supplied report claim. Each claim assessment must contain claim_id, claim, "
    "status=supported|partially_supported|unsupported|contradicted|miscited|uncited, evidence_ids, "
    "and reason. Use only Evidence IDs actually mapped by that report claim. Assess every supplied "
    "gold claim. Each gold assessment must contain gold_claim_id, "
    "status=covered|partially_covered|not_covered|contradicted, report_claim_ids, and reason. "
    "Use stable IDs from the input and do not create new IDs."
)

# v2 只收紧 authored gold-claim coverage 的语义。v1 常量不可修改，
# 否则历史实验中保存的协议指纹将无法复现。
JUDGE_SYSTEM_PROMPT_V2 = JUDGE_SYSTEM_PROMPT_V1 + (
    " For gold-claim coverage, judge only what the report claims explicitly state. "
    "A paper title, author list, citation, topic mention, method name, abstract header, or "
    "bibliographic identity does not by itself cover a research conclusion or metric definition. "
    "Use covered only when the linked report claims express every material semantic component of "
    "the gold claim. Use partially_covered only when at least one material component is explicitly "
    "stated. Do not infer that the report covers information merely because a cited source might "
    "contain it. Every non-not_covered decision must list the report_claim_ids that provide the "
    "actual wording."
)

# v3 fixes two ambiguity classes exposed by the Flash/Pro B3 audit while
# retaining every historical prompt and fingerprint unchanged.
JUDGE_SYSTEM_PROMPT_V3 = JUDGE_SYSTEM_PROMPT_V2 + (
    " For report-claim evidence support, every independently checkable factual clause must be "
    "explicitly supported by the Evidence IDs mapped to that report claim. A statement that the "
    "source or available evidence lacks experiments, numbers, validation, or other material is not "
    "supported merely because the mapped quote omits it. If a compound claim contains a supported "
    "fact plus such an unsupported absence or source-characterization clause, use "
    "partially_supported. A clearly marked proposal or recommendation synthesized from several "
    "mapped Evidence items may be supported only when every factual premise is explicit and the "
    "report does not misattribute the combined proposal to one source. For gold coverage, topical "
    "similarity, an outcome statistic, or a discussion of complete support does not cover a requested "
    "rate definition unless the report explicitly states the required analysis unit or proportion. "
    "When a gold claim has multiple material components, explicit wording of one component, such as "
    "atomic-claim analysis without the requested error taxonomy, is partially_covered rather than "
    "not_covered."
)

# v4 clarifies that benchmark authors may provide complete alternative satisfaction
# routes. v1-v3 remain immutable so their stored experiment fingerprints stay valid.
JUDGE_SYSTEM_PROMPT_V4 = JUDGE_SYSTEM_PROMPT_V3 + (
    " Treat accepted_variants as benchmark-authored alternative satisfaction routes, not merely "
    "as search keywords. Mark a gold claim covered when the report explicitly expresses either "
    "all material semantic components of the main gold text or the complete semantic requirement "
    "of at least one accepted variant. Mark it partially_covered only when the report states part "
    "of the main route or part of an accepted-variant route. A topic mention or isolated shared "
    "keyword is never sufficient. In the reason, identify which main or accepted-variant route was "
    "used, and list only the report_claim_ids whose wording establishes that route."
)

# v5 separates wording aliases from complete, benchmark-authored satisfaction routes.
# The previous prompts stay byte-for-byte unchanged to preserve historical fingerprints.
JUDGE_SYSTEM_PROMPT_V5 = JUDGE_SYSTEM_PROMPT_V3 + (
    " Treat accepted_variants only as wording aliases useful for retrieval or annotation; they "
    "are not independently sufficient coverage conditions. Use only the supplied "
    "satisfaction_routes as complete routes for satisfying a gold claim. Mark a gold claim "
    "covered only when the linked report claims explicitly establish every material semantic "
    "component of one route. A covered assessment must return that route's exact "
    "matched_route_id and the report_claim_ids that establish it. For partially_covered, "
    "not_covered, or contradicted, matched_route_id must be null. Keep overall_reason and every "
    "item reason concise: do not restate the report, Evidence, or these scoring rules."
)

_JUDGE_PROMPTS = {
    "1.0": JUDGE_SYSTEM_PROMPT_V1,
    "2.0": JUDGE_SYSTEM_PROMPT_V2,
    "3.0": JUDGE_SYSTEM_PROMPT_V3,
    "4.0": JUDGE_SYSTEM_PROMPT_V4,
    "5.0": JUDGE_SYSTEM_PROMPT_V5,
}


class JudgeProtocolManifest(BaseModel):
    """Configuration needed to reproduce one family of Judge calls."""

    version: str
    fingerprint: str
    judge_profiles: list[str | None]
    judge_models: list[str | None] = Field(default_factory=list)
    independent_model_versions: bool = False
    repetitions: int
    shuffle_claims: bool
    random_seed: int
    raw_trace_enabled: bool
    # 重复 Judge 的标签先多数表决；平票时选择更保守的支持/覆盖状态。
    label_aggregation_policy: str = "majority_with_conservative_tie"
    min_successful_repetitions_per_profile: int = 1
    automatic_gate_thresholds: dict[str, float] = Field(default_factory=dict)


def judge_protocol_fingerprint(version: str = JUDGE_PROTOCOL_VERSION) -> str:
    """Hash prompt and output-label semantics, not provider credentials."""

    if version not in _JUDGE_PROMPTS:
        raise ValueError(f"Unsupported Judge protocol version: {version}")
    payload: dict[str, Any] = {
        "version": version,
        "system_prompt": _JUDGE_PROMPTS[version],
        "claim_statuses": [
            "supported",
            "partially_supported",
            "unsupported",
            "contradicted",
            "miscited",
            "uncited",
        ],
        "gold_claim_statuses": [
            "covered",
            "partially_covered",
            "not_covered",
            "contradicted",
        ],
        "score_range": [1, 5],
    }
    if version == "5.0":
        payload["gold_route_semantics"] = (
            "explicit_satisfaction_routes_with_matched_route_id"
        )
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"sha256:{hashlib.sha256(canonical).hexdigest()}"


def judge_system_prompt(version: str = JUDGE_PROTOCOL_VERSION) -> str:
    """Return the immutable prompt associated with one supported protocol version."""

    try:
        return _JUDGE_PROMPTS[version]
    except KeyError as exc:
        raise ValueError(f"Unsupported Judge protocol version: {version}") from exc

"""
RedTeamAgent
是质量检查雏形。当前主要检查证据数量是否不足、趋势性判断是否需要时间范围说明。后续可扩展为真正的对抗式事实核查。
"""

from __future__ import annotations

import asyncio
from typing import Literal

from pydantic import BaseModel, Field

from deepresearch.llm import ChatMessage, LLMClient
from deepresearch.quality import validate_report_citations
from deepresearch.schema import (
    Evidence,
    EvidenceCoverageMatrix,
    RedTeamFinding,
    RedTeamTrace,
    ResearchReport,
    ResearchTask,
)


class RedTeamAgent:
    """Rule-based quality reviewer for the MVP report draft."""

    def review(
        self,
        key_findings: list[str],
        evidence: list[Evidence],
        *,
        task: ResearchTask | None = None,
        report: ResearchReport | None = None,
        mode: Literal["attack", "verify"] = "attack",
        prior_findings: list[RedTeamFinding] | None = None,
        coverage_matrix: EvidenceCoverageMatrix | None = None,
    ) -> list[RedTeamFinding]:
        """Return potential quality issues that should appear in the final replay/report."""

        findings: list[RedTeamFinding] = []
        # 证据数量少于结论数量时，引用覆盖不足的风险会直接进入 RedTeam findings。
        if len(evidence) < max(2, len(key_findings)):
            findings.append(
                RedTeamFinding(
                    issue="部分结论的证据数量不足。",
                    severity="medium",
                    suggested_fix="为每个关键结论补充至少一条可追溯引用。",
                )
            )
        # 趋势性表述通常需要时间范围；离线语料无法保证时效性，因此单独提示。
        if any("可能" in finding or "趋势" in finding for finding in key_findings):
            findings.append(
                RedTeamFinding(
                    issue="趋势性判断需要明确时间范围和来源覆盖。",
                    severity="low",
                    suggested_fix="在报告局限性中说明当前离线语料和时间范围限制。",
                )
            )
        return findings


class RedTeamReviewItem(BaseModel):
    """One LLM-raised review issue, grounded in existing report claims and evidence IDs."""

    claim: str = ""
    status: Literal["supported", "weakly_supported", "unsupported", "contradicted", "unclear"] = "unclear"
    severity: Literal["low", "medium", "high"] = "medium"
    issue: str = Field(min_length=1)
    evidence_ids: list[str] = Field(default_factory=list)
    reason: str = ""
    suggested_fix: str = Field(min_length=1)
    category: Literal[
        "factuality",
        "citation_integrity",
        "citation_support",
        "logic",
        "coverage",
        "time_scope",
        "other",
    ] = "other"
    repair_action: Literal["retrieve", "narrow", "remove", "qualify", "none"] = (
        "retrieve"
    )


class RedTeamOutput(BaseModel):
    """Structured output required from the LLM RedTeam / verifier pass."""

    overall_status: Literal["pass", "needs_revision", "fail"] = "needs_revision"
    summary: str = ""
    findings: list[RedTeamReviewItem] = Field(default_factory=list)
    missing_perspectives: list[str] = Field(default_factory=list)


class LLMRedTeamAgent:
    """Use an LLM as a RedTeam reviewer with verifier-style evidence checks.

    第一版刻意复用 RedTeamAgent 的公共输出 `RedTeamFinding`，避免在 pipeline 中并行引入
    RedTeam 与 Verifier 两套审查模型。后续如果需要更细的 Verifier，可从这里拆出独立 Agent。
    """

    def __init__(
        self,
        llm: LLMClient,
        *,
        fallback: RedTeamAgent | None = None,
        fallback_to_rule: bool = True,
        max_evidence_items: int = 12,
    ) -> None:
        self.llm = llm
        self.fallback = fallback or RedTeamAgent()
        self.fallback_to_rule = fallback_to_rule
        self.max_evidence_items = max(1, max_evidence_items)
        self.last_trace = RedTeamTrace(requested_provider="llm", used_provider="llm")

    def review(
        self,
        key_findings: list[str],
        evidence: list[Evidence],
        *,
        task: ResearchTask | None = None,
        report: ResearchReport | None = None,
        mode: Literal["attack", "verify"] = "attack",
        prior_findings: list[RedTeamFinding] | None = None,
        coverage_matrix: EvidenceCoverageMatrix | None = None,
    ) -> list[RedTeamFinding]:
        """Synchronous wrapper for callers outside the async Pipeline."""

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.review_async(
                    key_findings,
                    evidence,
                    task=task,
                    report=report,
                    mode=mode,
                    prior_findings=prior_findings,
                    coverage_matrix=coverage_matrix,
                )
            )
        raise RuntimeError("LLMRedTeamAgent.review() cannot run inside an event loop; await review_async().")

    async def review_async(
        self,
        key_findings: list[str],
        evidence: list[Evidence],
        *,
        task: ResearchTask | None = None,
        report: ResearchReport | None = None,
        mode: Literal["attack", "verify"] = "attack",
        prior_findings: list[RedTeamFinding] | None = None,
        coverage_matrix: EvidenceCoverageMatrix | None = None,
    ) -> list[RedTeamFinding]:
        """Attack a draft or verify a revised report with the same grounded reviewer."""

        try:
            output = await self.llm.json_chat(
                self._build_messages(
                    key_findings,
                    evidence,
                    task=task,
                    report=report,
                    mode=mode,
                    prior_findings=prior_findings or [],
                    coverage_matrix=coverage_matrix,
                ),
                RedTeamOutput,
                agent="redteam" if mode == "attack" else "verifier",
            )
            findings = self._build_findings(output, evidence)
            self.last_trace = RedTeamTrace(
                requested_provider="llm",
                used_provider="llm",
                summary=output.summary,
                overall_status=output.overall_status,
            )
            return findings
        except Exception as exc:  # noqa: BLE001 - provider, parsing, and grounding errors share fallback.
            if not self.fallback_to_rule:
                self.last_trace = RedTeamTrace(
                    requested_provider="llm",
                    used_provider="llm",
                    error=f"{type(exc).__name__}: {exc}",
                )
                raise
            findings = self.fallback.review(
                key_findings,
                evidence,
                task=task,
                report=report,
            )
            self.last_trace = RedTeamTrace(
                requested_provider="llm",
                used_provider="rule",
                fallback_used=True,
                error=f"{type(exc).__name__}: {exc}",
            )
            return findings

    def _build_messages(
        self,
        key_findings: list[str],
        evidence: list[Evidence],
        *,
        task: ResearchTask | None,
        report: ResearchReport | None,
        mode: Literal["attack", "verify"],
        prior_findings: list[RedTeamFinding],
        coverage_matrix: EvidenceCoverageMatrix | None = None,
    ) -> list[ChatMessage]:
        """Build a compact prompt for adversarial report review and evidence verification."""

        query = task.query if task else (report.query if report else "unspecified")
        evidence_text = "\n".join(
            (
                f"- id={item.id}; claim={item.claim}; quote={item.quote}; "
                f"source={item.source_title}; method={item.extraction_method}; "
                f"reader_extraction_confidence={item.confidence:.2f}; "
                f"source_authority_heuristic_tier={item.source_credibility_tier or 'unknown'}"
            )
            for item in evidence[: self.max_evidence_items]
        ) or "- no evidence available"
        findings_text = "\n".join(
            f"{index}. {finding}" for index, finding in enumerate(key_findings, start=1)
        ) or "No key findings."
        prior_findings_text = "\n".join(
            (
                f"- status={item.status or 'unclear'}; claim={item.claim or 'unspecified'}; "
                f"issue={item.issue}; expected_fix={item.suggested_fix}"
            )
            for item in prior_findings
        ) or "- no prior findings"
        report_text = report.markdown if report else "No full report draft supplied."
        coverage_text = self._format_coverage_matrix(coverage_matrix)
        citation_validation = (
            validate_report_citations(report, require_citations=bool(report.evidence))
            if report
            else None
        )
        citation_map_text = (
            ", ".join(
                f"[{number}]={evidence_id}"
                for number, evidence_id in citation_validation.citation_map.items()
            )
            if citation_validation
            else "unavailable"
        )
        citation_integrity_text = (
            f"valid={citation_validation.valid}; map={citation_map_text}; "
            f"issues={citation_validation.issues or 'none'}"
            if citation_validation
            else "unavailable"
        )
        role_instruction = (
            "Attack the draft adversarially and identify actionable quality problems."
            if mode == "attack"
            else (
                "Verify the revised report. Check whether prior findings were repaired and return only "
                "remaining or newly introduced problems. Use overall_status=pass when none remain."
            )
        )
        system = (
            "You are the shared RedTeam and final verification component of a deep research system. "
            f"Current mode: {mode}. {role_instruction} "
            "Use only the supplied evidence table. "
            "Return exactly one valid JSON object and nothing else. "
            "The JSON must match this shape: "
            '{"overall_status":"pass|needs_revision|fail","summary":"...",'
            '"findings":[{"claim":"...","status":"supported|weakly_supported|unsupported|contradicted|unclear",'
            '"severity":"low|medium|high","issue":"...","evidence_ids":["ev_id"],'
            '"reason":"...","suggested_fix":"...",'
            '"category":"factuality|citation_integrity|citation_support|logic|coverage|time_scope|other",'
            '"repair_action":"retrieve|narrow|remove|qualify|none"}],'
            '"missing_perspectives":["..."]}. '
            "Focus on unsupported claims, weak or mismatched citations, overclaims, missing counterpoints, "
            "logical gaps, and time-scope limitations. Do not invent evidence IDs. "
            "reader_extraction_confidence measures extraction fidelity, not source authority. "
            "source_authority_heuristic_tier is the separate rule-based source estimate; never call "
            "a source low-confidence solely because its authority tier is low. "
            "Citation numbering and evidence-table structure are checked deterministically by the application. "
            "Use category=citation_integrity only when that supplied check is invalid; use citation_support "
            "when an existing citation does not semantically support its claim. "
            "For every actionable finding, classify its repair_action. Use retrieve when new evidence is "
            "required; use narrow/remove/qualify only when the report can be made correct solely by deleting "
            "unsupported detail, narrowing to supplied evidence, or clearly qualifying an existing cited "
            "claim. Use none only for supported/non-actionable observations. "
            "Use the deterministic coverage matrix to distinguish a missing subject from a subject "
            "with only some missing dimensions. Do not report the whole subject as absent when any "
            "matrix cell lists Evidence IDs. Put non-blocking optional perspectives in "
            "missing_perspectives, not as high/medium findings. "
            "Before returning pass, check every explicit quantitative obligation in the research query. "
            "If it asks for at least N empirical studies, the report must identify N distinct studies, state "
            "a directly supported result or conclusion for each, and compare them. Author, title, method, or "
            "bibliographic identity without a study result does not count; report the deficit as a blocking "
            "coverage finding with repair_action=retrieve. "
            "If a claim is well supported, you may omit it from findings."
        )
        user = (
            f"Research query: {query}\n\n"
            f"Key findings:\n{findings_text}\n\n"
            f"Evidence table:\n{evidence_text}\n\n"
            f"Deterministic citation integrity:\n{citation_integrity_text}\n\n"
            f"Prior RedTeam findings:\n{prior_findings_text}\n\n"
            f"Evidence coverage matrix:\n{coverage_text}\n\n"
            f"Draft report markdown:\n{report_text}"
        )
        return [
            ChatMessage(role="system", content=system),
            ChatMessage(role="user", content=user),
        ]

    @staticmethod
    def _format_coverage_matrix(matrix: EvidenceCoverageMatrix | None) -> str:
        """Render the same compact coverage view used by the Synthesizer."""

        if matrix is None or not matrix.subjects:
            return "- unavailable"
        return "\n".join(
            (
                f"- subject={row.subject}; "
                + "; ".join(
                    f"{dimension}={','.join(ids) if ids else 'MISSING'}"
                    for dimension, ids in row.dimension_evidence_ids.items()
                )
            )
            for row in matrix.subjects
        )

    @staticmethod
    def _build_findings(
        output: RedTeamOutput,
        evidence: list[Evidence],
    ) -> list[RedTeamFinding]:
        """Validate model evidence references and map review items to public findings."""

        known_ids = {item.id for item in evidence}
        findings: list[RedTeamFinding] = []
        for item in output.findings:
            unknown_ids = [evidence_id for evidence_id in item.evidence_ids if evidence_id not in known_ids]
            if unknown_ids:
                raise ValueError(
                    "RedTeam referenced unknown evidence IDs: "
                    f"{', '.join(sorted(unknown_ids))}."
                )
            findings.append(
                RedTeamFinding(
                    issue=item.issue.strip(),
                    severity=item.severity,
                    suggested_fix=item.suggested_fix.strip(),
                    claim=item.claim.strip() or None,
                    status=item.status,
                    evidence_ids=list(dict.fromkeys(item.evidence_ids)),
                    reason=item.reason.strip() or None,
                    category=item.category,
                    repair_action=item.repair_action,
                )
            )

        for perspective in output.missing_perspectives:
            cleaned = perspective.strip()
            if not cleaned:
                continue
            findings.append(
                RedTeamFinding(
                    issue=f"可能遗漏视角：{cleaned}",
                    severity="low",
                    suggested_fix="补充检索或在局限性中说明该视角尚未覆盖。",
                    status="unclear",
                    reason=output.summary or None,
                    category="coverage",
                )
            )
        return findings

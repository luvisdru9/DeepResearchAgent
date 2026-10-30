"""Rule and LLM synthesizers that build evidence-grounded reports."""

from __future__ import annotations

import asyncio
from collections import OrderedDict

from pydantic import BaseModel, Field

from deepresearch.llm import ChatMessage, LLMClient
from deepresearch.quality import (
    FindingSubstanceDiagnostic,
    MethodologyObligationDiagnostic,
    SourceAbsenceDiagnostic,
    assess_methodology_obligations,
    assess_report_findings,
    assess_source_absence_claims,
)
from deepresearch.schema import (
    Evidence,
    EvidenceCoverageMatrix,
    RedTeamFinding,
    ResearchReport,
    ResearchTask,
    SynthesizerTrace,
)


class SynthesizedFinding(BaseModel):
    """One model-generated finding grounded in existing evidence IDs."""

    text: str = Field(min_length=1)
    evidence_ids: list[str] = Field(default_factory=list)


class SynthesizerOutput(BaseModel):
    """Restricted structured report content returned by the model."""

    title: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    key_findings: list[SynthesizedFinding] = Field(min_length=1)
    limitations: list[str] = Field(default_factory=list)


class SynthesizerAgent:
    """Builds a cited Markdown report from extracted evidence."""

    def synthesize(
        self,
        task: ResearchTask,
        evidence: list[Evidence],
        redteam_findings: list[RedTeamFinding] | None = None,
        *,
        coverage_matrix: EvidenceCoverageMatrix | None = None,
    ) -> ResearchReport:
        """Generate the final report object.

        这里遵循“证据先行”：先从 Evidence 推导 key_findings，再把 evidence table 写入报告。
        """

        redteam_findings = redteam_findings or []
        unique_evidence = self._dedupe_evidence(evidence)
        candidate_evidence = unique_evidence[:5]
        finding_pairs: list[tuple[str, Evidence]] = []
        replaced_truncated_claims = 0
        dropped_evidence = 0
        for item in candidate_evidence:
            claim_diagnostic = assess_report_findings(
                task.query,
                [item.claim],
                [item],
            )[0]
            if claim_diagnostic.status == "substantive":
                finding_pairs.append((item.claim, item))
                continue
            quote_diagnostic = assess_report_findings(
                task.query,
                [item.quote],
                [item],
            )[0]
            if quote_diagnostic.status == "substantive":
                finding_pairs.append((item.quote, item))
                replaced_truncated_claims += 1
            else:
                dropped_evidence += 1
        valid_evidence = [item for _, item in finding_pairs]
        key_findings = [
            f"{finding} [{index}]"
            for index, (finding, _) in enumerate(finding_pairs, start=1)
        ]
        if not key_findings:
            key_findings = [
                "The retrieved material does not contain enough evidence; further review is required."
                if task.language.casefold().startswith("en")
                else "当前语料未检索到足够证据，结论需要人工补充。"
            ]

        if task.language.casefold().startswith("en"):
            limitations = [
                "The report is limited to retrieved material and makes no factual claim about uncovered issues.",
                "Citations locate supporting material; important findings should still be checked against original sources.",
            ]
        else:
            limitations = [
                "报告仅基于本次检索到的资料，未覆盖的内容不作事实结论。",
                "引用用于定位支持材料，关键结论仍建议结合原始来源复核。",
            ]
        if replaced_truncated_claims:
            limitations.append(
                "部分抽取摘要存在截断，报告改用同一证据的完整原文片段形成结论。"
            )
        if dropped_evidence:
            limitations.append(
                "部分检索内容仅包含书目信息、标题或截断片段，未作为研究结论使用。"
            )
        limitations.extend(
            self._source_credibility_limitations(valid_evidence, language=task.language)
        )

        markdown = self._to_markdown(
            task,
            key_findings,
            valid_evidence,
            redteam_findings,
            limitations,
        )
        return ResearchReport(
            run_id=task.run_id,
            query=task.query,
            title=f"Deep Research Report: {task.query[:40]}",
            summary=(
                "This report organizes findings from traceable retrieved material and states limitations where evidence is insufficient."
                if task.language.casefold().startswith("en")
                else "本报告根据本次检索到的可追溯材料组织结论，并在证据不足处保留限制说明。"
            ),
            key_findings=key_findings,
            evidence=valid_evidence,
            limitations=limitations,
            redteam_findings=redteam_findings,
            markdown=markdown,
        )

    @staticmethod
    def _dedupe_evidence(evidence: list[Evidence]) -> list[Evidence]:
        """Keep the first evidence item for each source+quote pair."""

        deduped: OrderedDict[str, Evidence] = OrderedDict()
        for item in evidence:
            # 同一文档中高度相似的 quote 只保留一次，减少报告证据表重复。
            key = f"{item.source_id}:{item.quote[:80]}"
            deduped.setdefault(key, item)
        return list(deduped.values())

    @staticmethod
    def _source_credibility_limitations(
        evidence: list[Evidence],
        *,
        language: str = "zh",
    ) -> list[str]:
        """Expose low-confidence source usage without silently discarding potentially useful clues."""

        low_count = sum(item.source_credibility_tier == "low" for item in evidence)
        if not low_count:
            return []
        if language.casefold().startswith("en"):
            return [
                f"The report uses {low_count} evidence item(s) from low-credibility sources; "
                "this tier is heuristic and important findings require independent, higher-quality verification."
            ]
        return [
            f"报告使用了 {low_count} 条来自低可信度来源的证据；"
            "该等级为规则启发式，关键结论仍需独立高质量来源复核。"
        ]

    @staticmethod
    def _to_markdown(
        task: ResearchTask,
        key_findings: list[str],
        evidence: list[Evidence],
        redteam_findings: list[RedTeamFinding],
        limitations: list[str],
    ) -> str:
        """Render a Markdown report from structured report sections."""

        english = task.language.casefold().startswith("en")
        headings = (
            ("Summary", "Key Findings", "Evidence", "Limitations")
            if english
            else ("摘要", "关键发现", "证据表", "局限性")
        )
        lines = [
            f"# Deep Research Report: {task.query}",
            "",
            f"## {headings[0]}",
            (
                "This report organizes findings from traceable retrieved material and states limitations where evidence is insufficient."
                if english
                else "本报告根据本次检索到的可追溯材料组织结论，并在证据不足处保留限制说明。"
            ),
            "",
            f"## {headings[1]}",
        ]
        lines.extend(f"- {finding}" for finding in key_findings)
        lines.extend(["", f"## {headings[2]}"])
        for index, item in enumerate(evidence, start=1):
            quote = item.quote.replace("\n", "\n   ")
            lines.append(f"{index}. {quote}  ")
            if english:
                lines.append(
                    f"   Source: [{item.source_title}]({item.source_url}); "
                    f"extraction confidence: {item.confidence:.2f}; "
                    "source credibility (rule estimate): "
                    f"{item.source_credibility_tier or 'unknown'}"
                )
            else:
                lines.append(
                    f"   来源：[{item.source_title}]({item.source_url})，"
                    f"抽取置信度：{item.confidence:.2f}，"
                    f"来源可信度（规则估计）：{item.source_credibility_tier or 'unknown'}"
                )
        lines.extend(["", f"## {headings[3]}"])
        lines.extend(f"- {item}" for item in limitations)
        return "\n".join(lines) + "\n"


class LLMSynthesizerAgent:
    """Generate report content with an LLM while binding citations locally."""

    def __init__(
        self,
        llm: LLMClient,
        *,
        fallback: SynthesizerAgent | None = None,
        fallback_to_rule: bool = True,
    ) -> None:
        self.llm = llm
        self.fallback = fallback or SynthesizerAgent()
        self.fallback_to_rule = fallback_to_rule
        self.last_trace = SynthesizerTrace(
            requested_provider="llm",
            used_provider="llm",
        )

    def synthesize(
        self,
        task: ResearchTask,
        evidence: list[Evidence],
        redteam_findings: list[RedTeamFinding] | None = None,
    ) -> ResearchReport:
        """Synchronous wrapper for callers outside the async Pipeline."""

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.synthesize_async(
                    task,
                    evidence,
                    redteam_findings,
                    coverage_matrix=coverage_matrix,
                )
            )
        raise RuntimeError(
            "LLMSynthesizerAgent.synthesize() cannot run inside an event loop; "
            "await synthesize_async()."
        )

    async def synthesize_async(
        self,
        task: ResearchTask,
        evidence: list[Evidence],
        redteam_findings: list[RedTeamFinding] | None = None,
        *,
        coverage_matrix: EvidenceCoverageMatrix | None = None,
    ) -> ResearchReport:
        """Generate a structured report and reject invented citation IDs."""

        redteam_findings = redteam_findings or []
        unique_evidence = self.fallback._dedupe_evidence(evidence)
        diagnostics: list[FindingSubstanceDiagnostic] = []
        methodology_diagnostics: list[MethodologyObligationDiagnostic] = []
        source_absence_diagnostics: list[SourceAbsenceDiagnostic] = []
        try:
            messages = self._build_messages(
                task,
                unique_evidence,
                redteam_findings,
                coverage_matrix=coverage_matrix,
            )
            output = await self.llm.json_chat(
                messages,
                SynthesizerOutput,
                agent="synthesizer",
            )
            diagnostics = self._substance_diagnostics(task, output, unique_evidence)
            methodology_diagnostics = self._methodology_diagnostics(
                task, output, unique_evidence
            )
            source_absence_diagnostics = self._source_absence_diagnostics(
                output, unique_evidence
            )
            repairable_methodology = self._repairable_methodology_gaps(
                methodology_diagnostics
            )
            unverified_absence = self._unverified_source_absence(
                source_absence_diagnostics
            )
            invalid_substance = self._invalid_substance(diagnostics)
            if invalid_substance or repairable_methodology or unverified_absence:
                # Post-parse semantic shape checks cannot use the JSON parser's
                # retry hook, so perform one bounded structured correction call.
                output = await self.llm.json_chat(
                    self._build_quality_retry_messages(
                        messages,
                        output,
                        diagnostics,
                        methodology_diagnostics,
                        source_absence_diagnostics,
                    ),
                    SynthesizerOutput,
                    agent=(
                        "synthesizer_substance_retry"
                        if invalid_substance
                        and not repairable_methodology
                        and not unverified_absence
                        else "synthesizer_methodology_retry"
                    ),
                )
                diagnostics = self._substance_diagnostics(task, output, unique_evidence)
                methodology_diagnostics = self._methodology_diagnostics(
                    task, output, unique_evidence
                )
                source_absence_diagnostics = self._source_absence_diagnostics(
                    output, unique_evidence
                )
                if self._invalid_substance(diagnostics):
                    raise ValueError("Synthesizer retry still contains non-substantive findings.")
                output = self._drop_unverified_source_absence_findings(
                    output,
                    source_absence_diagnostics,
                )
                # Pruning can change stable finding indexes and methodology coverage.
                diagnostics = self._substance_diagnostics(task, output, unique_evidence)
                methodology_diagnostics = self._methodology_diagnostics(
                    task, output, unique_evidence
                )
                source_absence_diagnostics = self._source_absence_diagnostics(
                    output, unique_evidence
                )
            self._validate_citations(
                output,
                unique_evidence,
                require_citations=task.require_citations,
            )
            report = self._build_report(
                task,
                output,
                unique_evidence,
                redteam_findings,
            )
            self.last_trace = SynthesizerTrace(
                requested_provider="llm",
                used_provider="llm",
                substance_diagnostics=[
                    item.model_dump(mode="json") for item in diagnostics
                ],
                methodology_diagnostics=[
                    item.model_dump(mode="json") for item in methodology_diagnostics
                ],
                source_absence_diagnostics=[
                    item.model_dump(mode="json") for item in source_absence_diagnostics
                ],
            )
            return report
        except Exception as exc:  # noqa: BLE001 - provider, parsing, and grounding share fallback.
            self.last_trace = SynthesizerTrace(
                requested_provider="llm",
                used_provider="rule" if self.fallback_to_rule else "llm",
                fallback_used=self.fallback_to_rule,
                error=f"{type(exc).__name__}: {exc}",
                substance_diagnostics=[
                    item.model_dump(mode="json") for item in diagnostics
                ],
                methodology_diagnostics=[
                    item.model_dump(mode="json") for item in methodology_diagnostics
                ],
                source_absence_diagnostics=[
                    item.model_dump(mode="json") for item in source_absence_diagnostics
                ],
            )
            if not self.fallback_to_rule:
                raise
            return self.fallback.synthesize(task, unique_evidence, redteam_findings)

    def revise(
        self,
        task: ResearchTask,
        draft: ResearchReport,
        evidence: list[Evidence],
        redteam_findings: list[RedTeamFinding],
        *,
        coverage_matrix: EvidenceCoverageMatrix | None = None,
    ) -> ResearchReport:
        """Synchronous wrapper for one evidence-grounded Blue revision."""

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.revise_async(
                    task,
                    draft,
                    evidence,
                    redteam_findings,
                    coverage_matrix=coverage_matrix,
                )
            )
        raise RuntimeError(
            "LLMSynthesizerAgent.revise() cannot run inside an event loop; "
            "await revise_async()."
        )

    async def revise_async(
        self,
        task: ResearchTask,
        draft: ResearchReport,
        evidence: list[Evidence],
        redteam_findings: list[RedTeamFinding],
        *,
        coverage_matrix: EvidenceCoverageMatrix | None = None,
    ) -> ResearchReport:
        """Revise one draft once, using only supplied evidence and actionable review findings.

        修订失败时保留原始 draft，比重新调用规则 Synthesizer 更安全：后者可能改变原报告结构，
        也无法真正响应 RedTeam 指出的具体问题。
        """

        unique_evidence = self.fallback._dedupe_evidence(evidence)
        diagnostics: list[FindingSubstanceDiagnostic] = []
        methodology_diagnostics: list[MethodologyObligationDiagnostic] = []
        source_absence_diagnostics: list[SourceAbsenceDiagnostic] = []
        try:
            output = await self.llm.json_chat(
                self._build_revision_messages(
                    task,
                    draft,
                    unique_evidence,
                    redteam_findings,
                    coverage_matrix=coverage_matrix,
                ),
                SynthesizerOutput,
                agent="blue_revision",
            )
            output = self._drop_uncited_revision_findings(
                output,
                require_citations=task.require_citations,
            )
            output, diagnostics = self._drop_non_substantive_revision_findings(
                task,
                output,
                unique_evidence,
            )
            source_absence_diagnostics = self._source_absence_diagnostics(
                output, unique_evidence
            )
            output = self._drop_unverified_source_absence_findings(
                output,
                source_absence_diagnostics,
            )
            methodology_diagnostics = self._methodology_diagnostics(
                task, output, unique_evidence
            )
            source_absence_diagnostics = self._source_absence_diagnostics(
                output, unique_evidence
            )
            self._validate_citations(
                output,
                unique_evidence,
                require_citations=task.require_citations,
            )
            revised = self._build_report(task, output, unique_evidence, [])
            self.last_trace = SynthesizerTrace(
                requested_provider="llm",
                used_provider="llm",
                substance_diagnostics=[
                    item.model_dump(mode="json") for item in diagnostics
                ],
                methodology_diagnostics=[
                    item.model_dump(mode="json") for item in methodology_diagnostics
                ],
                source_absence_diagnostics=[
                    item.model_dump(mode="json") for item in source_absence_diagnostics
                ],
            )
            return revised
        except Exception as exc:  # noqa: BLE001 - failed revision must preserve the accepted draft.
            self.last_trace = SynthesizerTrace(
                requested_provider="llm",
                used_provider="draft" if self.fallback_to_rule else "llm",
                fallback_used=self.fallback_to_rule,
                error=f"{type(exc).__name__}: {exc}",
                substance_diagnostics=[
                    item.model_dump(mode="json") for item in diagnostics
                ],
                methodology_diagnostics=[
                    item.model_dump(mode="json") for item in methodology_diagnostics
                ],
                source_absence_diagnostics=[
                    item.model_dump(mode="json") for item in source_absence_diagnostics
                ],
            )
            if not self.fallback_to_rule:
                raise
            return draft

    async def revise_text_only_async(
        self,
        task: ResearchTask,
        draft: ResearchReport,
        evidence: list[Evidence],
        verifier_findings: list[RedTeamFinding],
        *,
        coverage_matrix: EvidenceCoverageMatrix | None = None,
    ) -> ResearchReport:
        """Apply one final wording-only repair without retrieval or new evidence.

        该入口只由 Pipeline 在判定所有剩余问题都属于措辞/逻辑边界时调用。
        本地引用校验仍会拒绝未知 Evidence ID；调用失败时保留已核验的 draft。
        """

        unique_evidence = self.fallback._dedupe_evidence(evidence)
        diagnostics: list[FindingSubstanceDiagnostic] = []
        methodology_diagnostics: list[MethodologyObligationDiagnostic] = []
        source_absence_diagnostics: list[SourceAbsenceDiagnostic] = []
        try:
            output = await self.llm.json_chat(
                self._build_final_text_revision_messages(
                    task,
                    draft,
                    unique_evidence,
                    verifier_findings,
                    coverage_matrix=coverage_matrix,
                ),
                SynthesizerOutput,
                agent="final_text_revision",
            )
            output = self._drop_uncited_revision_findings(
                output,
                require_citations=task.require_citations,
            )
            output, diagnostics = self._drop_non_substantive_revision_findings(
                task,
                output,
                unique_evidence,
            )
            source_absence_diagnostics = self._source_absence_diagnostics(
                output, unique_evidence
            )
            output = self._drop_unverified_source_absence_findings(
                output,
                source_absence_diagnostics,
            )
            methodology_diagnostics = self._methodology_diagnostics(
                task, output, unique_evidence
            )
            source_absence_diagnostics = self._source_absence_diagnostics(
                output, unique_evidence
            )
            self._validate_citations(
                output,
                unique_evidence,
                require_citations=task.require_citations,
            )
            # 终局文字修订只能删减或收窄现有报告；即使某条 Evidence 在完整上下文中
            # 可用，也不能在这个阶段悄悄引入原报告未引用的新证据。
            draft_evidence_ids = {item.id for item in draft.evidence}
            revised_evidence_ids = {
                evidence_id
                for finding in output.key_findings
                for evidence_id in finding.evidence_ids
            }
            newly_cited_ids = sorted(revised_evidence_ids - draft_evidence_ids)
            if newly_cited_ids:
                raise ValueError(
                    "Final text-only revision introduced new evidence IDs: "
                    f"{', '.join(newly_cited_ids)}."
                )
            revised = self._build_report(task, output, unique_evidence, [])
            self.last_trace = SynthesizerTrace(
                requested_provider="llm",
                used_provider="llm",
                substance_diagnostics=[
                    item.model_dump(mode="json") for item in diagnostics
                ],
                methodology_diagnostics=[
                    item.model_dump(mode="json") for item in methodology_diagnostics
                ],
                source_absence_diagnostics=[
                    item.model_dump(mode="json") for item in source_absence_diagnostics
                ],
            )
            return revised
        except Exception as exc:  # noqa: BLE001 - preserve the last verified report on repair failure.
            self.last_trace = SynthesizerTrace(
                requested_provider="llm",
                used_provider="draft" if self.fallback_to_rule else "llm",
                fallback_used=self.fallback_to_rule,
                error=f"{type(exc).__name__}: {exc}",
                substance_diagnostics=[
                    item.model_dump(mode="json") for item in diagnostics
                ],
                methodology_diagnostics=[
                    item.model_dump(mode="json") for item in methodology_diagnostics
                ],
                source_absence_diagnostics=[
                    item.model_dump(mode="json") for item in source_absence_diagnostics
                ],
            )
            if not self.fallback_to_rule:
                raise
            return draft

    @staticmethod
    def _drop_uncited_revision_findings(
        output: SynthesizerOutput,
        *,
        require_citations: bool,
    ) -> SynthesizerOutput:
        """Delete isolated uncited findings from a repair response before strict validation.

        Blue 修订的目标本就包含删除不可支持内容。当模型在一份其他部分
        可用的修订中留下单条无引用 finding 时，确定性删除该条比回退整份
        草稿更安全。这里不会猜测或自动补全 Evidence ID；若删除后无内容则仍失败。
        """

        if not require_citations:
            return output
        cited_findings = [item for item in output.key_findings if item.evidence_ids]
        if not cited_findings:
            raise ValueError("Revision has no evidence-grounded findings after pruning.")
        if len(cited_findings) == len(output.key_findings):
            return output
        return output.model_copy(update={"key_findings": cited_findings})

    @staticmethod
    def _substance_diagnostics(
        task: ResearchTask,
        output: SynthesizerOutput,
        evidence: list[Evidence],
    ) -> list[FindingSubstanceDiagnostic]:
        """Assess model-authored findings before local citation rendering."""

        return assess_report_findings(
            task.query,
            [item.text for item in output.key_findings],
            evidence,
        )

    @staticmethod
    def _invalid_substance(
        diagnostics: list[FindingSubstanceDiagnostic],
    ) -> list[FindingSubstanceDiagnostic]:
        return [item for item in diagnostics if item.status != "substantive"]

    @staticmethod
    def _finding_inputs(output: SynthesizerOutput) -> list[tuple[str, list[str]]]:
        """Expose the model's explicit finding-to-Evidence mapping to local gates."""

        return [(item.text, item.evidence_ids) for item in output.key_findings]

    @classmethod
    def _methodology_diagnostics(
        cls,
        task: ResearchTask,
        output: SynthesizerOutput,
        evidence: list[Evidence],
    ) -> list[MethodologyObligationDiagnostic]:
        return assess_methodology_obligations(
            task.query,
            cls._finding_inputs(output),
            evidence,
        )

    @classmethod
    def _source_absence_diagnostics(
        cls,
        output: SynthesizerOutput,
        evidence: list[Evidence],
    ) -> list[SourceAbsenceDiagnostic]:
        return assess_source_absence_claims(cls._finding_inputs(output), evidence)

    @staticmethod
    def _repairable_methodology_gaps(
        diagnostics: list[MethodologyObligationDiagnostic],
    ) -> list[MethodologyObligationDiagnostic]:
        """Only missing wording with available Evidence is safe to ask the model to add."""

        return [item for item in diagnostics if item.status == "missing"]

    @staticmethod
    def _unverified_source_absence(
        diagnostics: list[SourceAbsenceDiagnostic],
    ) -> list[SourceAbsenceDiagnostic]:
        return [item for item in diagnostics if item.status == "unverified"]

    @classmethod
    def _drop_unverified_source_absence_findings(
        cls,
        output: SynthesizerOutput,
        diagnostics: list[SourceAbsenceDiagnostic],
    ) -> SynthesizerOutput:
        """Remove source-absence assertions that were inferred from retrieval scope."""

        invalid_indexes = {
            item.finding_index for item in diagnostics if item.status == "unverified"
        }
        if not invalid_indexes:
            return output
        retained = [
            item
            for index, item in enumerate(output.key_findings, start=1)
            if index not in invalid_indexes
        ]
        if not retained:
            raise ValueError("Revision has no grounded findings after source-absence pruning.")
        limitation = (
            "本次检索材料不足以证明相关来源明确缺少某项内容；"
            "因此报告不把检索范围内的未发现表述为来源事实。"
        )
        limitations = list(output.limitations)
        if limitation not in limitations:
            limitations.append(limitation)
        return output.model_copy(
            update={"key_findings": retained, "limitations": limitations}
        )

    @classmethod
    def _drop_non_substantive_revision_findings(
        cls,
        task: ResearchTask,
        output: SynthesizerOutput,
        evidence: list[Evidence],
    ) -> tuple[SynthesizerOutput, list[FindingSubstanceDiagnostic]]:
        """Prune isolated malformed Blue findings without inventing replacements."""

        diagnostics = cls._substance_diagnostics(task, output, evidence)
        valid = [
            finding
            for finding, diagnostic in zip(
                output.key_findings,
                diagnostics,
                strict=True,
            )
            if diagnostic.status == "substantive"
        ]
        if not valid:
            raise ValueError("Revision has no substantive findings after pruning.")
        if len(valid) == len(output.key_findings):
            return output, diagnostics
        return output.model_copy(update={"key_findings": valid}), diagnostics

    @staticmethod
    def _build_substance_retry_messages(
        original: list[ChatMessage],
        output: SynthesizerOutput,
        diagnostics: list[FindingSubstanceDiagnostic],
    ) -> list[ChatMessage]:
        """Ask once for complete propositions while preserving the JSON contract."""

        invalid = "\n".join(
            f"- finding {item.finding_index}: status={item.status}; reason={item.reason}"
            for item in diagnostics
            if item.status != "substantive"
        )
        return [
            *original,
            ChatMessage(role="assistant", content=output.model_dump_json()),
            ChatMessage(
                role="user",
                content=(
                    "The structured output failed the deterministic user-facing finding gate:\n"
                    f"{invalid}\n"
                    "Return a complete replacement JSON object. Every key finding must be a complete, "
                    "checkable proposition supported by its Evidence IDs. Do not emit source titles, "
                    "author lists, abstract headers, or truncated excerpts as findings."
                ),
            ),
        ]

    @staticmethod
    def _build_quality_retry_messages(
        original: list[ChatMessage],
        output: SynthesizerOutput,
        substance: list[FindingSubstanceDiagnostic],
        methodology: list[MethodologyObligationDiagnostic],
        source_absence: list[SourceAbsenceDiagnostic],
    ) -> list[ChatMessage]:
        """Request one complete replacement for all deterministic report-shape failures."""

        issues: list[str] = []
        issues.extend(
            f"finding {item.finding_index}: status={item.status}; reason={item.reason}"
            for item in substance
            if item.status != "substantive"
        )
        issues.extend(
            f"methodology {item.obligation_type}: status={item.status}; reason={item.reason}"
            for item in methodology
            if item.status == "missing"
        )
        issues.extend(
            f"finding {item.finding_index}: source_absence=unverified; reason={item.reason}"
            for item in source_absence
            if item.status == "unverified"
        )
        return [
            *original,
            ChatMessage(role="assistant", content=output.model_dump_json()),
            ChatMessage(
                role="user",
                content=(
                    "The structured output failed deterministic user-facing quality obligations:\n"
                    + "\n".join(f"- {item}" for item in issues)
                    + "\nReturn a complete replacement JSON object. Put every required operational "
                    "definition in a directly cited key finding. Keep findings complete and atomic. "
                    "Do not state that a source lacks data, experiments, definitions, or validation "
                    "unless one of that finding's Evidence quotes explicitly says so; move a limitation "
                    "about this retrieval scope to limitations instead. Never invent missing premises. "
                    "For an explicit Red-Blue process query, retain separate cited findings for Blue revision "
                    "grounded in Evidence and Verifier rechecking after revision while marking remaining uncertainty."
                ),
            ),
        ]

    @staticmethod
    def _build_messages(
        task: ResearchTask,
        evidence: list[Evidence],
        redteam_findings: list[RedTeamFinding],
        *,
        coverage_matrix: EvidenceCoverageMatrix | None = None,
    ) -> list[ChatMessage]:
        """Build a compact provider-neutral prompt containing traceable evidence."""

        evidence_text = "\n".join(
            (
                f"- id={item.id}; claim={item.claim}; quote={item.quote}; "
                f"source={item.source_title}; "
                f"reader_extraction_confidence={item.confidence:.2f}; "
                f"source_authority_heuristic_tier={item.source_credibility_tier or 'unknown'}"
            )
            for item in evidence
        ) or "- no evidence available"
        redteam_text = "\n".join(
            f"- {item.severity}: {item.issue}; fix={item.suggested_fix}"
            for item in redteam_findings
        ) or "- no review findings"
        coverage_text = LLMSynthesizerAgent._format_coverage_matrix(coverage_matrix)
        system = (
            "You are the synthesis component of a deep research system. "
            "Use only the supplied evidence and do not invent facts, sources, URLs, or evidence IDs. "
            "Return only one valid JSON object matching this shape: "
            '{"title":"...","summary":"...","key_findings":['
            '{"text":"...","evidence_ids":["ev_id"]}],'
            '"limitations":["..."]}. '
            "Every factual key finding must cite one or more supplied evidence IDs. "
            "Keep each key finding atomic: do not combine conclusions from unrelated studies or "
            "attach one citation set to multiple independently checkable factual claims. "
            "Every cited Evidence item must directly support the complete factual wording of that finding. "
            "Bibliographic metadata alone can establish that a study exists, but cannot establish its results. "
            "Do not emit a key finding that merely says a bibliography lists, cites, or ranks a paper unless "
            "the research query explicitly asks for bibliometric evidence. When the query asks to compare "
            "studies, identify each compared study by a supplied title, author, or year and make the "
            "head-to-head comparison explicit; never invent missing identity details. "
            "If the query explicitly requires at least N empirical studies, include a directly cited result "
            "or conclusion for N distinct identifiable studies; a method or bibliography description alone "
            "does not satisfy that requirement. "
            "reader_extraction_confidence measures quote/extraction fidelity; it does not measure "
            "the authority of the source. source_authority_heuristic_tier is the separate, "
            "rule-based source-quality estimate. Never describe one as the other. "
            "If evidence is insufficient, state that as a limitation instead of guessing. "
            "Do not turn absence from this retrieval window into a source fact: claims such as a paper "
            "having no experiment, definition, data, or validation require an Evidence quote that "
            "explicitly states that absence. Otherwise describe only the limits of this report. "
            "When the query asks how to evaluate or design a method, provide an explicit operational plan rather "
            "than only summarizing papers: state the analysis unit, numerator/denominator for rates where the "
            "evidence permits it, category taxonomy, and evaluation procedure. Put these operational definitions "
            "in directly cited key_findings, not only in the summary or limitations. For citation/hallucination "
            "evaluation questions, keep three distinct operational findings when their premises are supported: "
            "citation coverage as the proportion of evidence-requiring atomic claims with valid citations; "
            "citation correctness as semantic entailment/support rather than marker or link existence; and "
            "atomic-claim error accounting that distinguishes unsupported, contradicted, miscited, and uncited "
            "claims and reports category counts over atomic claims. A definition synthesized from "
            "multiple sources must be clearly phrased as a proposed adaptation or recommendation, must cite all "
            "Evidence needed for its premises, and must not be attributed to any one source as its original metric. "
            "When the research query explicitly asks about a Red-Blue verification process, include distinct cited "
            "findings that explain evidence-grounded Blue revision and Verifier rechecking after revision while "
            "preserving remaining uncertainty. State only workflow details that the supplied Evidence supports. "
            "Use the deterministic coverage matrix as a checklist: cover cells with Evidence IDs, "
            "and explicitly limit only cells marked missing. Never claim an entire subject is absent "
            "when the matrix lists evidence for that subject. "
            "The output is a user-facing report: never mention the coverage matrix, MISSING cells, "
            "Evidence IDs, retries, fallback, Replay, or other backend logs. Do not expose this run's internal "
            "RedTeam/Verifier events. If Red-Blue or Verifier is the explicit research subject, discuss the supplied "
            "subject evidence without presenting it as a log of this run. "
            "Describe unavailable material naturally as a source limitation. "
            "Do not include Markdown citation numbers; the application binds citations locally."
        )
        user = (
            f"Research query: {task.query}\n"
            f"Language: {task.language}\n"
            f"Domain: {task.domain or 'unspecified'}\n"
            f"Time range: {task.time_range or 'unspecified'}\n"
            f"Report length: {task.report_length}\n"
            f"Citations required: {task.require_citations}\n\n"
            f"Evidence:\n{evidence_text}\n\n"
            f"RedTeam review:\n{redteam_text}\n\n"
            f"Evidence coverage matrix:\n{coverage_text}"
        )
        return [
            ChatMessage(role="system", content=system),
            ChatMessage(role="user", content=user),
        ]

    @staticmethod
    def _build_revision_messages(
        task: ResearchTask,
        draft: ResearchReport,
        evidence: list[Evidence],
        redteam_findings: list[RedTeamFinding],
        *,
        coverage_matrix: EvidenceCoverageMatrix | None = None,
    ) -> list[ChatMessage]:
        """Build a bounded Blue prompt that repairs the draft without inventing evidence."""

        evidence_text = "\n".join(
            (
                f"- id={item.id}; claim={item.claim}; quote={item.quote}; "
                f"source={item.source_title}; "
                f"reader_extraction_confidence={item.confidence:.2f}; "
                f"source_authority_heuristic_tier={item.source_credibility_tier or 'unknown'}"
            )
            for item in evidence
        ) or "- no evidence available"
        review_text = "\n".join(
            (
                f"- status={item.status or 'unclear'}; severity={item.severity}; "
                f"claim={item.claim or 'unspecified'}; issue={item.issue}; "
                f"reason={item.reason or 'unspecified'}; fix={item.suggested_fix}"
            )
            for item in redteam_findings
        ) or "- no actionable review findings"
        coverage_text = LLMSynthesizerAgent._format_coverage_matrix(coverage_matrix)
        system = (
            "You are the Blue revision component of a deep research system. "
            "Revise the draft once to address every supplied RedTeam issue. "
            "Use only the supplied evidence; do not invent facts, sources, URLs, or evidence IDs. "
            "Prefer narrowing or removing unsupported claims over preserving confident wording. "
            "Return only one valid JSON object matching this shape: "
            '{"title":"...","summary":"...","key_findings":['
            '{"text":"...","evidence_ids":["ev_id"]}],'
            '"limitations":["..."]}. '
            "Every factual key finding must cite supplied evidence IDs. "
            "Keep every key finding atomic and retain only wording directly supported by all attached "
            "Evidence. If a prior citation is mismatched, remove or narrow the claim instead of "
            "rephrasing the unsupported detail. Bibliographic metadata does not support study results. "
            "Remove bibliography-only key findings unless the query explicitly asks for bibliometrics. "
            "For a study-comparison query, keep the compared studies explicitly identifiable from supplied "
            "titles, authors, or years and state the comparison directly without inventing metadata. "
            "Preserve any explicit minimum count of empirical studies: each counted study needs its own "
            "directly cited result or conclusion, not merely an author, topic, or method description. "
            "reader_extraction_confidence is extraction fidelity, while "
            "source_authority_heuristic_tier is a separate rule-based source estimate; do not "
            "merge or relabel these concepts. "
            "A claim that a source has no experiment, definition, data, or validation is allowed only "
            "when its cited Evidence explicitly states that absence. If it is merely absent from the "
            "retrieved excerpt, remove it from key findings and describe the retrieval limit naturally. "
            "For methodology queries, preserve or repair explicit operational definitions: analysis unit, "
            "rate numerator/denominator, error categories, and procedure, and keep them as directly cited "
            "key_findings rather than moving them only to summary or limitations. For citation/hallucination "
            "evaluation, preserve distinct findings for citation coverage, semantic citation support, and "
            "atomic-claim error accounting across unsupported, contradicted, miscited, and uncited claims. "
            "For an explicit Red-Blue verification query, preserve separate cited findings for evidence-grounded "
            "Blue revision and Verifier rechecking after revision with remaining uncertainty. Do not add retrieval "
            "or blocking semantics unless supplied Evidence states them. "
            "Mark cross-source combinations as "
            "proposed adaptations and cite every Evidence premise instead of attributing the combined metric "
            "to one paper. "
            "Use every covered matrix cell relevant to the query; describe only missing cells as "
            "limitations, and do not call a whole subject absent when it has listed Evidence IDs. "
            "The output is user-facing. Do not expose coverage-matrix labels, MISSING, Evidence IDs, "
            "retries, fallback, Replay, or backend diagnostics. Do not reveal this run's internal RedTeam/Verifier "
            "events; when that workflow is itself the research subject, discuss only the supplied subject evidence. "
            "Do not include Markdown citation numbers because the application binds them locally."
        )
        user = (
            f"Research query: {task.query}\n"
            f"Language: {task.language}\n"
            f"Citations required: {task.require_citations}\n\n"
            f"Draft report:\n{draft.markdown}\n\n"
            f"Actionable RedTeam findings:\n{review_text}\n\n"
            f"Evidence table:\n{evidence_text}\n\n"
            f"Evidence coverage matrix:\n{coverage_text}"
        )
        return [
            ChatMessage(role="system", content=system),
            ChatMessage(role="user", content=user),
        ]

    @staticmethod
    def _build_final_text_revision_messages(
        task: ResearchTask,
        draft: ResearchReport,
        evidence: list[Evidence],
        verifier_findings: list[RedTeamFinding],
        *,
        coverage_matrix: EvidenceCoverageMatrix | None = None,
    ) -> list[ChatMessage]:
        """Build the stricter prompt for the single post-verification text repair."""

        evidence_text = "\n".join(
            (
                f"- id={item.id}; claim={item.claim}; quote={item.quote}; "
                f"source={item.source_title}; "
                f"reader_extraction_confidence={item.confidence:.2f}; "
                f"source_authority_heuristic_tier={item.source_credibility_tier or 'unknown'}"
            )
            for item in evidence
        ) or "- no evidence available"
        findings_text = "\n".join(
            (
                f"- category={item.category}; status={item.status or 'unclear'}; "
                f"repair_action={item.repair_action}; issue={item.issue}; "
                f"fix={item.suggested_fix}"
            )
            for item in verifier_findings
        )
        coverage_text = LLMSynthesizerAgent._format_coverage_matrix(coverage_matrix)
        system = (
            "You are performing the one allowed final text-only repair after verification. "
            "Fix only wording, logical boundaries, terminology, or overstatement identified below. "
            "Do not add facts, retrieval results, sources, URLs, or evidence IDs. Do not claim that "
            "missing evidence was found. Narrow or remove text when support is insufficient. "
            "For repair_action=remove, delete the unsupported claim or detail. For narrow, retain only "
            "what the cited Evidence directly supports. For qualify, make the existing source and "
            "uncertainty boundary explicit without turning an unsupported claim into a supported one. "
            "Keep findings atomic. Do not reuse a citation for a different study or for an additional "
            "factual clause. A bibliographic reference without result text supports only study identity. "
            "Do not assert that a source lacks experiments, definitions, data, or validation unless the "
            "cited Evidence explicitly states that absence. "
            "Delete bibliography-only findings unless bibliometrics is part of the user query. In study "
            "comparisons, preserve explicit supplied study identifiers and a direct comparison. "
            "Do not claim an explicit minimum-study requirement is met unless each counted study retains "
            "a directly cited empirical result or conclusion. "
            "The output is user-facing: do not mention repair_action, coverage matrices, MISSING, "
            "Evidence IDs, retries, fallback, Replay, or backend diagnostics. Do not reveal this run's internal "
            "RedTeam/Verifier events; when that workflow is itself the research subject, discuss only the supplied "
            "subject evidence. "
            "reader_extraction_confidence means extraction fidelity; "
            "source_authority_heuristic_tier is a separate rule-based source-quality estimate. "
            "Return exactly one valid JSON object shaped as "
            '{"title":"...","summary":"...","key_findings":['
            '{"text":"...","evidence_ids":["ev_id"]}],"limitations":["..."]}. '
            "Every factual key finding must cite only supplied evidence IDs. Markdown citation "
            "numbers are forbidden because the application binds them locally."
        )
        user = (
            f"Research query: {task.query}\n\n"
            f"Current report:\n{draft.markdown}\n\n"
            f"Verifier wording/boundary findings:\n{findings_text}\n\n"
            f"Existing evidence table (reference only; no new evidence):\n{evidence_text}\n\n"
            f"Evidence coverage matrix:\n{coverage_text}"
        )
        return [
            ChatMessage(role="system", content=system),
            ChatMessage(role="user", content=user),
        ]

    @staticmethod
    def _format_coverage_matrix(matrix: EvidenceCoverageMatrix | None) -> str:
        """Render compact deterministic coverage rows without duplicating Evidence bodies."""

        if matrix is None or not matrix.subjects:
            return "- unavailable"
        rows: list[str] = []
        for subject in matrix.subjects:
            cells = "; ".join(
                f"{dimension}={','.join(ids) if ids else 'MISSING'}"
                for dimension, ids in subject.dimension_evidence_ids.items()
            )
            rows.append(f"- subject={subject.subject}; {cells}")
        return "\n".join(rows)

    @staticmethod
    def _validate_citations(
        output: SynthesizerOutput,
        evidence: list[Evidence],
        *,
        require_citations: bool,
    ) -> None:
        """Require model citations to reference only the supplied evidence."""

        known_ids = {item.id for item in evidence}
        for index, finding in enumerate(output.key_findings, start=1):
            unknown_ids = [
                evidence_id
                for evidence_id in finding.evidence_ids
                if evidence_id not in known_ids
            ]
            if unknown_ids:
                raise ValueError(
                    f"Finding {index} references unknown evidence IDs: "
                    f"{', '.join(unknown_ids)}."
                )
            if require_citations and known_ids and not finding.evidence_ids:
                raise ValueError(f"Finding {index} has no evidence citation.")

    def _build_report(
        self,
        task: ResearchTask,
        output: SynthesizerOutput,
        evidence: list[Evidence],
        redteam_findings: list[RedTeamFinding],
    ) -> ResearchReport:
        """Bind validated evidence IDs to stable local citation numbers."""

        cited_ids = {
            evidence_id
            for finding in output.key_findings
            for evidence_id in finding.evidence_ids
        }
        # 报告证据表只展示正文实际引用的条目；完整上下文仍保留在 Replay。
        report_evidence = [item for item in evidence if item.id in cited_ids]
        citation_numbers = {
            item.id: index
            for index, item in enumerate(report_evidence, start=1)
        }
        key_findings = []
        for finding in output.key_findings:
            citations = "".join(
                f" [{citation_numbers[evidence_id]}]"
                for evidence_id in dict.fromkeys(finding.evidence_ids)
            )
            key_findings.append(f"{finding.text.strip()}{citations}")

        # 来源等级与计数由应用根据最终实际引用 Evidence 统一生成；丢弃模型自行写出的
        # 同类提示，避免多轮修订后出现重复或与最终证据表数量不一致。
        limitations = [
            item.strip()
            for item in output.limitations
            if item.strip() and not self._is_model_source_credibility_limitation(item)
        ]
        for limitation in self.fallback._source_credibility_limitations(
            report_evidence,
            language=task.language,
        ):
            if limitation not in limitations:
                limitations.append(limitation)
        markdown = self._to_markdown(
            title=output.title.strip(),
            summary=output.summary.strip(),
            key_findings=key_findings,
            evidence=report_evidence,
            redteam_findings=redteam_findings,
            limitations=limitations,
            language=task.language,
        )
        return ResearchReport(
            run_id=task.run_id,
            query=task.query,
            title=output.title.strip(),
            summary=output.summary.strip(),
            key_findings=key_findings,
            evidence=report_evidence,
            limitations=limitations,
            redteam_findings=redteam_findings,
            markdown=markdown,
        )

    @staticmethod
    def _is_model_source_credibility_limitation(text: str) -> bool:
        """Recognize application-owned source-tier disclosures in model-authored wording."""

        lowered = text.casefold()
        return (
            "低可信度" in text
            or "low-credibility" in lowered
            or "low credibility" in lowered
            or "source_authority_heuristic_tier" in lowered
        )

    @staticmethod
    def _to_markdown(
        *,
        title: str,
        summary: str,
        key_findings: list[str],
        evidence: list[Evidence],
        redteam_findings: list[RedTeamFinding],
        limitations: list[str],
        language: str = "zh",
    ) -> str:
        """Render model-authored sections with application-owned citations."""

        english = language.casefold().startswith("en")
        headings = (
            ("Summary", "Key Findings", "Evidence", "Limitations")
            if english
            else ("摘要", "关键发现", "证据表", "局限性")
        )
        lines = [
            f"# {title}",
            "",
            f"## {headings[0]}",
            summary,
            "",
            f"## {headings[1]}",
        ]
        lines.extend(f"- {finding}" for finding in key_findings)
        lines.extend(["", f"## {headings[2]}"])
        for index, item in enumerate(evidence, start=1):
            quote = item.quote.replace("\n", "\n   ")
            lines.append(f"{index}. {quote}  ")
            if english:
                lines.append(
                    f"   Source: [{item.source_title}]({item.source_url}); "
                    f"extraction confidence: {item.confidence:.2f}; "
                    "source credibility (rule estimate): "
                    f"{item.source_credibility_tier or 'unknown'}"
                )
            else:
                lines.append(
                    f"   来源：[{item.source_title}]({item.source_url})，"
                    f"抽取置信度：{item.confidence:.2f}，"
                    f"来源可信度（规则估计）：{item.source_credibility_tier or 'unknown'}"
                )
        lines.extend(["", f"## {headings[3]}"])
        if limitations:
            lines.extend(f"- {item}" for item in limitations)
        else:
            lines.append(
                "- The model did not add further limitations."
                if english
                else "- 模型未补充额外局限性。"
            )
        return "\n".join(lines) + "\n"

"""Rule and LLM planners that produce validated research DAGs."""

from __future__ import annotations

import asyncio
from collections import Counter, deque
import re
from typing import Literal

from pydantic import BaseModel, Field

from deepresearch.llm import ChatMessage, LLMClient
from deepresearch.schema import PlannerTrace, ResearchTask, SubTask


class PlannedSubTask(BaseModel):
    """Restricted schema exposed to the model; runtime status is assigned locally."""

    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    depends_on: list[str] = Field(default_factory=list)
    task_type: Literal["research", "synthesis"] = "research"
    # 面向搜索引擎的短查询与 Reader 需要回答的完整研究问题分离。
    retrieval_query: str | None = None


class PlannerOutput(BaseModel):
    """Structured output required from an LLM planner."""

    reasoning_summary: str = ""
    subtasks: list[PlannedSubTask] = Field(min_length=1)


class PlannerAgent:
    """Create the deterministic fallback DAG used by offline runs and recovery."""

    def __init__(self, max_subtasks: int = 4) -> None:
        self.max_subtasks = max(1, max_subtasks)

    def plan(self, task: ResearchTask) -> list[SubTask]:
        """Split one research query into a stable branching DAG."""

        fragments = [part.strip() for part in re.split(r"[?？;；。]", task.query) if part.strip()]
        if not fragments:
            fragments = [task.query]

        templates = [
            "梳理问题背景与关键概念：{query}",
            "收集支持性事实与关键证据：{query}",
            "识别争议、风险与反例：{query}",
            "综合形成结论与行动建议：{query}",
        ]
        questions = [
            templates[index].format(query=fragments[min(index, len(fragments) - 1)])
            for index in range(min(self.max_subtasks, len(templates)))
        ]
        subtasks: list[SubTask] = []
        for index, question in enumerate(questions):
            subtasks.append(
                SubTask(
                    id=f"task_{index + 1}",
                    question=question,
                    depends_on=self._depends_on_for(index, subtasks),
                    task_type="synthesis" if index == 3 else "research",
                )
            )
        return subtasks

    @staticmethod
    def _depends_on_for(index: int, existing_subtasks: list[SubTask]) -> list[str]:
        if index == 0:
            return []
        if index in (1, 2):
            return [existing_subtasks[0].id]
        return [subtask.id for subtask in existing_subtasks[1:3]]


class LLMPlannerAgent:
    """Ask the configured model for a DAG, validate it, and optionally fall back."""

    def __init__(
        self,
        llm: LLMClient,
        *,
        max_subtasks: int = 4,
        fallback: PlannerAgent | None = None,
        fallback_to_rule: bool = True,
    ) -> None:
        self.llm = llm
        self.max_subtasks = max(1, max_subtasks)
        self.fallback = fallback or PlannerAgent(max_subtasks=self.max_subtasks)
        self.fallback_to_rule = fallback_to_rule
        self.last_trace = PlannerTrace(requested_provider="llm", used_provider="llm")

    def plan(self, task: ResearchTask) -> list[SubTask]:
        """Synchronous wrapper used by the current CLI Pipeline."""

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.plan_async(task))
        raise RuntimeError("LLMPlannerAgent.plan() cannot run inside an event loop; await plan_async().")

    async def plan_async(self, task: ResearchTask) -> list[SubTask]:
        """Generate and validate a research DAG."""

        try:
            output = await self.llm.json_chat(
                self._build_messages(task),
                PlannerOutput,
                agent="planner",
            )
            subtasks = [
                SubTask(
                    id=item.id,
                    question=item.question.strip(),
                    depends_on=item.depends_on,
                    task_type=item.task_type,
                    retrieval_query=(
                        item.retrieval_query.strip() if item.retrieval_query else None
                    ),
                )
                for item in output.subtasks
            ]
            validate_planned_dag(subtasks, self.max_subtasks)
            self.last_trace = PlannerTrace(
                requested_provider="llm",
                used_provider="llm",
                reasoning_summary=output.reasoning_summary,
            )
            return subtasks
        except Exception as exc:  # noqa: BLE001 - fallback must cover provider, parse, and DAG errors.
            if not self.fallback_to_rule:
                self.last_trace = PlannerTrace(
                    requested_provider="llm",
                    used_provider="llm",
                    error=f"{type(exc).__name__}: {exc}",
                )
                raise
            subtasks = self.fallback.plan(task)
            self.last_trace = PlannerTrace(
                requested_provider="llm",
                used_provider="rule",
                fallback_used=True,
                error=f"{type(exc).__name__}: {exc}",
            )
            return subtasks

    def _build_messages(self, task: ResearchTask) -> list[ChatMessage]:
        """Build a provider-neutral prompt that requires JSON-only DAG output."""

        system = (
            "You are the planning component of a deep research system. "
            "Return only one valid JSON object matching this shape: "
            '{"reasoning_summary":"short summary","subtasks":['
            '{"id":"task_1","question":"research question","depends_on":[],'
            '"task_type":"research","retrieval_query":"concise search query"}]} . '
            "Create a directed acyclic graph. IDs must be unique. Every dependency must reference "
            "an earlier task. Include background, evidence collection, counter-evidence or risks, "
            'and a final task_type="synthesis" node when multiple research branches need integration. '
            "Synthesis nodes must depend on research nodes and should not request new retrieval. "
            "For comparison questions with multiple named systems, datasets, or benchmarks, create one "
            "research branch per named object when the subtask limit allows it. "
            "For every research node, provide a concise retrieval_query optimized for a web search engine; "
            "prefer exact paper/study names and English scholarly keywords when primary research is needed. "
            "Synthesis nodes should omit retrieval_query. "
            "Do not answer the research question."
        )
        user = (
            f"Research query: {task.query}\n"
            f"Language: {task.language}\n"
            f"Domain: {task.domain or 'unspecified'}\n"
            f"Time range: {task.time_range or 'unspecified'}\n"
            f"Report length: {task.report_length}\n"
            f"Citations required: {task.require_citations}\n"
            f"Maximum subtasks: {self.max_subtasks}"
        )
        return [
            ChatMessage(role="system", content=system),
            ChatMessage(role="user", content=user),
        ]


def validate_planned_dag(subtasks: list[SubTask], max_subtasks: int) -> None:
    """Reject oversized, duplicate, missing-dependency, self-dependent, or cyclic DAGs."""

    if not subtasks:
        raise ValueError("Planner returned no subtasks.")
    if len(subtasks) > max_subtasks:
        raise ValueError(f"Planner returned {len(subtasks)} subtasks; maximum is {max_subtasks}.")

    ids = [subtask.id for subtask in subtasks]
    duplicates = [node_id for node_id, count in Counter(ids).items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicate subtask ids: {', '.join(sorted(duplicates))}.")

    known = set(ids)
    for subtask in subtasks:
        if subtask.task_type == "synthesis" and not subtask.depends_on:
            raise ValueError(f"Synthesis subtask '{subtask.id}' must depend on research outputs.")
        if subtask.id in subtask.depends_on:
            raise ValueError(f"Subtask '{subtask.id}' depends on itself.")
        missing = [dependency for dependency in subtask.depends_on if dependency not in known]
        if missing:
            raise ValueError(
                f"Subtask '{subtask.id}' has unknown dependencies: {', '.join(missing)}."
            )

    indegree = {subtask.id: len(subtask.depends_on) for subtask in subtasks}
    children = {subtask.id: [] for subtask in subtasks}
    for subtask in subtasks:
        for dependency in subtask.depends_on:
            children[dependency].append(subtask.id)

    roots = deque(node_id for node_id, degree in indegree.items() if degree == 0)
    visited = 0
    while roots:
        node_id = roots.popleft()
        visited += 1
        for child in children[node_id]:
            indegree[child] -= 1
            if indegree[child] == 0:
                roots.append(child)
    if visited != len(subtasks):
        raise ValueError("Planner returned a cyclic subtask graph.")


PlannerProvider = Literal["rule", "llm"]

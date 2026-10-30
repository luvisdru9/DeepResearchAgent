import asyncio

import pytest

from deepresearch.agents.planner import (
    LLMPlannerAgent,
    PlannerAgent,
    validate_planned_dag,
)
from deepresearch.llm import MockLLMClient
from deepresearch.schema import ResearchTask, SubTask


def test_planner_creates_branching_dag_dependencies():
    subtasks = PlannerAgent(max_subtasks=4).plan(ResearchTask(query="如何构建可追溯的深度研究系统？"))

    assert [subtask.id for subtask in subtasks] == ["task_1", "task_2", "task_3", "task_4"]
    assert subtasks[0].depends_on == []
    assert subtasks[1].depends_on == ["task_1"]
    assert subtasks[2].depends_on == ["task_1"]
    assert subtasks[3].depends_on == ["task_2", "task_3"]
    assert [subtask.task_type for subtask in subtasks] == [
        "research",
        "research",
        "research",
        "synthesis",
    ]


def test_planner_keeps_valid_dependencies_when_limited_to_two_subtasks():
    subtasks = PlannerAgent(max_subtasks=2).plan(ResearchTask(query="如何降低幻觉？"))

    assert [subtask.id for subtask in subtasks] == ["task_1", "task_2"]
    assert subtasks[0].depends_on == []
    assert subtasks[1].depends_on == ["task_1"]


def test_llm_planner_builds_valid_dynamic_dag():
    client = MockLLMClient(
        responses=[
            """
            {
              "reasoning_summary": "Research background and two evidence branches.",
              "subtasks": [
                {"id": "scope", "question": "Define the scope", "depends_on": []},
                {"id": "evidence", "question": "Collect supporting evidence", "depends_on": ["scope"],
                 "retrieval_query": "primary empirical evidence reliability agents"},
                {"id": "risks", "question": "Collect counter-evidence", "depends_on": ["scope"]},
                {
                  "id": "synthesis",
                  "question": "Synthesize findings",
                  "depends_on": ["evidence", "risks"],
                  "task_type": "synthesis"
                }
              ]
            }
            """
        ]
    )
    planner = LLMPlannerAgent(client, max_subtasks=4)

    subtasks = asyncio.run(planner.plan_async(ResearchTask(query="研究 Agent 的可靠性")))

    assert [subtask.id for subtask in subtasks] == ["scope", "evidence", "risks", "synthesis"]
    assert subtasks[-1].task_type == "synthesis"
    assert subtasks[1].retrieval_query == "primary empirical evidence reliability agents"
    assert planner.last_trace.used_provider == "llm"
    assert planner.last_trace.fallback_used is False
    assert planner.last_trace.reasoning_summary
    assert client.trace_collector.since(0)[0].agent == "planner"


def test_llm_planner_falls_back_when_dag_is_invalid():
    client = MockLLMClient(
        responses=[
            """
            {
              "reasoning_summary": "invalid cycle",
              "subtasks": [
                {"id": "a", "question": "A", "depends_on": ["b"]},
                {"id": "b", "question": "B", "depends_on": ["a"]}
              ]
            }
            """
        ],
        max_retries=0,
    )
    planner = LLMPlannerAgent(client, max_subtasks=4, fallback_to_rule=True)

    subtasks = asyncio.run(planner.plan_async(ResearchTask(query="测试回退")))

    assert [subtask.id for subtask in subtasks] == ["task_1", "task_2", "task_3", "task_4"]
    assert planner.last_trace.used_provider == "rule"
    assert planner.last_trace.fallback_used is True
    assert "cyclic" in (planner.last_trace.error or "")


def test_llm_planner_can_disable_rule_fallback():
    client = MockLLMClient(
        responses=['{"reasoning_summary":"bad","subtasks":[{"id":"a","question":"A","depends_on":["missing"]}]}'],
        max_retries=0,
    )
    planner = LLMPlannerAgent(client, max_subtasks=4, fallback_to_rule=False)

    with pytest.raises(ValueError, match="unknown dependencies"):
        asyncio.run(planner.plan_async(ResearchTask(query="测试失败")))


def test_validate_planned_dag_rejects_duplicate_and_oversized_plans():
    with pytest.raises(ValueError, match="Duplicate"):
        validate_planned_dag(
            [
                SubTask(id="same", question="first"),
                SubTask(id="same", question="second"),
            ],
            max_subtasks=4,
        )

    with pytest.raises(ValueError, match="maximum"):
        validate_planned_dag(
            [SubTask(id=f"task_{index}", question="question") for index in range(3)],
            max_subtasks=2,
        )


def test_validate_planned_dag_rejects_synthesis_without_dependencies():
    with pytest.raises(ValueError, match="must depend"):
        validate_planned_dag(
            [SubTask(id="final", question="synthesize", task_type="synthesis")],
            max_subtasks=1,
        )

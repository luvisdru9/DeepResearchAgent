import asyncio
import time

from deepresearch.orchestrator.dag import DAGExecutor
from deepresearch.schema import NodeStatus, SubTask


def test_dag_executor_runs_nodes_after_dependencies():
    subtasks = [
        SubTask(id="task_1", question="first"),
        SubTask(id="task_2", question="second", depends_on=["task_1"]),
    ]
    order: list[str] = []

    def handler(subtask: SubTask) -> str:
        order.append(subtask.id)
        return subtask.question

    result = DAGExecutor().run(subtasks, handler)

    assert order == ["task_1", "task_2"]
    assert result.results == {"task_1": "first", "task_2": "second"}
    assert [trace.status for trace in result.traces] == [NodeStatus.DONE, NodeStatus.DONE]
    assert [trace.attempts for trace in result.traces] == [1, 1]
    assert all(trace.started_at is not None for trace in result.traces)
    assert all(trace.finished_at >= trace.started_at for trace in result.traces if trace.started_at)


def test_dag_executor_passes_dependency_outputs_to_downstream_handler():
    subtasks = [
        SubTask(id="task_1", question="collect"),
        SubTask(id="task_2", question="synthesize", depends_on=["task_1"]),
    ]
    received_dependencies: dict[str, str] = {}

    def handler(subtask: SubTask, dependency_results: dict[str, str]) -> str:
        if subtask.id == "task_2":
            received_dependencies.update(dependency_results)
        return f"result:{subtask.id}"

    result = DAGExecutor().run(subtasks, handler)

    assert received_dependencies == {"task_1": "result:task_1"}
    assert result.results["task_2"] == "result:task_2"


def test_dag_executor_skips_nodes_after_failed_dependency():
    subtasks = [
        SubTask(id="task_1", question="first"),
        SubTask(id="task_2", question="second", depends_on=["task_1"]),
    ]

    def handler(subtask: SubTask) -> str:
        if subtask.id == "task_1":
            raise RuntimeError("source unavailable")
        return subtask.question

    result = DAGExecutor().run(subtasks, handler)

    assert result.results == {}
    assert subtasks[0].status == NodeStatus.FAILED
    assert subtasks[1].status == NodeStatus.SKIPPED
    assert [trace.status for trace in result.traces] == [NodeStatus.FAILED, NodeStatus.SKIPPED]
    assert [trace.attempts for trace in result.traces] == [1, 0]
    assert "source unavailable" in (result.traces[0].error or "")
    assert "task_1" in (result.traces[1].error or "")
    assert result.traces[1].started_at is None


def test_dag_executor_retries_failed_node_until_success():
    subtasks = [SubTask(id="task_1", question="first")]
    calls = 0

    def handler(subtask: SubTask) -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("temporary failure")
        return subtask.question

    result = DAGExecutor(max_retries=1).run(subtasks, handler)

    assert calls == 2
    assert result.results == {"task_1": "first"}
    assert result.traces[0].status == NodeStatus.DONE
    assert result.traces[0].attempts == 2
    assert result.traces[0].error is None
    assert [attempt.status for attempt in result.traces[0].attempt_traces] == [
        NodeStatus.FAILED,
        NodeStatus.DONE,
    ]
    assert "temporary failure" in (result.traces[0].attempt_traces[0].error or "")
    assert result.traces[0].attempt_traces[1].error is None


def test_dag_executor_fails_after_retry_budget_is_exhausted():
    subtasks = [SubTask(id="task_1", question="first")]
    calls = 0

    def handler(subtask: SubTask) -> str:
        nonlocal calls
        calls += 1
        raise RuntimeError("still unavailable")

    result = DAGExecutor(max_retries=2).run(subtasks, handler)

    assert calls == 3
    assert result.results == {}
    assert result.traces[0].status == NodeStatus.FAILED
    assert result.traces[0].attempts == 3
    assert "still unavailable" in (result.traces[0].error or "")


def test_dag_executor_marks_slow_node_as_timeout():
    subtasks = [SubTask(id="task_1", question="first")]

    def handler(subtask: SubTask) -> str:
        time.sleep(0.01)
        return subtask.question

    result = DAGExecutor(timeout_seconds=0.001).run(subtasks, handler)

    assert result.results == {}
    assert subtasks[0].status == NodeStatus.TIMEOUT
    assert result.traces[0].status == NodeStatus.TIMEOUT
    assert result.traces[0].attempts == 1
    assert "exceeded timeout" in (result.traces[0].error or "")


def test_dag_executor_skips_nodes_after_timeout_dependency():
    subtasks = [
        SubTask(id="task_1", question="first"),
        SubTask(id="task_2", question="second", depends_on=["task_1"]),
    ]

    def handler(subtask: SubTask) -> str:
        if subtask.id == "task_1":
            time.sleep(0.01)
        return subtask.question

    result = DAGExecutor(timeout_seconds=0.001).run(subtasks, handler)

    assert result.results == {}
    assert [trace.status for trace in result.traces] == [NodeStatus.TIMEOUT, NodeStatus.SKIPPED]
    assert [trace.attempts for trace in result.traces] == [1, 0]
    assert "task_1" in (result.traces[1].error or "")


def test_async_dag_runs_independent_nodes_concurrently():
    subtasks = [
        SubTask(id="task_1", question="first"),
        SubTask(id="task_2", question="second"),
    ]

    async def handler(subtask: SubTask) -> str:
        await asyncio.sleep(0.03)
        return subtask.question

    started = time.perf_counter()
    result = asyncio.run(DAGExecutor(max_concurrency=2).run_async(subtasks, handler))
    elapsed = time.perf_counter() - started

    # 串行执行约需 0.06s；留出调度余量后仍应明显小于串行耗时。
    assert elapsed < 0.055
    assert result.started_order == ["task_1", "task_2"]
    assert set(result.completed_order) == {"task_1", "task_2"}


def test_async_dag_respects_semaphore_limit():
    subtasks = [SubTask(id=f"task_{index}", question=str(index)) for index in range(4)]
    active = 0
    peak_active = 0

    async def handler(subtask: SubTask) -> str:
        nonlocal active, peak_active
        active += 1
        peak_active = max(peak_active, active)
        await asyncio.sleep(0.01)
        active -= 1
        return subtask.id

    result = asyncio.run(DAGExecutor(max_concurrency=2).run_async(subtasks, handler))

    assert peak_active == 2
    assert len(result.results) == 4


def test_async_dag_cancels_timed_out_handler():
    subtasks = [SubTask(id="task_1", question="slow")]
    cancelled = False

    async def handler(subtask: SubTask) -> str:
        nonlocal cancelled
        try:
            await asyncio.sleep(1)
        except asyncio.CancelledError:
            cancelled = True
            raise
        return subtask.question

    result = asyncio.run(DAGExecutor(timeout_seconds=0.01).run_async(subtasks, handler))

    assert cancelled is True
    assert result.traces[0].status == NodeStatus.TIMEOUT
    assert result.traces[0].attempt_traces[0].status == NodeStatus.TIMEOUT


def test_async_dag_records_start_and_completion_order_separately():
    subtasks = [
        SubTask(id="task_1", question="slow"),
        SubTask(id="task_2", question="fast"),
    ]

    async def handler(subtask: SubTask) -> str:
        await asyncio.sleep(0.03 if subtask.id == "task_1" else 0.005)
        return subtask.id

    result = asyncio.run(DAGExecutor(max_concurrency=2).run_async(subtasks, handler))

    assert result.planned_order == ["task_1", "task_2"]
    assert result.started_order == ["task_1", "task_2"]
    assert result.completed_order == ["task_2", "task_1"]


def test_async_dag_resume_skips_seeded_done_node_and_passes_its_result():
    subtasks = [
        SubTask(id="task_1", question="already done"),
        SubTask(id="task_2", question="continue", depends_on=["task_1"]),
    ]
    first = DAGExecutor().run([SubTask(id="task_1", question="already done")], lambda item: "seed")
    calls: list[str] = []

    async def handler(subtask: SubTask, dependencies: dict[str, str]) -> str:
        calls.append(subtask.id)
        assert dependencies == {"task_1": "seed"}
        return "continued"

    result = asyncio.run(
        DAGExecutor().run_async(
            subtasks,
            handler,
            seed_results={"task_1": "seed"},
            seed_traces={"task_1": first.traces[0]},
        )
    )

    assert calls == ["task_2"]
    assert result.results == {"task_1": "seed", "task_2": "continued"}
    assert result.planned_order == ["task_1", "task_2"]


def test_dag_rejects_cycle_and_missing_dependency():
    subtasks = [
        SubTask(id="task_1", question="cycle one", depends_on=["task_2"]),
        SubTask(id="task_2", question="cycle two", depends_on=["task_1"]),
        SubTask(id="task_3", question="missing", depends_on=["unknown"]),
    ]

    result = DAGExecutor().run(subtasks, lambda subtask: subtask.id)

    assert result.results == {}
    assert all(trace.status == NodeStatus.FAILED for trace in result.traces)
    assert all(trace.attempts == 0 for trace in result.traces)
    assert all("Unresolved dependency or cycle" in (trace.error or "") for trace in result.traces)

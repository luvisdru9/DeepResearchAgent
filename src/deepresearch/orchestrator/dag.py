"""
异步 DAG 执行器。

负责依赖调度、并发限制、超时重试、失败传播和可复盘的执行顺序记录。
节点业务逻辑由调用方提供，执行器本身不依赖具体 Agent。
"""

from __future__ import annotations

import asyncio
from collections import Counter, deque
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
import inspect
import time
from typing import Generic, TypeVar, cast

from deepresearch.schema import NodeAttemptTrace, NodeStatus, NodeTrace, SubTask, utc_now

T = TypeVar("T")
DependencyResults = Mapping[str, T]
NodeHandler = Callable[..., T | Awaitable[T]]
NodeCompleteCallback = Callable[[SubTask, NodeTrace, T | None], None | Awaitable[None]]


@dataclass
class DAGExecutionResult(Generic[T]):
    """Node outputs, traces, and the three execution orders used by replay."""

    results: dict[str, T]
    traces: list[NodeTrace]
    planned_order: list[str]
    started_order: list[str]
    completed_order: list[str]


class DAGExecutor:
    """Execute ready DAG nodes concurrently while preserving dependency semantics."""

    def __init__(
        self,
        max_retries: int = 0,
        timeout_seconds: float | None = None,
        max_concurrency: int = 4,
    ) -> None:
        self.max_retries = max(0, max_retries)
        self.timeout_seconds = timeout_seconds if timeout_seconds and timeout_seconds > 0 else None
        self.max_concurrency = max(1, max_concurrency)

    def run(self, subtasks: list[SubTask], handler: NodeHandler[T]) -> DAGExecutionResult[T]:
        """Synchronous compatibility wrapper for CLI and existing callers."""

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.run_async(subtasks, handler))
        raise RuntimeError("DAGExecutor.run() cannot be used inside an event loop; await run_async() instead.")

    async def run_async(
        self,
        subtasks: list[SubTask],
        handler: NodeHandler[T],
        *,
        seed_results: Mapping[str, T] | None = None,
        seed_traces: Mapping[str, NodeTrace] | None = None,
        on_node_complete: NodeCompleteCallback[T] | None = None,
    ) -> DAGExecutionResult[T]:
        """Run a validated DAG with dependency-aware asyncio tasks."""

        planned_order = [subtask.id for subtask in subtasks]
        invalid_ids = self._find_unresolvable_nodes(subtasks)
        results: dict[str, T] = {}
        trace_by_id: dict[str, NodeTrace] = {}
        started_order: list[str] = []
        completed_order: list[str] = []
        semaphore = asyncio.Semaphore(self.max_concurrency)
        node_tasks: dict[str, asyncio.Task[None]] = {}

        # 恢复时只信任 DONE 且同时有结构化结果的节点；失败/运行中节点重新调度。
        supplied_results = dict(seed_results or {})
        supplied_traces = dict(seed_traces or {})
        for subtask in subtasks:
            trace = supplied_traces.get(subtask.id)
            if trace is None or trace.status != NodeStatus.DONE or subtask.id not in supplied_results:
                continue
            subtask.status = NodeStatus.DONE
            results[subtask.id] = supplied_results[subtask.id]
            trace_by_id[subtask.id] = trace
            if trace.started_at is not None:
                started_order.append(subtask.id)
            completed_order.append(subtask.id)

        # 无法解析的依赖和环在调度前失败，避免异步任务互相等待形成死锁。
        for subtask in subtasks:
            if subtask.id not in invalid_ids:
                continue
            subtask.status = NodeStatus.FAILED
            trace_by_id[subtask.id] = NodeTrace(
                subtask_id=subtask.id,
                question=subtask.question,
                depends_on=subtask.depends_on,
                status=NodeStatus.FAILED,
                attempts=0,
                error="Unresolved dependency or cycle in subtask graph.",
            )
            completed_order.append(subtask.id)
            await self._notify_node_complete(
                on_node_complete,
                subtask,
                trace_by_id[subtask.id],
                None,
            )

        async def execute_node(subtask: SubTask) -> None:
            # 等待所有有效上游结束，再判断本节点是否仍可执行。
            dependencies = [
                node_tasks[dependency]
                for dependency in subtask.depends_on
                if dependency in node_tasks
            ]
            if dependencies:
                await asyncio.gather(*dependencies)

            failed_dependencies = [
                dependency
                for dependency in subtask.depends_on
                if dependency not in trace_by_id or trace_by_id[dependency].status != NodeStatus.DONE
            ]
            if failed_dependencies:
                subtask.status = NodeStatus.SKIPPED
                trace_by_id[subtask.id] = NodeTrace(
                    subtask_id=subtask.id,
                    question=subtask.question,
                    depends_on=subtask.depends_on,
                    status=NodeStatus.SKIPPED,
                    attempts=0,
                    error=f"Skipped because dependencies failed: {', '.join(failed_dependencies)}.",
                )
                completed_order.append(subtask.id)
                await self._notify_node_complete(
                    on_node_complete,
                    subtask,
                    trace_by_id[subtask.id],
                    None,
                )
                return

            dependency_results = {
                dependency: results[dependency]
                for dependency in subtask.depends_on
            }
            async with semaphore:
                started_order.append(subtask.id)
                trace_by_id[subtask.id] = await self._run_node(
                    subtask,
                    handler,
                    results,
                    dependency_results,
                )
                completed_order.append(subtask.id)
                await self._notify_node_complete(
                    on_node_complete,
                    subtask,
                    trace_by_id[subtask.id],
                    results.get(subtask.id),
                )

        # 按计划顺序创建任务；依赖等待由每个节点协程内部完成。
        for subtask in subtasks:
            if subtask.id not in invalid_ids and subtask.id not in results:
                node_tasks[subtask.id] = asyncio.create_task(execute_node(subtask))
        if node_tasks:
            await asyncio.gather(*node_tasks.values())

        traces = [trace_by_id[subtask.id] for subtask in subtasks]
        return DAGExecutionResult(
            results=results,
            traces=traces,
            planned_order=planned_order,
            started_order=started_order,
            completed_order=completed_order,
        )

    @staticmethod
    async def _notify_node_complete(
        callback: NodeCompleteCallback[T] | None,
        subtask: SubTask,
        trace: NodeTrace,
        result: T | None,
    ) -> None:
        """Persist or observe a node without coupling the executor to SQLite."""

        if callback is None:
            return
        callback_result = callback(subtask, trace, result)
        if inspect.isawaitable(callback_result):
            await callback_result

    async def _run_node(
        self,
        subtask: SubTask,
        handler: NodeHandler[T],
        results: dict[str, T],
        dependency_results: dict[str, T],
    ) -> NodeTrace:
        """Execute one ready node and retain a trace for every attempt."""

        node_started_at = utc_now()
        node_started = time.perf_counter()
        max_attempts = 1 + self.max_retries
        attempt_traces: list[NodeAttemptTrace] = []
        final_error: str | None = None

        for attempt in range(1, max_attempts + 1):
            subtask.status = NodeStatus.RUNNING
            attempt_started_at = utc_now()
            attempt_started = time.perf_counter()
            try:
                value = await self._invoke_handler(
                    handler,
                    subtask,
                    dependency_results,
                )
                elapsed = time.perf_counter() - attempt_started
                if (
                    self.timeout_seconds is not None
                    and not inspect.iscoroutinefunction(handler)
                    and elapsed > self.timeout_seconds
                ):
                    # 同步兼容 handler 无法抢占，只能在返回后标记超时；异步 handler 可被 wait_for 取消。
                    raise asyncio.TimeoutError
                subtask.status = NodeStatus.DONE
                results[subtask.id] = value
                attempt_traces.append(
                    NodeAttemptTrace(
                        attempt=attempt,
                        status=NodeStatus.DONE,
                        started_at=attempt_started_at,
                        finished_at=utc_now(),
                        latency_seconds=elapsed,
                    )
                )
                final_error = None
                break
            except asyncio.TimeoutError:
                elapsed = time.perf_counter() - attempt_started
                final_error = f"Node execution exceeded timeout of {self.timeout_seconds:.3f}s."
                attempt_traces.append(
                    NodeAttemptTrace(
                        attempt=attempt,
                        status=NodeStatus.TIMEOUT,
                        started_at=attempt_started_at,
                        finished_at=utc_now(),
                        latency_seconds=elapsed,
                        error=final_error,
                    )
                )
                if attempt < max_attempts:
                    subtask.status = NodeStatus.RETRYING
                    continue
                subtask.status = NodeStatus.TIMEOUT
            except Exception as exc:  # noqa: BLE001 - replay must retain the original failure.
                elapsed = time.perf_counter() - attempt_started
                final_error = f"{type(exc).__name__}: {exc}"
                attempt_traces.append(
                    NodeAttemptTrace(
                        attempt=attempt,
                        status=NodeStatus.FAILED,
                        started_at=attempt_started_at,
                        finished_at=utc_now(),
                        latency_seconds=elapsed,
                        error=final_error,
                    )
                )
                if attempt < max_attempts:
                    subtask.status = NodeStatus.RETRYING
                    continue
                subtask.status = NodeStatus.FAILED

        return NodeTrace(
            subtask_id=subtask.id,
            question=subtask.question,
            depends_on=subtask.depends_on,
            status=subtask.status,
            attempts=len(attempt_traces),
            started_at=node_started_at,
            finished_at=utc_now(),
            latency_seconds=time.perf_counter() - node_started,
            error=final_error,
            attempt_traces=attempt_traces,
        )

    async def _invoke_handler(
        self,
        handler: NodeHandler[T],
        subtask: SubTask,
        dependency_results: dict[str, T],
    ) -> T:
        """Invoke one- or two-argument handlers without blocking independent DAG branches."""

        accepts_dependencies = self._accepts_dependency_results(handler)

        def call_handler() -> T | Awaitable[T]:
            if accepts_dependencies:
                return handler(subtask, dependency_results)
            return handler(subtask)

        if inspect.iscoroutinefunction(handler):
            awaitable = cast(Awaitable[T], call_handler())
            if self.timeout_seconds is None:
                result = await awaitable
            else:
                result = await asyncio.wait_for(awaitable, timeout=self.timeout_seconds)
        else:
            # 保留旧同步 handler 的确定性行为；需要真实并发的业务节点应提供 async handler。
            result = call_handler()

        # 兼容返回 awaitable 的普通 callable（例如包装器或 partial）。
        if inspect.isawaitable(result):
            return await cast(Awaitable[T], result)
        return cast(T, result)

    @staticmethod
    def _accepts_dependency_results(handler: NodeHandler[T]) -> bool:
        """Return whether a handler declares a second positional argument."""

        try:
            parameters = inspect.signature(handler).parameters.values()
        except (TypeError, ValueError):
            return False
        positional = [
            parameter
            for parameter in parameters
            if parameter.kind
            in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        return len(positional) >= 2 or any(
            parameter.kind == inspect.Parameter.VAR_POSITIONAL
            for parameter in parameters
        )

    @staticmethod
    def _find_unresolvable_nodes(subtasks: list[SubTask]) -> set[str]:
        """Return duplicate, missing-dependency, cyclic, and cycle-dependent node ids."""

        ids = [subtask.id for subtask in subtasks]
        duplicate_ids = {node_id for node_id, count in Counter(ids).items() if count > 1}
        known_ids = set(ids)
        invalid = set(duplicate_ids)
        invalid.update(
            subtask.id
            for subtask in subtasks
            if any(dependency not in known_ids for dependency in subtask.depends_on)
        )

        # 从合法根节点做 Kahn 拓扑遍历；未被访问的节点处于环中或依赖无效节点。
        candidates = [subtask for subtask in subtasks if subtask.id not in invalid]
        indegree = {
            subtask.id: sum(dependency not in invalid for dependency in subtask.depends_on)
            for subtask in candidates
        }
        children: dict[str, list[str]] = {subtask.id: [] for subtask in candidates}
        for subtask in candidates:
            for dependency in subtask.depends_on:
                if dependency in children:
                    children[dependency].append(subtask.id)

        queue = deque(node_id for node_id, degree in indegree.items() if degree == 0)
        resolved: set[str] = set()
        while queue:
            node_id = queue.popleft()
            resolved.add(node_id)
            for child in children[node_id]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    queue.append(child)

        invalid.update(set(indegree) - resolved)
        return invalid

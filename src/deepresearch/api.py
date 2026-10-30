"""Optional FastAPI delivery layer around the persisted research Pipeline.

The API deliberately reuses ResearchTask, ResearchPipeline, SQLite state, and Replay.
It does not introduce a second orchestration path. The in-process background runner is
appropriate for the local MVP; a durable external worker can replace it later without
changing the HTTP response schemas.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, status
from pydantic import BaseModel, Field

from deepresearch.config import AppConfig, load_config
from deepresearch.memory import SQLiteMemoryStore
from deepresearch.orchestrator import ReplayStore, ResearchPipeline
from deepresearch.schema import ResearchTask
from deepresearch.tools import ApprovalRequest, ApprovalStatus


class ResearchTaskRequest(BaseModel):
    """Public task input; provider credentials and model profiles stay server-side."""

    query: str = Field(min_length=1)
    domain: str | None = None
    language: str | None = None
    time_range: str | None = None
    report_length: str | None = None
    require_citations: bool = True
    max_budget_usd: float | None = Field(default=None, ge=0)
    max_llm_calls: int | None = Field(default=None, ge=0)
    max_total_tokens: int | None = Field(default=None, ge=0)
    max_runtime_seconds: float | None = Field(default=None, gt=0)


class ResearchTaskAccepted(BaseModel):
    """Immediate response returned before the background Pipeline finishes."""

    run_id: str
    status: str = "RUNNING"


class ResearchTaskStatus(BaseModel):
    """Compact persisted status view for polling clients."""

    run_id: str
    query: str
    status: str
    stop_reason: str | None = None
    current_round: int = 0
    error: str | None = None
    created_at: str
    updated_at: str


class ApprovalDecisionRequest(BaseModel):
    """Local-MVP operator decision; idempotency_key prevents duplicate writes."""

    decision: str = Field(pattern=r"^(approve|reject)$")
    idempotency_key: str = Field(min_length=1, max_length=200)
    reason: str | None = Field(default=None, max_length=1000)


TaskRunner = Callable[[ResearchTask], None]
ResumeRunner = Callable[[str], None]


def create_app(
    config: AppConfig | None = None,
    *,
    config_path: str | Path = "configs/default.yaml",
    task_runner: TaskRunner | None = None,
    resume_runner: ResumeRunner | None = None,
) -> FastAPI:
    """Create the optional HTTP app while keeping CLI and tests dependency-injectable."""

    resolved_config = config or load_config(config_path)
    if not resolved_config.memory.enabled:
        raise ValueError("FastAPI task polling requires memory.enabled=true.")

    database_path = resolved_config.memory.sqlite_path or (
        resolved_config.runtime.run_dir / "deepresearch.sqlite3"
    )
    store = SQLiteMemoryStore(database_path)
    replay_store = ReplayStore(resolved_config.runtime.run_dir)

    def default_runner(task: ResearchTask) -> None:
        # 每个后台任务使用独立 Pipeline；SQLite 每次操作也使用短连接，避免共享可变 Agent 状态。
        ResearchPipeline(resolved_config.model_copy(deep=True)).run(task)

    def default_resume_runner(run_id: str) -> None:
        ResearchPipeline(resolved_config.model_copy(deep=True)).resume(run_id)

    runner = task_runner or default_runner
    run_resumer = resume_runner or default_resume_runner

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # 单进程本地服务启动时，旧 RUNNING 行不可能仍由本进程管理，应转为可恢复状态。
        store.interrupt_running_runs("API process restarted before the task completed.")
        app.state.accepting_tasks = True
        try:
            yield
        finally:
            app.state.accepting_tasks = False
            jobs_by_run = dict(app.state.research_jobs)
            if jobs_by_run:
                _, pending = await asyncio.wait(
                    list(jobs_by_run.values()),
                    timeout=resolved_config.runtime.api_shutdown_grace_seconds,
                )
                for run_id, job in jobs_by_run.items():
                    if job in pending:
                        store.interrupt_run(
                            run_id,
                            "API shutdown grace period expired before task completion.",
                        )
                        job.cancel()

    app = FastAPI(
        title="DeepResearchAgent API",
        version="0.1.0",
        description="Submit persisted deep-research tasks and inspect reports or replay traces.",
        lifespan=lifespan,
    )
    # run_id 映射既保留强引用，也阻止同一个持久化任务被重复恢复。
    app.state.research_jobs = {}
    app.state.accepting_tasks = True

    async def execute_task(task: ResearchTask, *, resume: bool = False) -> None:
        try:
            if resume:
                if resume_runner is None:
                    # 真实同步 Pipeline 在线程中运行，避免阻塞 ASGI 事件循环。
                    await asyncio.to_thread(run_resumer, task.run_id)
                else:
                    # 注入 runner 只用于快速、无外部副作用的接口测试。
                    run_resumer(task.run_id)
            else:
                if task_runner is None:
                    await asyncio.to_thread(runner, task)
                else:
                    task_runner(task)
        except Exception as exc:
            # Pipeline 内部通常已经保存失败状态；这里覆盖初始化阶段失败且不泄露凭据。
            store.fail_run(task.run_id, f"{type(exc).__name__}: {exc}")

    def schedule(task: ResearchTask, *, resume: bool = False) -> None:
        if task.run_id in app.state.research_jobs:
            raise HTTPException(status_code=409, detail="Research task is already running.")
        job = asyncio.create_task(execute_task(task, resume=resume))
        app.state.research_jobs[task.run_id] = job
        job.add_done_callback(
            lambda _job, run_id=task.run_id: app.state.research_jobs.pop(run_id, None)
        )

    @app.post(
        "/research/tasks",
        response_model=ResearchTaskAccepted,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def submit_task(
        request: ResearchTaskRequest,
    ) -> ResearchTaskAccepted:
        if not app.state.accepting_tasks:
            raise HTTPException(status_code=503, detail="Research service is shutting down.")
        task = ResearchTask(
            query=request.query,
            domain=request.domain,
            language=request.language or resolved_config.runtime.language,
            time_range=request.time_range,
            report_length=request.report_length or resolved_config.runtime.report_length,
            require_citations=request.require_citations,
            max_budget_usd=request.max_budget_usd,
            max_llm_calls=request.max_llm_calls,
            max_total_tokens=request.max_total_tokens,
            max_runtime_seconds=request.max_runtime_seconds,
        )
        # 先写 RUNNING 行，保证客户端拿到 run_id 后立即可以轮询。
        store.start_run(task)
        schedule(task)
        return ResearchTaskAccepted(run_id=task.run_id)

    def status_view(row: dict[str, Any]) -> ResearchTaskStatus:
        run_id = str(row["run_id"])
        state = store.load_research_state(run_id)
        return ResearchTaskStatus(
            run_id=run_id,
            query=str(row["query"]),
            status=str(row["status"]),
            stop_reason=state.stop_reason.value if state and state.stop_reason else None,
            current_round=state.current_round if state else 0,
            error=str(row["error"]) if row.get("error") else None,
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )

    @app.get("/research/tasks", response_model=list[ResearchTaskStatus])
    async def list_tasks(
        task_status: str | None = Query(default=None, alias="status"),
        limit: int = 100,
    ) -> list[ResearchTaskStatus]:
        if not 1 <= limit <= 500:
            raise HTTPException(status_code=422, detail="limit must be between 1 and 500.")
        return [status_view(row) for row in store.list_runs(status=task_status, limit=limit)]

    @app.get("/research/tasks/{run_id}", response_model=ResearchTaskStatus)
    async def get_task_status(run_id: str) -> ResearchTaskStatus:
        row = store.get_run(run_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Research task not found.")
        return status_view(row)

    @app.post(
        "/research/tasks/{run_id}/resume",
        response_model=ResearchTaskAccepted,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def resume_task(run_id: str) -> ResearchTaskAccepted:
        if not app.state.accepting_tasks:
            raise HTTPException(status_code=503, detail="Research service is shutting down.")
        row = store.get_run(run_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Research task not found.")
        if row["status"] == "DONE":
            raise HTTPException(status_code=409, detail="Research task is already complete.")
        if row["status"] == "RUNNING" or run_id in app.state.research_jobs:
            raise HTTPException(status_code=409, detail="Research task is already running.")
        if row["status"] not in {"FAILED", "INTERRUPTED"}:
            raise HTTPException(status_code=409, detail="Research task cannot be resumed.")
        task = store.load_task(run_id)
        if task is None:
            raise HTTPException(status_code=404, detail="Original task payload not found.")
        store.start_run(task)
        schedule(task, resume=True)
        return ResearchTaskAccepted(run_id=run_id)

    @app.get("/approvals", response_model=list[ApprovalRequest])
    async def list_approvals(
        run_id: str | None = None,
        approval_status: ApprovalStatus | None = Query(default=None, alias="status"),
        limit: int = 100,
    ) -> list[ApprovalRequest]:
        """List local approval metadata without exposing raw tool arguments."""

        if not 1 <= limit <= 500:
            raise HTTPException(status_code=422, detail="limit must be between 1 and 500.")
        return store.list_approvals(
            run_id=run_id,
            status=approval_status,
            limit=limit,
        )

    @app.post("/approvals/{approval_id}/decision", response_model=ApprovalRequest)
    async def decide_approval(
        approval_id: str,
        request: ApprovalDecisionRequest,
    ) -> ApprovalRequest:
        """Apply one idempotent local decision; this MVP has no remote authentication."""

        decision = (
            ApprovalStatus.APPROVED
            if request.decision == "approve"
            else ApprovalStatus.REJECTED
        )
        try:
            return store.decide_approval(
                approval_id,
                decision=decision,
                decision_key=request.idempotency_key,
                decided_by="local_api",
                reason=request.reason,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Approval request not found.") from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    def completed_replay(run_id: str) -> dict[str, Any]:
        row = store.get_run(run_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Research task not found.")
        if row["status"] != "DONE":
            raise HTTPException(status_code=409, detail="Research task is not complete.")
        try:
            return replay_store.read(run_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Replay artifact not found.") from exc

    @app.get("/research/tasks/{run_id}/report")
    async def get_task_report(run_id: str) -> dict[str, Any]:
        payload = completed_replay(run_id)
        report = payload.get("report")
        if not isinstance(report, dict):
            raise HTTPException(status_code=404, detail="Report artifact not found.")
        # 公开报告接口不返回内部 RedTeam/Verifier 诊断；完整信息仍可从 trace 接口查看。
        return {key: value for key, value in report.items() if key != "redteam_findings"}

    @app.get("/research/tasks/{run_id}/trace")
    async def get_task_trace(run_id: str) -> dict[str, Any]:
        return completed_replay(run_id)

    return app

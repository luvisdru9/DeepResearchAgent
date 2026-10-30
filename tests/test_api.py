from __future__ import annotations

import asyncio

import httpx

from deepresearch.api import create_app
from deepresearch.config import AppConfig, MemoryConfig, RuntimeConfig
from deepresearch.memory import SQLiteMemoryStore
from deepresearch.orchestrator import ReplayStore
from deepresearch.schema import ResearchReport, ResearchTask
from deepresearch.tools import ApprovalStatus


def _config(tmp_path) -> AppConfig:
    return AppConfig(
        runtime=RuntimeConfig(run_dir=tmp_path),
        memory=MemoryConfig(enabled=True, sqlite_path=tmp_path / "state.sqlite3"),
    )


def test_api_submits_background_task_and_exposes_status(tmp_path):
    config = _config(tmp_path)
    seen: list[ResearchTask] = []

    def runner(task: ResearchTask) -> None:
        seen.append(task)
        store = SQLiteMemoryStore(config.memory.sqlite_path)
        store.finish_run(task.run_id, replay_path=tmp_path / "unused.json")

    app = create_app(config, task_runner=runner)

    async def request_sequence():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/research/tasks",
                json={"query": "如何评价引用质量？", "max_llm_calls": 5},
            )
            run_id = response.json()["run_id"]
            for _ in range(100):
                if not app.state.research_jobs:
                    break
                await asyncio.sleep(0.01)
            status_response = await client.get(f"/research/tasks/{run_id}")
            return response, status_response

    response, status_response = asyncio.run(request_sequence())

    assert response.status_code == 202
    run_id = response.json()["run_id"]
    assert seen[0].run_id == run_id
    assert seen[0].max_llm_calls == 5
    assert status_response.status_code == 200
    assert status_response.json()["status"] == "DONE"


def test_api_returns_completed_report_and_trace(tmp_path):
    config = _config(tmp_path)
    task = ResearchTask(query="测试 API", run_id="api_complete")
    report = ResearchReport(
        run_id=task.run_id,
        query=task.query,
        title="API 报告",
        summary="摘要",
        key_findings=[],
        evidence=[],
        markdown="# API 报告",
    )
    store = SQLiteMemoryStore(config.memory.sqlite_path)
    store.start_run(task)
    replay_path = ReplayStore(config.runtime.run_dir).write(
        task.run_id,
        {"task": task, "report": report},
    )
    store.finish_run(task.run_id, replay_path=replay_path)
    app = create_app(config, task_runner=lambda task: None)

    async def request_sequence():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            report_response = await client.get(f"/research/tasks/{task.run_id}/report")
            trace_response = await client.get(f"/research/tasks/{task.run_id}/trace")
            return report_response, trace_response

    report_response, trace_response = asyncio.run(request_sequence())

    assert report_response.status_code == 200
    assert report_response.json()["title"] == "API 报告"
    assert "redteam_findings" not in report_response.json()
    assert trace_response.status_code == 200
    assert trace_response.json()["schema_version"] == "2.0"


def test_api_rejects_missing_or_incomplete_artifacts(tmp_path):
    config = _config(tmp_path)
    task = ResearchTask(query="仍在运行", run_id="api_running")
    SQLiteMemoryStore(config.memory.sqlite_path).start_run(task)
    app = create_app(config, task_runner=lambda task: None)

    async def request_sequence():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            missing = await client.get("/research/tasks/missing")
            incomplete = await client.get(f"/research/tasks/{task.run_id}/report")
            return missing, incomplete

    missing, incomplete = asyncio.run(request_sequence())

    assert missing.status_code == 404
    assert incomplete.status_code == 409


def test_api_startup_marks_orphaned_running_task_as_interrupted(tmp_path):
    config = _config(tmp_path)
    task = ResearchTask(query="进程重启恢复", run_id="api_orphaned")
    store = SQLiteMemoryStore(config.memory.sqlite_path)
    store.start_run(task)
    app = create_app(config, task_runner=lambda task: None)

    async def request_sequence():
        # ASGITransport 不自动执行 lifespan；显式进入以验证真实服务启动语义。
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://test",
            ) as client:
                return await client.get(f"/research/tasks/{task.run_id}")

    response = asyncio.run(request_sequence())

    assert response.status_code == 200
    assert response.json()["status"] == "INTERRUPTED"
    assert "restarted" in response.json()["error"]


def test_api_lists_and_resumes_interrupted_task_once(tmp_path):
    config = _config(tmp_path)
    task = ResearchTask(query="恢复 API 任务", run_id="api_resume")
    store = SQLiteMemoryStore(config.memory.sqlite_path)
    store.start_run(task)
    store.interrupt_run(task.run_id, "test interruption")
    resumed: list[str] = []

    def resume_runner(run_id: str) -> None:
        resumed.append(run_id)
        replay_path = ReplayStore(config.runtime.run_dir).write(
            run_id,
            {"task": task, "report": {"title": "恢复完成"}},
        )
        SQLiteMemoryStore(config.memory.sqlite_path).finish_run(
            run_id,
            replay_path=replay_path,
        )

    app = create_app(
        config,
        task_runner=lambda task: None,
        resume_runner=resume_runner,
    )

    async def request_sequence():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            listed_before = await client.get(
                "/research/tasks",
                params={"status": "INTERRUPTED", "limit": 10},
            )
            accepted = await client.post(f"/research/tasks/{task.run_id}/resume")
            for _ in range(100):
                if not app.state.research_jobs:
                    break
                await asyncio.sleep(0.01)
            listed_after = await client.get("/research/tasks", params={"status": "DONE"})
            duplicate = await client.post(f"/research/tasks/{task.run_id}/resume")
            return listed_before, accepted, listed_after, duplicate

    listed_before, accepted, listed_after, duplicate = asyncio.run(request_sequence())

    assert [item["run_id"] for item in listed_before.json()] == [task.run_id]
    assert accepted.status_code == 202
    assert resumed == [task.run_id]
    assert [item["run_id"] for item in listed_after.json()] == [task.run_id]
    assert duplicate.status_code == 409


def test_api_shutdown_marks_jobs_past_grace_period_as_interrupted(tmp_path):
    config = _config(tmp_path)
    config.runtime.api_shutdown_grace_seconds = 0
    task = ResearchTask(query="关闭时中断", run_id="api_shutdown")
    store = SQLiteMemoryStore(config.memory.sqlite_path)
    app = create_app(config, task_runner=lambda task: None)

    async def run_lifespan():
        async with app.router.lifespan_context(app):
            store.start_run(task)
            sleeper = asyncio.create_task(asyncio.sleep(60))
            app.state.research_jobs[task.run_id] = sleeper

    asyncio.run(run_lifespan())

    row = store.get_run(task.run_id)
    assert row is not None
    assert row["status"] == "INTERRUPTED"
    assert "grace period expired" in row["error"]


def test_api_lists_and_idempotently_decides_local_approval(tmp_path):
    config = _config(tmp_path)
    store = SQLiteMemoryStore(config.memory.sqlite_path)
    approval = store.request_approval(
        run_id="approval_run",
        agent="writer",
        tool_name="publish_report",
        action_signature="approval_action",
        argument_keys=["report_id"],
        risk_level="high_risk",
        reason="tool_requires_operator_approval",
    )
    app = create_app(config, task_runner=lambda task: None)

    async def request_sequence():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            listed = await client.get("/approvals", params={"run_id": "approval_run"})
            decided = await client.post(
                f"/approvals/{approval.approval_id}/decision",
                json={"decision": "approve", "idempotency_key": "decision_1"},
            )
            repeated = await client.post(
                f"/approvals/{approval.approval_id}/decision",
                json={"decision": "approve", "idempotency_key": "decision_1"},
            )
            conflict = await client.post(
                f"/approvals/{approval.approval_id}/decision",
                json={"decision": "reject", "idempotency_key": "decision_2"},
            )
            missing = await client.post(
                "/approvals/missing/decision",
                json={"decision": "approve", "idempotency_key": "decision_3"},
            )
            return listed, decided, repeated, conflict, missing

    listed, decided, repeated, conflict, missing = asyncio.run(request_sequence())

    assert listed.status_code == 200
    assert listed.json()[0]["argument_keys"] == ["report_id"]
    assert decided.status_code == 200
    assert decided.json()["status"] == ApprovalStatus.APPROVED.value
    assert repeated.json() == decided.json()
    assert conflict.status_code == 409
    assert missing.status_code == 404

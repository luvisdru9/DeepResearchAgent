"""
命令行入口。支持：
deepresearch run --query "..."
deepresearch replay --run-id ...
deepresearch report --run-id ...
deepresearch resume --run-id ...
deepresearch eval
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from deepresearch.config import load_config
from deepresearch.demos import run_failure_demo
from deepresearch.evaluation import (
    AutoDraftBuilder,
    CompressionComparisonRunner,
    EvaluationRunner,
    ResearchBench40Builder,
    RetrievalComparisonRunner,
    SystemComparisonRunner,
    export_comparison_examples,
    export_run_examples,
)
from deepresearch.orchestrator import ReplayStore, ResearchPipeline
from deepresearch.memory import SQLiteMemoryStore
from deepresearch.retrieval import run_retrieval_smoke
from deepresearch.schema import ResearchTask
from deepresearch.tools import ApprovalStatus


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI parser and subcommands."""

    parser = argparse.ArgumentParser(prog="deepresearch")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument(
        "--llm-profile",
        default=None,
        help="Temporarily override the LLM profile for every LLM-backed Agent.",
    )
    parser.add_argument(
        "--planner-llm-profile",
        default=None,
        help="Override only planning.llm_profile.",
    )
    parser.add_argument(
        "--reader-llm-profile",
        default=None,
        help="Override only reader.llm_profile.",
    )
    parser.add_argument(
        "--synthesizer-llm-profile",
        default=None,
        help="Override only synthesis.llm_profile.",
    )
    parser.add_argument(
        "--redteam-llm-profile",
        default=None,
        help="Override only redteam.llm_profile.",
    )
    parser.add_argument(
        "--evaluator-llm-profile",
        default=None,
        help="Override only evaluation.llm_profile.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run a deep research task.")
    run_parser.add_argument("--query", required=True)
    run_parser.add_argument("--domain")
    run_parser.add_argument("--language", default=None)
    run_parser.add_argument("--max-budget-usd", type=float, default=None)
    run_parser.add_argument("--max-llm-calls", type=int, default=None)
    run_parser.add_argument("--max-total-tokens", type=int, default=None)
    run_parser.add_argument("--max-runtime-seconds", type=float, default=None)
    # 这些覆盖仅作用于本次运行，便于真实回归在不复制或改写主 YAML 的前提下固定实验变量。
    run_parser.add_argument(
        "--retrieval-provider",
        choices=["local", "tavily", "hybrid"],
        default=None,
        help="Override retrieval.provider for this run only.",
    )
    run_parser.add_argument(
        "--retrieval-top-k",
        type=int,
        default=None,
        help="Override retrieval.top_k for this run only.",
    )
    run_parser.add_argument(
        "--enable-research-loop",
        action="store_true",
        help="Enable bounded evidence-gap follow-up research for this run.",
    )
    run_parser.add_argument(
        "--max-research-rounds",
        type=int,
        default=None,
        help="Override research_loop.max_rounds; use with --enable-research-loop.",
    )
    run_parser.add_argument(
        "--disable-memory-retrieval",
        action="store_true",
        help="Do not add historical Evidence to retrieval for this run.",
    )

    replay_parser = subparsers.add_parser("replay", help="Print replay JSON for a run.")
    replay_parser.add_argument("--run-id", required=True)
    replay_parser.add_argument(
        "--summary",
        action="store_true",
        help="Print a compact research-state and DAG summary instead of full JSON.",
    )

    resume_parser = subparsers.add_parser(
        "resume",
        help="Resume an interrupted SQLite-backed run without repeating completed DAG nodes.",
    )
    resume_parser.add_argument("--run-id", required=True)

    report_parser = subparsers.add_parser("report", help="Print Markdown report for a run.")
    report_parser.add_argument("--run-id", required=True)

    eval_parser = subparsers.add_parser("eval", help="Run ResearchBench smoke evaluation.")
    eval_parser.add_argument("--bench", default=None)

    retrieval_parser = subparsers.add_parser(
        "retrieval-smoke",
        help="Validate one retrieval provider call without invoking any LLM.",
    )
    retrieval_parser.add_argument("--query", required=True)
    retrieval_parser.add_argument("--provider", choices=["local", "tavily", "hybrid"], default=None)
    retrieval_parser.add_argument("--top-k", type=int, default=None)

    compare_parser = subparsers.add_parser(
        "compare-retrieval",
        help="Run one benchmark across multiple retrieval providers.",
    )
    compare_parser.add_argument(
        "--providers",
        default="local,tavily",
        help="Comma-separated retrieval providers, for example local,tavily.",
    )
    compare_parser.add_argument("--bench", default=None)
    compression_parser = subparsers.add_parser(
        "compare-compression",
        help="Run compression-off/on ablation and persist quality, token, and latency metrics.",
    )
    compression_parser.add_argument("--bench", default=None)
    compression_parser.add_argument(
        "--provider", choices=["local", "tavily", "hybrid"], default=None
    )
    compression_parser.add_argument("--max-evidence", type=int, default=None)
    system_parser = subparsers.add_parser(
        "compare-system",
        help="Run the PLAN.md five-strategy system ablation matrix.",
    )
    system_parser.add_argument("--bench", default=None)
    system_parser.add_argument(
        "--strategies",
        default=None,
        help="Optional comma-separated subset of system strategies.",
    )
    system_parser.add_argument(
        "--judge-strategies",
        default=None,
        help=(
            "Comma-separated strategies that receive semantic Judges; other "
            "strategies still receive deterministic rule/process metrics."
        ),
    )
    system_parser.add_argument(
        "--memory-seed-mode",
        choices=["current", "empty"],
        default="current",
        help="Freeze the current SQLite store or a clean empty store for memory variants.",
    )
    system_parser.add_argument("--max-llm-calls", type=int, default=None)
    system_parser.add_argument("--max-total-tokens", type=int, default=None)
    system_parser.add_argument("--max-runtime-seconds", type=float, default=None)
    system_parser.add_argument(
        "--sample-ids",
        default=None,
        help="Optional comma-separated benchmark sample IDs, preserving the requested order.",
    )
    draft_parser = subparsers.add_parser(
        "build-auto-draft",
        help="Build the fixed eight-sample Track A engineering Benchmark draft.",
    )
    draft_parser.add_argument("--source-bench", default="data/researchbench_full.json")
    draft_parser.add_argument(
        "--output-root",
        default="data/benchmarks/researchbench_v2/track_a_auto_draft",
    )
    draft_parser.add_argument(
        "--sample-ids",
        default=None,
        help="Optional comma-separated fixed subset for staged smoke tests.",
    )
    draft_parser.add_argument("--max-llm-calls", type=int, default=12)
    draft_parser.add_argument("--max-total-tokens", type=int, default=40000)
    draft_parser.add_argument("--max-runtime-seconds", type=float, default=300.0)
    researchbench40_parser = subparsers.add_parser(
        "build-researchbench40",
        help="Build or resume the fixed forty-sample engineering ResearchBench draft.",
    )
    researchbench40_parser.add_argument(
        "--manifest",
        default=(
            "data/benchmarks/researchbench_v2/"
            "researchbench_40_v1/candidates.json"
        ),
    )
    researchbench40_parser.add_argument(
        "--output-dir",
        default=None,
        help="Defaults to the candidate manifest directory.",
    )
    researchbench40_parser.add_argument(
        "--sample-ids",
        default=None,
        help="Optional comma-separated fixed subset for smoke tests or staged batches.",
    )
    researchbench40_parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Rebuild requested samples instead of validating and reusing passed artifacts.",
    )
    researchbench40_parser.set_defaults(resume=True)
    researchbench40_parser.add_argument("--max-llm-calls", type=int, default=12)
    researchbench40_parser.add_argument("--max-total-tokens", type=int, default=40000)
    researchbench40_parser.add_argument("--max-runtime-seconds", type=float, default=300.0)
    export_parser = subparsers.add_parser(
        "export-examples",
        help="Export complete replay/report examples from a comparison artifact.",
    )
    export_parser.add_argument("--comparison", required=True)
    export_parser.add_argument("--strategy", default="dag_memory_redteam_compress")
    export_parser.add_argument("--output-dir", default="examples/generated")
    export_parser.add_argument("--limit", type=int, default=3)
    runs_export_parser = subparsers.add_parser(
        "export-runs",
        help="Export explicitly selected real runs as review candidates.",
    )
    runs_export_parser.add_argument("--run-ids", required=True)
    runs_export_parser.add_argument("--output-dir", default="examples/real")
    subparsers.add_parser(
        "demo-failures",
        help="Run four deterministic failure-defense demos without API keys or network access.",
    )
    approval_list_parser = subparsers.add_parser(
        "approval-list",
        help="List compact local HITL approval records from SQLite.",
    )
    approval_list_parser.add_argument("--run-id", default=None)
    approval_list_parser.add_argument(
        "--status",
        choices=[item.value for item in ApprovalStatus],
        default=None,
    )
    approval_list_parser.add_argument("--limit", type=int, default=100)
    approval_decide_parser = subparsers.add_parser(
        "approval-decide",
        help="Approve or reject one local HITL request idempotently.",
    )
    approval_decide_parser.add_argument("--approval-id", required=True)
    approval_decide_parser.add_argument(
        "--decision",
        choices=["approve", "reject"],
        required=True,
    )
    approval_decide_parser.add_argument("--decision-key", required=True)
    approval_decide_parser.add_argument("--reason", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint used by both python -m and the installed console script."""

    parser = build_parser()
    args = parser.parse_args(argv)
    # 加载参数
    config = load_config(args.config)
    if args.llm_profile:
        # 全局覆盖会应用到所有已启用的 LLM Agent，但不改写 YAML。
        config.llm.active_profile = args.llm_profile
        config.planning.llm_profile = args.llm_profile
        config.reader.llm_profile = args.llm_profile
        config.synthesis.llm_profile = args.llm_profile
        config.redteam.llm_profile = args.llm_profile
        config.evaluation.llm_profile = args.llm_profile
        config.evaluation.judge_profiles = [args.llm_profile]
    if args.planner_llm_profile:
        config.planning.llm_profile = args.planner_llm_profile
    if args.reader_llm_profile:
        config.reader.llm_profile = args.reader_llm_profile
    if args.synthesizer_llm_profile:
        config.synthesis.llm_profile = args.synthesizer_llm_profile
    if args.redteam_llm_profile:
        config.redteam.llm_profile = args.redteam_llm_profile
    if args.evaluator_llm_profile:
        config.evaluation.llm_profile = args.evaluator_llm_profile
        # 显式 CLI 覆盖优先于 YAML 中的多 Judge profile 列表。
        config.evaluation.judge_profiles = [args.evaluator_llm_profile]

    if args.command == "run":
        # 真实回归需要隔离少量运行策略；覆盖保留在内存中，不改写主配置文件。
        if args.retrieval_provider:
            config.retrieval.provider = args.retrieval_provider
        if args.retrieval_top_k is not None:
            if args.retrieval_top_k < 1:
                parser.error("--retrieval-top-k must be at least 1")
            config.retrieval.top_k = args.retrieval_top_k
        if args.max_research_rounds is not None:
            if not 1 <= args.max_research_rounds <= 10:
                parser.error("--max-research-rounds must be between 1 and 10")
            if not args.enable_research_loop:
                parser.error(
                    "--max-research-rounds requires --enable-research-loop"
                )
            config.research_loop.max_rounds = args.max_research_rounds
        if args.enable_research_loop:
            config.research_loop.enabled = True
        if args.disable_memory_retrieval:
            config.memory.retrieval_enabled = False

        # 预算属于 ResearchTask；Agent/profile 与其余运行策略仍来自 YAML。
        task = ResearchTask(
            query=args.query,
            domain=args.domain,
            language=args.language or config.runtime.language,
            report_length=config.runtime.report_length,
            max_budget_usd=args.max_budget_usd,
            max_llm_calls=args.max_llm_calls,
            max_total_tokens=args.max_total_tokens,
            max_runtime_seconds=args.max_runtime_seconds,
        )
        report = ResearchPipeline(config).run(task)
        print(report.markdown)
        print(f"Run ID: {report.run_id}")
        return 0

    if args.command == "replay":
        # replay 输出完整 JSON，便于调试 DAG 节点、证据和指标。
        payload = ReplayStore(config.runtime.run_dir).read(args.run_id)
        output = build_replay_summary(payload) if args.summary else payload
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return 0

    if args.command == "resume":
        report = ResearchPipeline(config).resume(args.run_id)
        print(report.markdown)
        print(f"Run ID: {report.run_id}")
        return 0

    if args.command == "report":
        # report 子命令只打印最终 Markdown，适合重定向到文件或人工阅读。
        payload = ReplayStore(config.runtime.run_dir).read(args.run_id)
        print(payload["report"]["markdown"])
        return 0

    if args.command == "eval":
        # eval 使用相同 pipeline 跑 smoke benchmark，输出机器可读 JSON 指标。
        runner = EvaluationRunner(config)
        results = runner.run(Path(args.bench) if args.bench else None)
        print(
            json.dumps(
                {
                    "artifact_path": str(runner.last_output_path)
                    if runner.last_output_path
                    else None,
                    "results": [item.model_dump(mode="json") for item in results],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.command == "retrieval-smoke":
        # 命令行覆盖只作用于本次 smoke，不改写 YAML；输出不包含正文和凭据。
        if args.provider:
            config.retrieval.provider = args.provider
        if args.top_k is not None:
            if args.top_k < 1:
                parser.error("--top-k must be at least 1")
            config.retrieval.top_k = args.top_k
        payload = asyncio.run(run_retrieval_smoke(config.retrieval, args.query))
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.command == "compare-retrieval":
        providers = [item.strip() for item in args.providers.split(",") if item.strip()]
        if not providers:
            parser.error("--providers must contain at least one provider")
        runner = RetrievalComparisonRunner(config)
        results = runner.run(
            providers,
            Path(args.bench) if args.bench else None,
        )
        print(
            json.dumps(
                {
                    "artifact_path": str(runner.last_output_path) if runner.last_output_path else None,
                    "results": [item.model_dump(mode="json") for item in results],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.command == "compare-compression":
        if args.max_evidence is not None and args.max_evidence < 1:
            parser.error("--max-evidence must be at least 1")
        runner = CompressionComparisonRunner(config)
        results = runner.run(
            Path(args.bench) if args.bench else None,
            provider=args.provider,
            compressed_max_evidence=args.max_evidence,
        )
        print(
            json.dumps(
                {
                    "artifact_path": str(runner.last_output_path) if runner.last_output_path else None,
                    "results": [item.model_dump(mode="json") for item in results],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.command == "compare-system":
        strategies = (
            [item.strip() for item in args.strategies.split(",") if item.strip()]
            if args.strategies
            else None
        )
        judge_strategies = (
            [item.strip() for item in args.judge_strategies.split(",") if item.strip()]
            if args.judge_strategies
            else None
        )
        sample_ids = (
            [item.strip() for item in args.sample_ids.split(",") if item.strip()]
            if args.sample_ids
            else None
        )
        if args.max_llm_calls is not None and args.max_llm_calls < 0:
            parser.error("--max-llm-calls must be non-negative")
        if args.max_total_tokens is not None and args.max_total_tokens < 0:
            parser.error("--max-total-tokens must be non-negative")
        if args.max_runtime_seconds is not None and args.max_runtime_seconds <= 0:
            parser.error("--max-runtime-seconds must be positive")
        runner = SystemComparisonRunner(config)
        try:
            results = runner.run(
                Path(args.bench) if args.bench else None,
                strategies=strategies,
                judge_strategies=judge_strategies,
                memory_seed_mode=args.memory_seed_mode,
                max_llm_calls=args.max_llm_calls,
                max_total_tokens=args.max_total_tokens,
                max_runtime_seconds=args.max_runtime_seconds,
                sample_ids=sample_ids,
            )
        except ValueError as exc:
            parser.error(str(exc))
        print(
            json.dumps(
                {
                    "artifact_path": str(runner.last_output_path),
                    "markdown_path": str(runner.last_markdown_path),
                    "results": [item.model_dump(mode="json") for item in results],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.command == "build-auto-draft":
        sample_ids = (
            [item.strip() for item in args.sample_ids.split(",") if item.strip()]
            if args.sample_ids
            else None
        )
        if args.max_llm_calls < 1:
            parser.error("--max-llm-calls must be at least 1")
        if args.max_total_tokens < 1:
            parser.error("--max-total-tokens must be at least 1")
        if args.max_runtime_seconds <= 0:
            parser.error("--max-runtime-seconds must be positive")
        try:
            result = AutoDraftBuilder(config).build(
                source_benchmark=args.source_bench,
                output_root=args.output_root,
                sample_ids=sample_ids,
                max_calls_per_sample=args.max_llm_calls,
                max_tokens_per_sample=args.max_total_tokens,
                max_seconds_per_sample=args.max_runtime_seconds,
            )
        except ValueError as exc:
            parser.error(str(exc))
        print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2))
        return 0

    if args.command == "build-researchbench40":
        sample_ids = (
            [item.strip() for item in args.sample_ids.split(",") if item.strip()]
            if args.sample_ids
            else None
        )
        if args.max_llm_calls < 1:
            parser.error("--max-llm-calls must be at least 1")
        if args.max_total_tokens < 1:
            parser.error("--max-total-tokens must be at least 1")
        if args.max_runtime_seconds <= 0:
            parser.error("--max-runtime-seconds must be positive")
        try:
            result = ResearchBench40Builder(config).build(
                manifest_path=args.manifest,
                output_dir=args.output_dir,
                sample_ids=sample_ids,
                resume=args.resume,
                max_calls_per_sample=args.max_llm_calls,
                max_tokens_per_sample=args.max_total_tokens,
                max_seconds_per_sample=args.max_runtime_seconds,
            )
        except ValueError as exc:
            parser.error(str(exc))
        print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2))
        return 0

    if args.command == "export-examples":
        try:
            manifest = export_comparison_examples(
                args.comparison,
                run_dir=config.runtime.run_dir,
                output_dir=args.output_dir,
                strategy=args.strategy,
                limit=args.limit,
            )
        except ValueError as exc:
            parser.error(str(exc))
        print(json.dumps({"manifest_path": str(manifest)}, ensure_ascii=False, indent=2))
        return 0

    if args.command == "export-runs":
        run_ids = [item.strip() for item in args.run_ids.split(",") if item.strip()]
        try:
            manifest = export_run_examples(
                run_ids,
                run_dir=config.runtime.run_dir,
                output_dir=args.output_dir,
            )
        except ValueError as exc:
            parser.error(str(exc))
        print(json.dumps({"manifest_path": str(manifest)}, ensure_ascii=False, indent=2))
        return 0

    if args.command == "demo-failures":
        # 该演示只使用合成输入和本地假响应，可安全用于面试现场或离线 CI。
        result = asyncio.run(run_failure_demo())
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["passed"] else 1

    if args.command in {"approval-list", "approval-decide"}:
        if not config.memory.enabled:
            parser.error("approval commands require memory.enabled=true")
        database_path = config.memory.sqlite_path or (
            config.runtime.run_dir / "deepresearch.sqlite3"
        )
        store = SQLiteMemoryStore(database_path)
        if args.command == "approval-list":
            if not 1 <= args.limit <= 500:
                parser.error("--limit must be between 1 and 500")
            requested_status = ApprovalStatus(args.status) if args.status else None
            approvals = store.list_approvals(
                run_id=args.run_id,
                status=requested_status,
                limit=args.limit,
            )
            print(
                json.dumps(
                    [item.model_dump(mode="json") for item in approvals],
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0

        decision = (
            ApprovalStatus.APPROVED
            if args.decision == "approve"
            else ApprovalStatus.REJECTED
        )
        try:
            approval = store.decide_approval(
                args.approval_id,
                decision=decision,
                decision_key=args.decision_key,
                decided_by="local_cli",
                reason=args.reason,
            )
        except KeyError:
            parser.error(f"approval request not found: {args.approval_id}")
        except ValueError as exc:
            parser.error(str(exc))
        print(json.dumps(approval.model_dump(mode="json"), ensure_ascii=False, indent=2))
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


def build_replay_summary(payload: dict[str, object]) -> dict[str, object]:
    """Build a compact operator view while retaining full replay as the source of truth."""

    state = payload.get("research_state")
    state_dict = state if isinstance(state, dict) else {}
    budget = state_dict.get("budget")
    budget_dict = budget if isinstance(budget, dict) else {}
    dag = payload.get("dag")
    dag_dict = dag if isinstance(dag, dict) else {}
    return {
        "schema_version": payload.get("schema_version"),
        "run_id": state_dict.get("run_id") or _nested_value(payload, "task", "run_id"),
        "query": state_dict.get("query") or _nested_value(payload, "task", "query"),
        "status": state_dict.get("status"),
        "current_round": state_dict.get("current_round", 0),
        "stop_reason": state_dict.get("stop_reason"),
        "pending_questions": state_dict.get("pending_questions", []),
        "open_gap_count": sum(
            1
            for gap in state_dict.get("evidence_gaps", [])
            if isinstance(gap, dict) and gap.get("status") == "OPEN"
        ),
        "conflict_count": len(state_dict.get("conflicts", [])),
        "rejected_evidence_count": len(state_dict.get("rejected_evidence", [])),
        "budget": budget_dict,
        "dag_summary": dag_dict.get("summary", {}),
        "report_title": _nested_value(payload, "report", "title"),
    }


def _nested_value(payload: dict[str, object], parent: str, key: str) -> object | None:
    value = payload.get(parent)
    return value.get(key) if isinstance(value, dict) else None


if __name__ == "__main__":
    raise SystemExit(main())

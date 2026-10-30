from deepresearch.cli import build_replay_summary, main
from deepresearch.cli import build_parser


def test_cli_accepts_llm_profile_override():
    args = build_parser().parse_args(
        ["--llm-profile", "planner_deepseek", "run", "--query", "test"]
    )

    assert args.llm_profile == "planner_deepseek"


def test_cli_accepts_agent_specific_llm_profile_overrides():
    args = build_parser().parse_args(
        [
            "--planner-llm-profile",
            "planner_model",
            "--reader-llm-profile",
            "reader_model",
            "--synthesizer-llm-profile",
            "writer_model",
            "--redteam-llm-profile",
            "reviewer_model",
            "--evaluator-llm-profile",
            "judge_model",
            "run",
            "--query",
            "test",
        ]
    )

    assert args.planner_llm_profile == "planner_model"
    assert args.reader_llm_profile == "reader_model"
    assert args.synthesizer_llm_profile == "writer_model"
    assert args.redteam_llm_profile == "reviewer_model"
    assert args.evaluator_llm_profile == "judge_model"


def test_cli_accepts_real_run_research_overrides():
    args = build_parser().parse_args(
        [
            "run",
            "--query",
            "test",
            "--retrieval-provider",
            "hybrid",
            "--retrieval-top-k",
            "6",
            "--enable-research-loop",
            "--max-research-rounds",
            "3",
            "--disable-memory-retrieval",
        ]
    )

    assert args.retrieval_provider == "hybrid"
    assert args.retrieval_top_k == 6
    assert args.enable_research_loop is True
    assert args.max_research_rounds == 3
    assert args.disable_memory_retrieval is True


def test_cli_accepts_retrieval_smoke_overrides():
    args = build_parser().parse_args(
        ["retrieval-smoke", "--query", "test", "--provider", "tavily", "--top-k", "3"]
    )

    assert args.command == "retrieval-smoke"
    assert args.provider == "tavily"
    assert args.top_k == 3


def test_cli_accepts_retrieval_comparison_providers():
    args = build_parser().parse_args(
        ["compare-retrieval", "--providers", "local,tavily", "--bench", "bench.json"]
    )

    assert args.command == "compare-retrieval"
    assert args.providers == "local,tavily"
    assert args.bench == "bench.json"


def test_cli_accepts_compression_comparison_options():
    args = build_parser().parse_args(
        ["compare-compression", "--provider", "local", "--max-evidence", "4"]
    )

    assert args.command == "compare-compression"
    assert args.provider == "local"
    assert args.max_evidence == 4


def test_cli_accepts_system_comparison_options():
    args = build_parser().parse_args(
        ["compare-system", "--bench", "bench.json", "--strategies", "baseline_linear,dag_multi_agent"]
    )

    assert args.command == "compare-system"
    assert args.strategies == "baseline_linear,dag_multi_agent"


def test_cli_accepts_track_a_system_controls():
    args = build_parser().parse_args(
        [
            "compare-system",
            "--judge-strategies",
            "baseline_linear,dag_memory_redteam_compress",
            "--memory-seed-mode",
            "empty",
            "--max-llm-calls",
            "40",
            "--max-total-tokens",
            "120000",
            "--max-runtime-seconds",
            "600",
            "--sample-ids",
            "full_005,full_015,full_025",
        ]
    )

    assert args.judge_strategies == "baseline_linear,dag_memory_redteam_compress"
    assert args.memory_seed_mode == "empty"
    assert args.max_llm_calls == 40
    assert args.sample_ids == "full_005,full_015,full_025"


def test_cli_accepts_auto_draft_staging_subset():
    args = build_parser().parse_args(
        ["build-auto-draft", "--sample-ids", "full_005,full_015"]
    )

    assert args.command == "build-auto-draft"
    assert args.sample_ids == "full_005,full_015"
    assert args.max_llm_calls == 12


def test_cli_accepts_researchbench40_resume_controls():
    args = build_parser().parse_args(
        [
            "build-researchbench40",
            "--sample-ids",
            "full_005,rb40_controversy_en_01",
            "--no-resume",
        ]
    )

    assert args.command == "build-researchbench40"
    assert args.sample_ids == "full_005,rb40_controversy_en_01"
    assert args.resume is False
    assert args.max_llm_calls == 12


def test_cli_accepts_example_export_options():
    args = build_parser().parse_args(
        ["export-examples", "--comparison", "result.json", "--limit", "2"]
    )

    assert args.command == "export-examples"
    assert args.comparison == "result.json"
    assert args.limit == 2


def test_cli_accepts_real_run_export_options():
    args = build_parser().parse_args(
        ["export-runs", "--run-ids", "run_a,run_b", "--output-dir", "examples/real"]
    )

    assert args.command == "export-runs"
    assert args.run_ids == "run_a,run_b"


def test_cli_accepts_resume_and_replay_summary_options():
    resume_args = build_parser().parse_args(["resume", "--run-id", "run_a"])
    replay_args = build_parser().parse_args(["replay", "--run-id", "run_a", "--summary"])

    assert resume_args.command == "resume"
    assert replay_args.summary is True


def test_cli_accepts_local_approval_commands():
    list_args = build_parser().parse_args(
        ["approval-list", "--run-id", "run_a", "--status", "PENDING"]
    )
    decide_args = build_parser().parse_args(
        [
            "approval-decide",
            "--approval-id",
            "approval_1",
            "--decision",
            "approve",
            "--decision-key",
            "operator_request_1",
        ]
    )

    assert list_args.command == "approval-list"
    assert list_args.status == "PENDING"
    assert decide_args.command == "approval-decide"
    assert decide_args.decision_key == "operator_request_1"


def test_build_replay_summary_exposes_controller_state():
    summary = build_replay_summary(
        {
            "schema_version": "2.0",
            "research_state": {
                "run_id": "run_a",
                "query": "question",
                "status": "DONE",
                "current_round": 2,
                "stop_reason": "no_progress",
                "pending_questions": ["missing"],
                "evidence_gaps": [{"status": "OPEN"}, {"status": "RESOLVED"}],
                "conflicts": [{}],
                "rejected_evidence": [{}, {}],
                "budget": {"llm_calls": 3},
            },
            "dag": {"summary": {"total_nodes": 4}},
            "report": {"title": "Report"},
        }
    )

    assert summary["current_round"] == 2
    assert summary["open_gap_count"] == 1
    assert summary["conflict_count"] == 1
    assert summary["rejected_evidence_count"] == 2


def test_cli_run_writes_replay(tmp_path, capsys):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "retrieval:",
                "  corpus_dir: data/local_corpus",
                "  top_k: 2",
                "runtime:",
                f"  run_dir: {tmp_path}",
                "  max_subtasks: 2",
            ]
        ),
        encoding="utf-8",
    )

    code = main(["--config", str(config_path), "run", "--query", "Red-Blue 对抗有什么作用？"])
    output = capsys.readouterr().out

    assert code == 0
    assert "Run ID:" in output

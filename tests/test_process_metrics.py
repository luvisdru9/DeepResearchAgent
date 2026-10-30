from deepresearch.evaluation.process_metrics import score_research_process


def test_process_metrics_read_sources_fallback_retries_stop_and_resources_from_replay():
    replay = {
        "research_state": {
            "stop_reason": "budget_exhausted",
            "budget": {
                "exhausted": True,
                "exhausted_limit": "max_llm_calls",
                "llm_calls": 4,
            },
        },
        "planner": {"fallback_used": True, "error": None},
        "synthesizer": {"fallback_used": False, "error": None},
        "redteam": {"fallback_used": False, "error": None},
        "docs_by_task": {
            "task_1": [
                {
                    "id": "required_1",
                    "url": "https://one.example/a",
                    "source_type": "research_paper",
                    "metadata": {},
                },
                {
                    "id": "secondary_1",
                    "url": "https://two.example/b",
                    "source_type": "independent_analysis",
                    "metadata": {},
                },
            ],
            "task_2": [
                {
                    "id": "required_1",
                    "url": "https://one.example/a",
                    "source_type": "research_paper",
                    "metadata": {},
                }
            ],
        },
        "evidence": [{"id": "ev_1"}, {"id": "ev_2"}],
        "node_traces": [
            {"subtask_id": "task_1", "status": "DONE", "attempt_traces": [{}, {}]},
            {"subtask_id": "task_2", "status": "FAILED", "attempt_traces": [{}]},
        ],
        "node_results": {
            "task_1": {
                "diagnostics": {
                    "reader": {
                        "empty_result_rule_assist_used": True,
                        "doc_results": [
                            {"fallback_used": True},
                            {"fallback_used": False},
                        ]
                    }
                }
            }
        },
        "metrics": {
            "latency_seconds": 12.5,
            "llm_input_tokens": 100,
            "llm_output_tokens": 20,
            "cost_usd": 0.4,
        },
    }

    metrics = score_research_process(
        replay,
        required_source_ids=["required_1", "required_2"],
    )

    assert metrics["required_source_recall"] == 0.5
    assert metrics["independent_domain_count"] == 2
    assert metrics["primary_source_ratio"] == 0.5
    assert metrics["evidence_yield"] == 1.0
    assert metrics["fallback_count"] == 2
    assert metrics["empty_result_rule_assist_count"] == 1
    assert metrics["retry_count"] == 1
    assert metrics["failed_stages"] == ["task_2"]
    assert metrics["stop_reason"] == "budget_exhausted"
    assert metrics["resources"]["llm_calls"] == 4
    assert metrics["resources"]["llm_input_tokens"] == 100

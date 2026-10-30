import asyncio
import json

from deepresearch.cli import build_parser, main
from deepresearch.demos import run_failure_demo


def test_failure_demo_exercises_four_defenses_without_external_calls():
    result = asyncio.run(run_failure_demo())

    assert result["passed"] is True
    assert result["external_api_calls"] == 0
    assert [item["id"] for item in result["scenarios"]] == [
        "prompt_injection_scan",
        "fabricated_quote_rejection",
        "transient_retry_backoff",
        "honest_quality_failure",
    ]
    assert all(item["passed"] for item in result["scenarios"])


def test_failure_demo_cli_prints_machine_readable_result(capsys):
    args = build_parser().parse_args(["demo-failures"])
    assert args.command == "demo-failures"

    exit_code = main(["demo-failures"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["passed"] is True
    assert payload["external_api_calls"] == 0

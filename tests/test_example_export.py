import json

from deepresearch.evaluation import export_comparison_examples, export_run_examples


def test_export_comparison_examples_writes_replay_report_and_manifest(tmp_path):
    run_dir = tmp_path / "runs"
    run_dir.mkdir()
    (run_dir / "sample_run.json").write_text(
        json.dumps(
            {
                "schema_version": "2.0",
                "report": {"markdown": "# Example report"},
            }
        ),
        encoding="utf-8",
    )
    comparison = tmp_path / "comparison.json"
    comparison.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "strategy": "dag_memory_redteam_compress",
                        "results": [{"sample_id": "sample", "run_id": "sample_run"}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    manifest_path = export_comparison_examples(
        comparison,
        run_dir=run_dir,
        output_dir=tmp_path / "examples",
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(manifest["examples"]) == 1
    assert (tmp_path / "examples" / "replays" / "sample_sample_run_replay.json").exists()
    assert (
        tmp_path / "examples" / "reports" / "sample_sample_run_report.md"
    ).read_text(encoding="utf-8") == "# Example report"


def test_export_run_examples_marks_real_runs_for_human_review(tmp_path):
    run_dir = tmp_path / "runs"
    run_dir.mkdir()
    (run_dir / "real_run.json").write_text(
        json.dumps(
            {
                "schema_version": "2.0",
                "task": {"query": "question", "language": "en"},
                "report": {"markdown": "# Real report"},
            }
        ),
        encoding="utf-8",
    )

    manifest_path = export_run_examples(
        ["real_run"], run_dir=run_dir, output_dir=tmp_path / "real"
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["kind"] == "real_pipeline_candidates"
    assert manifest["examples"][0]["review_status"] == "candidate_requires_human_review"
    assert (tmp_path / "real" / "reports" / "en_real_run_report.md").exists()

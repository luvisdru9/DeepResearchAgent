import json

from deepresearch.orchestrator.replay import ReplayStore


def test_replay_store_marks_new_files_as_schema_2(tmp_path):
    store = ReplayStore(tmp_path)

    store.write(
        "new-run",
        {"task": {"query": "test", "created_at": "2026-06-20T01:51:33.046931Z"}},
    )

    payload = json.loads(
        (tmp_path / "20260620T015133Z_new-run_replay.json").read_text(encoding="utf-8")
    )
    assert payload["schema_version"] == "2.0"


def test_replay_store_identifies_legacy_files_without_inventing_dag(tmp_path):
    store = ReplayStore(tmp_path)
    (tmp_path / "legacy-run.json").write_text(
        json.dumps({"task": {"query": "legacy"}}, ensure_ascii=False),
        encoding="utf-8",
    )

    payload = store.read("legacy-run")

    assert payload["schema_version"] == "1.0"
    assert "dag" not in payload


def test_replay_store_reads_new_readable_filename_by_run_id(tmp_path):
    store = ReplayStore(tmp_path)
    path = tmp_path / "20260620T015133Z_readable-run_replay.json"
    path.write_text(
        json.dumps({"task": {"query": "readable"}, "schema_version": "2.0"}, ensure_ascii=False),
        encoding="utf-8",
    )

    payload = store.read("readable-run")

    assert payload["schema_version"] == "2.0"
    assert payload["task"]["query"] == "readable"


def test_replay_store_reads_new_readable_filename_by_file_stem_or_name(tmp_path):
    store = ReplayStore(tmp_path)
    path = tmp_path / "20260620T015133Z_readable-run_replay.json"
    path.write_text(
        json.dumps({"task": {"query": "readable"}, "schema_version": "2.0"}, ensure_ascii=False),
        encoding="utf-8",
    )

    by_stem = store.read("20260620T015133Z_readable-run")
    by_name = store.read("20260620T015133Z_readable-run_replay.json")

    assert by_stem["task"]["query"] == "readable"
    assert by_name["task"]["query"] == "readable"

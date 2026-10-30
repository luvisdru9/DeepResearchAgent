import hashlib
import json
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "project"
ROOT = REPOSITORY_ROOT if (REPOSITORY_ROOT / "experiments").is_dir() else FIXTURE_ROOT
EXPERIMENTS = ROOT / "experiments"


def _sha256(path: Path) -> str:
    """Use the same prefixed hash notation as experiment manifests."""

    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def test_experiment_index_points_to_structured_records():
    index = json.loads((EXPERIMENTS / "index.json").read_text(encoding="utf-8"))
    schema = json.loads(
        (EXPERIMENTS / "schema/experiment_record.schema.json").read_text(encoding="utf-8")
    )
    required = set(schema["required"])
    ids: list[str] = []

    for item in index["records"]:
        path = EXPERIMENTS / item["path"]
        assert path.exists()
        record = json.loads(path.read_text(encoding="utf-8"))
        assert required.issubset(record)
        assert item["experiment_id"] == record["experiment_id"]
        assert item["status"] == record["status"]
        ids.append(record["experiment_id"])
    assert len(ids) == len(set(ids))


def test_current_baseline_pins_config_benchmark_and_unmeasured_metrics():
    path = EXPERIMENTS / "records/20261013_b2_claim_citation_process_v1/record.json"
    record = json.loads(path.read_text(encoding="utf-8"))

    config = record["code"]["config"]
    benchmark = record["benchmark"]
    if config["sha256"] is not None:
        assert config["sha256"] == _sha256(ROOT / config["path"])
    assert benchmark["sha256"] == _sha256(ROOT / benchmark["path"])
    for metric in record["results"]["metrics"]:
        if metric["status"] == "not_measured":
            assert metric["value"] is None


def test_experiment_records_do_not_store_secret_fields():
    forbidden_keys = {"api_key", "token", "cookie", "password", "secret"}
    for path in EXPERIMENTS.rglob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        pending = [payload]
        while pending:
            item = pending.pop()
            if isinstance(item, dict):
                assert not forbidden_keys.intersection(key.casefold() for key in item)
                pending.extend(item.values())
            elif isinstance(item, list):
                pending.extend(item)


def test_b3_annotation_packet_manifest_pins_frozen_handoff_files():
    """人工标注交付必须能发现冻结包被误改，不得静默漂移。"""

    annotation_dir = (
        EXPERIMENTS
        / "records/20261013_b3_judge_calibration_v1/annotations"
    )
    manifest = json.loads(
        (annotation_dir / "packet_manifest.json").read_text(encoding="utf-8")
    )

    assert manifest["status"] == "frozen_pending_human_annotation"
    assert manifest["handoff_ready"] is True
    for item in manifest["files"]:
        path = annotation_dir / item["path"]
        assert path.exists()
        assert item["sha256"] == _sha256(path).removeprefix("sha256:")


def test_current_b3_handoff_is_frozen_and_contains_no_prefilled_labels():
    """新版独立标注包不得继承旧人工答案或自动 Judge 标签。"""

    annotation_dir = (
        EXPERIMENTS
        / "records/20261016_b3_current_reports_three_judge_v1/annotations"
    )
    manifest = json.loads(
        (annotation_dir / "packet_manifest.json").read_text(encoding="utf-8")
    )

    assert manifest["status"] == "frozen_pending_human_annotation"
    assert manifest["judge_repetitions"] == 3
    assert manifest["report_claim_count_per_packet"] == 25
    for item in manifest["files"]:
        path = annotation_dir / item["path"]
        assert item["sha256"] == _sha256(path).removeprefix("sha256:")
        packet = json.loads(path.read_text(encoding="utf-8"))
        assert all(sample["status"] == "pending" for sample in packet["samples"])
        assert all(
            claim["label"] is None and claim["reason"] == ""
            for sample in packet["samples"]
            for field in ("report_claims", "gold_claims")
            for claim in sample[field]
        )

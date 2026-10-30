from __future__ import annotations

from deepresearch.evaluation.annotation_form import (
    render_annotation_form,
    write_annotation_form,
)
from deepresearch.evaluation.annotations import HumanAnnotationBundle, build_annotation_set
from deepresearch.evaluation.benchmark import AtomicClaim
from deepresearch.schema import Evidence, ResearchReport


def _bundle() -> HumanAnnotationBundle:
    report = ResearchReport(
        run_id="form_run",
        query="如何标注？",
        title="标注测试报告",
        summary="测试表单生成。",
        markdown="## 发现\nDAG 支持并发。 [1]",
        key_findings=["DAG 支持并发。 [1]"],
        evidence=[
            Evidence(
                id="ev_form",
                source_id="doc1",
                source_title="测试 </script> 来源",
                source_url="repo://test",
                claim="DAG 支持并发。",
                quote="DAG 支持并发。",
            )
        ],
        citation_map={1: "ev_form"},
    )
    annotation = build_annotation_set(
        benchmark_id="bench",
        benchmark_version="1.0",
        sample_id="sample_1",
        report=report,
        gold_claims=[AtomicClaim(id="gold_1", text="DAG 支持并发")],
        annotator_id="annotator_a_pending",
    )
    return HumanAnnotationBundle(
        annotator_id="annotator_a_pending", samples=[annotation]
    )


def test_form_is_self_contained_and_has_no_preselected_human_labels():
    html = render_annotation_form(_bundle(), source_name="packet.json")

    assert "独立标注表" in html
    assert "导出当前草稿 JSON" in html
    assert "检查并导出完成版" in html
    assert "supported" in html
    assert '"label": null' in html
    assert "测试 </script> 来源" not in html
    assert "测试 \\u003c/script> 来源" in html
    assert "https://" not in html
    assert "Only merge human-editable fields" in html
    assert "report claim ID 与冻结包不一致" in html


def test_write_form_creates_parent_directory(tmp_path):
    destination = write_annotation_form(
        _bundle(),
        tmp_path / "forms" / "annotator_a_form.html",
        source_name="annotator_a_pending.json",
    )

    assert destination.exists()
    assert "annotator_a_pending.json" in destination.read_text(encoding="utf-8")

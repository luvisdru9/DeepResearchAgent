import json
from pathlib import Path

import pytest

from deepresearch.config import AppConfig, RuntimeConfig
from deepresearch.evaluation import (
    ResearchBench40Builder,
    ResearchBench40Manifest,
    load_benchmark_bundle,
)
from deepresearch.llm import MockLLMClient
from deepresearch.retrieval.base import SearchProvider
from deepresearch.schema import SourceDoc


MANIFEST_PATH = Path(
    "data/benchmarks/researchbench_v2/researchbench_40_v1/candidates.json"
)


class FixedSearchProvider(SearchProvider):
    """Return two deterministic primary sources for offline builder tests."""

    def __init__(self, documents: list[SourceDoc]) -> None:
        self.documents = documents

    def search(self, query: str, top_k: int = 5) -> list[SourceDoc]:
        return self.documents[:top_k]

    def search_with_diagnostics(
        self,
        query: str,
        top_k: int = 5,
    ) -> tuple[list[SourceDoc], dict[str, object]]:
        return self.search(query, top_k), {"query": query, "provider": "fixed"}


def _documents() -> list[SourceDoc]:
    return [
        SourceDoc(
            id="oauth_doc",
            title="OAuth specification",
            url="https://www.ietf.org/example/oauth",
            content="OAuth 2.0 主要用于授权。授权服务器向客户端签发访问令牌。",
            source_type="web",
        ),
        SourceDoc(
            id="oidc_doc",
            title="OpenID Connect Core",
            url="https://openid.net/specs/example",
            content=(
                "OpenID Connect 在 OAuth 2.0 上增加身份认证层。"
                "ID Token 与 Access Token 用途不同。"
            ),
            source_type="web",
        ),
    ]


def _author_payload() -> str:
    return json.dumps(
        {
            "claims": [
                {
                    "claim_id": "full_007.c1",
                    "text": "OAuth 2.0 主要用于授权",
                    "source_id": "oauth_doc",
                    "quote": "OAuth 2.0 主要用于授权",
                },
                {
                    "claim_id": "full_007.c2",
                    "text": "OpenID Connect 在 OAuth 2.0 上增加身份认证层",
                    "source_id": "oidc_doc",
                    "quote": "OpenID Connect 在 OAuth 2.0 上增加身份认证层",
                },
                {
                    "claim_id": "full_007.c3",
                    "text": "ID Token 与 Access Token 用途不同",
                    "source_id": "oidc_doc",
                    "quote": "ID Token 与 Access Token 用途不同",
                },
            ]
        },
        ensure_ascii=False,
    )


def _review_payload(verdict: str = "supported") -> str:
    claims = [
        ("full_007.c1", "oauth_doc", "OAuth 2.0 主要用于授权"),
        ("full_007.c2", "oidc_doc", "OpenID Connect 在 OAuth 2.0 上增加身份认证层"),
        ("full_007.c3", "oidc_doc", "ID Token 与 Access Token 用途不同"),
    ]
    return json.dumps(
        {
            "claims": [
                {
                    "claim_id": claim_id,
                    "verdict": verdict,
                    "source_id": source_id if verdict == "supported" else None,
                    "quote": quote if verdict == "supported" else None,
                    "reason": "The exact frozen span directly supports the claim.",
                }
                for claim_id, source_id, quote in claims
            ]
        },
        ensure_ascii=False,
    )


def _builder(tmp_path: Path, *, supported: bool = True) -> ResearchBench40Builder:
    verdict = "supported" if supported else "unsupported"
    # A failed support round triggers one conservative rewrite and a second review.
    first_responses = [_author_payload(), _review_payload(verdict)]
    second_responses = [_review_payload(verdict)]
    if not supported:
        first_responses.extend([_author_payload(), _review_payload(verdict)])
        second_responses.append(_review_payload(verdict))
    return ResearchBench40Builder(
        AppConfig(runtime=RuntimeConfig(run_dir=tmp_path / "runs")),
        search_provider=FixedSearchProvider(_documents()),
        review_clients=[
            MockLLMClient(
                responses=first_responses,
                profile_name="flash_0",
                provider_name="mock",
            ),
            MockLLMClient(
                responses=second_responses,
                profile_name="flash_1",
                provider_name="mock",
            ),
        ],
    )


def test_researchbench40_manifest_is_strictly_balanced():
    manifest_file = MANIFEST_PATH.resolve()
    manifest = ResearchBench40Manifest.model_validate_json(
        manifest_file.read_text(encoding="utf-8")
    )
    samples = ResearchBench40Builder._resolve_candidates(manifest_file, manifest)

    ResearchBench40Builder._validate_distribution(samples)

    assert len(samples) == 40
    assert sum(item.language == "zh" for item in samples) == 20
    assert sum(item.language == "en" for item in samples) == 20
    for task_type in {
        "fact_qa",
        "comparative_research",
        "trend_analysis",
        "controversy_multihop",
    }:
        task_samples = [item for item in samples if item.task_type == task_type]
        assert len(task_samples) == 10
        assert sum(item.language == "zh" for item in task_samples) == 5
        assert sum(item.language == "en" for item in task_samples) == 5


def test_researchbench40_uses_hybrid_retrieval_without_mutating_app_config(
    tmp_path,
    monkeypatch,
):
    captured: dict[str, str] = {}

    def fake_factory(retrieval_config):
        captured["provider"] = retrieval_config.provider
        return FixedSearchProvider(_documents())

    monkeypatch.setattr(
        "deepresearch.evaluation.auto_draft.create_search_provider",
        fake_factory,
    )
    config = AppConfig(runtime=RuntimeConfig(run_dir=tmp_path / "runs"))
    ResearchBench40Builder(
        config,
        review_clients=[
            MockLLMClient(profile_name="flash_0", provider_name="mock"),
            MockLLMClient(profile_name="flash_1", provider_name="mock"),
        ],
    )

    assert captured["provider"] == "hybrid"
    assert config.retrieval.provider == "local"


def test_researchbench40_builds_partial_draft_and_resumes(tmp_path):
    output = tmp_path / "researchbench40"
    result = _builder(tmp_path).build(
        manifest_path=MANIFEST_PATH,
        output_dir=output,
        sample_ids=["full_007"],
    )

    assert result.passed_count == 1
    assert result.failed_count == 0
    assert result.pending_count == 39
    assert result.dataset_path is None
    assert result.dataset_draft_path is not None
    bundle = load_benchmark_bundle(result.dataset_draft_path)
    assert bundle.dataset_v2 is not None
    sample = bundle.dataset_v2.samples[0]
    assert len(sample.claims) == 3
    assert {item.dimension for item in sample.rubric} == {
        "information_recall",
        "analysis_reasoning",
        "citation_quality",
        "task_following",
        "presentation",
    }
    review = json.loads(
        (output / "annotations/automatic_reviews/full_007.json").read_text(encoding="utf-8")
    )
    assert review["review_profiles"] == ["flash_0", "flash_1"]
    assert "messages" not in review

    # No queued responses are available: a successful second build proves that
    # the accepted sample was hash-validated and skipped rather than re-called.
    resumed = ResearchBench40Builder(
        AppConfig(runtime=RuntimeConfig(run_dir=tmp_path / "resumed-runs")),
        search_provider=FixedSearchProvider(_documents()),
        review_clients=[
            MockLLMClient(responses=[], profile_name="flash_0", provider_name="mock"),
            MockLLMClient(responses=[], profile_name="flash_1", provider_name="mock"),
        ],
    ).build(
        manifest_path=MANIFEST_PATH,
        output_dir=output,
        sample_ids=["full_007"],
    )
    assert resumed.samples[0].resumed is True
    assert resumed.samples[0].llm_calls == 3


def test_researchbench40_resume_rejects_snapshot_tampering(tmp_path):
    output = tmp_path / "researchbench40"
    _builder(tmp_path).build(
        manifest_path=MANIFEST_PATH,
        output_dir=output,
        sample_ids=["full_007"],
    )
    manifest = json.loads(
        (output / "snapshots/full_007/manifest.json").read_text(encoding="utf-8")
    )
    document = output / "snapshots/full_007" / manifest["documents"][0]["path"]
    document.write_text("tampered", encoding="utf-8")

    with pytest.raises(ValueError, match="Frozen document hash mismatch"):
        _builder(tmp_path).build(
            manifest_path=MANIFEST_PATH,
            output_dir=output,
            sample_ids=["full_007"],
        )


def test_researchbench40_no_resume_archives_accepted_version(tmp_path):
    output = tmp_path / "researchbench40"
    _builder(tmp_path).build(
        manifest_path=MANIFEST_PATH,
        output_dir=output,
        sample_ids=["full_007"],
    )

    replaced = _builder(tmp_path).build(
        manifest_path=MANIFEST_PATH,
        output_dir=output,
        sample_ids=["full_007"],
        resume=False,
    )

    assert replaced.samples[0].status == "passed"
    assert replaced.samples[0].resumed is False
    archives = list((output / "validation/superseded/full_007").iterdir())
    assert len(archives) == 1
    assert (archives[0] / "sample.json").is_file()
    assert (archives[0] / "snapshot/manifest.json").is_file()
    assert (archives[0] / "automatic_review.json").is_file()
    assert (archives[0] / "result.json").is_file()


def test_researchbench40_keeps_failed_fixed_sample_without_replacement(tmp_path):
    output = tmp_path / "researchbench40"
    result = _builder(tmp_path, supported=False).build(
        manifest_path=MANIFEST_PATH,
        output_dir=output,
        sample_ids=["full_007"],
    )

    assert result.passed_count == 0
    assert result.failed_count == 1
    assert result.pending_count == 39
    assert result.dataset_path is None
    assert "Fewer than 3 candidate gold claims" in (result.samples[0].error or "")
    assert not (output / "validation/samples/full_007.sample.json").exists()


def test_metric_protocol_defines_future_metrics_without_results():
    payload = json.loads(
        Path(
            "data/benchmarks/researchbench_v2/researchbench_40_v1/metric_protocol.json"
        ).read_text(encoding="utf-8")
    )

    assert payload["evaluation_executed"] is False
    assert payload["citation_coverage_term"] == "citation_completeness"
    assert payload["aggregation"]["primary"] == "sample_macro_average"

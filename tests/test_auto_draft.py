import json

from deepresearch.config import AppConfig, RuntimeConfig
from deepresearch.evaluation import AutoDraftBuilder, load_benchmark_bundle
from deepresearch.evaluation.auto_draft import (
    DraftGoldSupportReview,
    _compact_source_payload,
    _recover_exact_quote,
    _sample_requires_primary,
    _select_review_documents,
)
from deepresearch.evaluation.benchmark import load_benchmark
from deepresearch.llm import MockLLMClient, MockReply
from deepresearch.retrieval.base import SearchProvider
from deepresearch.schema import SourceDoc


class FixedSearchProvider(SearchProvider):
    def __init__(self, documents):
        self.documents = documents

    def search(self, query: str, top_k: int = 5):
        return self.documents[:top_k]

    def search_with_diagnostics(self, query: str, top_k: int = 5):
        return self.search(query, top_k), {"query": query, "provider": "fixed"}


def _review_payload(verdict="supported"):
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


def _author_payload():
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


def _documents():
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


def test_auto_draft_builds_strict_engineering_dataset(tmp_path):
    config = AppConfig(runtime=RuntimeConfig(run_dir=tmp_path / "runs"))
    clients = [
        MockLLMClient(
            responses=[_author_payload(), _review_payload()],
            profile_name="flash_0",
            provider_name="mock",
        ),
        MockLLMClient(
            responses=[_review_payload()],
            profile_name="flash_1",
            provider_name="mock",
        ),
    ]
    builder = AutoDraftBuilder(
        config,
        search_provider=FixedSearchProvider(_documents()),
        review_clients=clients,
    )

    result = builder.build(
        output_root=tmp_path / "drafts",
        sample_ids=["full_007"],
    )

    assert result.passed_count == 1
    assert result.failed_count == 0
    assert result.dataset_path is not None
    bundle = load_benchmark_bundle(result.dataset_path)
    assert bundle.dataset_v2 is not None
    assert bundle.dataset_v2.metadata.release_tier == "engineering_automatic"
    assert bundle.dataset_v2.metadata.human_calibrated is False
    assert bundle.dataset_v2.samples[0].status == "draft"
    assert all(
        requirement.required_source_ids
        for requirement in bundle.dataset_v2.samples[0].evidence_requirements
    )
    assert result.samples[0].independent_domain_count == 2


def test_auto_draft_retains_fixed_sample_failure_without_dataset(tmp_path):
    config = AppConfig(runtime=RuntimeConfig(run_dir=tmp_path / "runs"))
    clients = [
        MockLLMClient(
            responses=[
                _author_payload(),
                _review_payload("unsupported"),
                _author_payload(),
                _review_payload("unsupported"),
            ],
            profile_name="flash_0",
            provider_name="mock",
        ),
        MockLLMClient(
            responses=[_review_payload("unsupported"), _review_payload("unsupported")],
            profile_name="flash_1",
            provider_name="mock",
        ),
    ]
    builder = AutoDraftBuilder(
        config,
        search_provider=FixedSearchProvider(_documents()),
        review_clients=clients,
    )

    result = builder.build(
        output_root=tmp_path / "drafts",
        sample_ids=["full_007"],
    )

    assert result.passed_count == 0
    assert result.failed_count == 1
    assert result.dataset_path is None
    assert "Fewer than two candidate gold claims" in (result.samples[0].error or "")
    build_records = list((tmp_path / "drafts").glob("*/build_result.json"))
    assert len(build_records) == 1


def test_auto_draft_review_normalizes_common_model_field_variants():
    review = DraftGoldSupportReview.model_validate(
        {
            "type": "json_object",
            "results": [
                {
                    "claim_id": "claim.c1",
                    "text": "ignored echo",
                    "supported": True,
                    "source_id": "doc",
                    "quote": "exact quote",
                    "reason": "Direct support.",
                }
            ],
        }
    )

    assert review.claims[0].verdict == "supported"
    assert review.claims[0].source_id == "doc"

    keyed = DraftGoldSupportReview.model_validate(
        {
            "c1": {
                "verdict": "supported",
                "source_id": "doc",
                "quote": "exact quote",
                "reason": "Direct support.",
            }
        }
    )
    assert keyed.claims[0].claim_id == "c1"

    status_alias = DraftGoldSupportReview.model_validate(
        {
            "claims": [
                {
                    "claim_id": "claim.c1",
                    "status": "supported",
                    "source_id": "doc",
                    "quote": "exact quote",
                }
            ]
        }
    )
    assert status_alias.claims[0].verdict == "supported"
    assert status_alias.claims[0].reason == "No reason supplied."

    note_alias = DraftGoldSupportReview.model_validate(
        {
            "claims": [
                {
                    "claim_id": "claim.c1",
                    "verdict": "unsupported",
                    "source_id": None,
                    "quote": None,
                    "note": "The supplied excerpt does not support the whole claim.",
                }
            ]
        }
    )
    assert note_alias.claims[0].reason.startswith("The supplied excerpt")


def test_auto_draft_compacts_to_relevant_contiguous_source_window():
    sample = {item.id: item for item in load_benchmark("data/researchbench_full.json")}["full_007"]
    relevant = "OpenID Connect 在 OAuth 2.0 上增加身份认证层"
    document = SourceDoc(
        id="long_doc",
        title="Long specification",
        url="https://openid.net/specs/long",
        content=("unrelated preface " * 200) + relevant + (" appendix" * 200),
        source_type="web",
    )

    payload = _compact_source_payload(sample, [document])

    assert len(payload[0]["content"]) <= 1200
    assert relevant in payload[0]["content"]
    assert payload[0]["content"] in document.content


def test_auto_draft_review_sources_prefer_primary_and_diverse_domains():
    documents = [
        SourceDoc(
            id="low_blog",
            title="Blog",
            url="https://example.com/blog",
            content="blog",
            source_type="web",
            credibility={"score": 0.3, "tier": "low", "signals": []},
        ),
        SourceDoc(
            id="primary_a",
            title="Primary A",
            url="https://arxiv.org/abs/1",
            content="paper a",
            source_type="web",
            credibility={
                "score": 0.8,
                "tier": "high",
                "signals": ["primary_research_repository"],
            },
        ),
        SourceDoc(
            id="primary_same_domain",
            title="Primary B",
            url="https://arxiv.org/abs/2",
            content="paper b",
            source_type="web",
            credibility={
                "score": 0.9,
                "tier": "high",
                "signals": ["primary_research_repository"],
            },
        ),
        SourceDoc(
            id="official_other_domain",
            title="Standard",
            url="https://ietf.org/rfc/example",
            content="standard",
            source_type="web",
            credibility={"score": 0.8, "tier": "high", "signals": []},
        ),
    ]

    selected = _select_review_documents(documents, limit=2)

    assert {item.id for item in selected} == {"primary_same_domain", "official_other_domain"}
    assert len({_domain for _domain in (item.url.split("/")[2] for item in selected)}) == 2


def test_auto_draft_quote_recovery_only_trims_edges():
    content = (
        "An SSL certificate chain is the ordered list of certificates that connects "
        "a server certificate back to a trusted root Certificate Authority."
    )
    prefixed = (
        "It is the ordered list of certificates that connects a server certificate "
        "back to a trusted root Certificate Authority."
    )

    recovered = _recover_exact_quote(content, prefixed)

    assert recovered == (
        "is the ordered list of certificates that connects a server certificate back "
        "to a trusted root Certificate Authority."
    )
    assert _recover_exact_quote(content, "A semantic paraphrase with no exact interior span.") is None


def test_auto_draft_primary_requirement_recognizes_english_source_language():
    sample = {item.id: item for item in load_benchmark("data/researchbench_full.json")}["full_007"]
    original_research = sample.model_copy(
        update={"required_sources": ["original peer-reviewed research"]}
    )
    formal_specification = sample.model_copy(
        update={"required_sources": ["W3C specification and authoritative material"]}
    )
    platform_documentation = sample.model_copy(
        update={"required_sources": ["platform documentation and engineering reports"]}
    )

    assert _sample_requires_primary(original_research) is True
    assert _sample_requires_primary(formal_specification) is True
    assert _sample_requires_primary(platform_documentation) is True


def test_auto_draft_stops_before_retry_that_would_exceed_token_budget(tmp_path):
    config = AppConfig(runtime=RuntimeConfig(run_dir=tmp_path / "runs"))
    clients = [
        MockLLMClient(
            responses=[
                MockReply(content=_author_payload(), input_tokens=7000),
                MockReply(content=_review_payload("unsupported"), input_tokens=7000),
            ],
            profile_name="flash_0",
            provider_name="mock",
        ),
        MockLLMClient(
            responses=[
                MockReply(content=_review_payload("unsupported"), input_tokens=7000),
            ],
            profile_name="flash_1",
            provider_name="mock",
        ),
    ]
    builder = AutoDraftBuilder(
        config,
        search_provider=FixedSearchProvider(_documents()),
        review_clients=clients,
    )

    result = builder.build(
        output_root=tmp_path / "drafts",
        sample_ids=["full_007"],
        max_tokens_per_sample=40000,
    )

    assert result.failed_count == 1
    assert "would exceed the per-sample token budget" in (result.samples[0].error or "")


def test_auto_draft_uses_single_conservative_retry_for_author_validation(tmp_path):
    invalid_author = json.loads(_author_payload())
    invalid_author["claims"][0]["source_id"] = "unknown_placeholder"
    config = AppConfig(runtime=RuntimeConfig(run_dir=tmp_path / "runs"))
    builder = AutoDraftBuilder(
        config,
        search_provider=FixedSearchProvider(_documents()),
        review_clients=[
            MockLLMClient(
                responses=[
                    json.dumps(invalid_author, ensure_ascii=False),
                    _author_payload(),
                    _review_payload(),
                ],
                profile_name="flash_0",
                provider_name="mock",
            ),
            MockLLMClient(
                responses=[_review_payload()],
                profile_name="flash_1",
                provider_name="mock",
            ),
        ],
    )

    result = builder.build(
        output_root=tmp_path / "drafts",
        sample_ids=["full_007"],
    )

    assert result.passed_count == 1
    assert result.samples[0].llm_calls == 4

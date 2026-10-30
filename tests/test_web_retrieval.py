from __future__ import annotations

import asyncio
import builtins
import json
from types import SimpleNamespace

import pytest

from deepresearch.agents.retriever import RetrieverAgent
from deepresearch.config import AppConfig, RetrievalConfig, RuntimeConfig
from deepresearch.retrieval.factory import create_search_provider
from deepresearch.retrieval.loaders import FetchResult, HTMLDocumentLoader, PDFDocumentLoader
from deepresearch.retrieval.smoke import run_retrieval_smoke
from deepresearch.retrieval.url_utils import canonicalize_url
from deepresearch.retrieval.web import TavilySearchProvider
from deepresearch.orchestrator.pipeline import ResearchPipeline
from deepresearch.schema import ResearchTask, SubTask


def test_canonicalize_url_removes_tracking_and_fragment():
    assert canonicalize_url(
        "HTTPS://Example.COM:443/report/?b=2&utm_source=test&a=1#results"
    ) == "https://example.com/report?a=1&b=2"


def test_html_loader_extracts_visible_text_without_scripts():
    def fake_fetcher(url: str, timeout_seconds: float, max_bytes: int) -> FetchResult:
        assert url == "https://example.com/article"
        assert timeout_seconds == 3
        assert max_bytes == 1000
        return FetchResult(
            body=(
                b"<html><head><title>Research Result</title><script>secret()</script></head>"
                b"<body><nav>menu</nav><h1>Main claim</h1><p>Supporting evidence.</p></body></html>"
            ),
            content_type="text/html",
            final_url=url,
        )

    loader = HTMLDocumentLoader(
        timeout_seconds=3,
        max_bytes=1000,
        max_chars=500,
        fetcher=fake_fetcher,
    )

    loaded = loader.load("https://example.com/article")

    assert loaded.title == "Research Result"
    assert "Main claim" in loaded.content
    assert "Supporting evidence." in loaded.content
    assert "secret" not in loaded.content
    assert "menu" not in loaded.content


def test_pdf_loader_explains_optional_dependency(monkeypatch):
    original_import = builtins.__import__

    def fail_pypdf_import(name, *args, **kwargs):
        if name == "pypdf":
            raise ImportError("not installed")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_pypdf_import)
    monkeypatch.setattr("deepresearch.retrieval.loaders.shutil.which", lambda _: None)

    with pytest.raises(RuntimeError, match="either the optional 'pypdf'"):
        PDFDocumentLoader().load("https://example.com/report.pdf")


def test_pdf_loader_uses_pdftotext_without_new_python_dependency(monkeypatch):
    original_import = builtins.__import__

    def fail_pypdf_import(name, *args, **kwargs):
        if name == "pypdf":
            raise ImportError("not installed")
        return original_import(name, *args, **kwargs)

    def fake_fetcher(url: str, timeout_seconds: float, max_bytes: int) -> FetchResult:
        del timeout_seconds, max_bytes
        return FetchResult(b"%PDF-test", "application/pdf", url)

    def fake_run(args, **kwargs):
        assert args[1:] == ["-layout", "-", "-"]
        assert kwargs["input"] == b"%PDF-test"
        return SimpleNamespace(
            returncode=0,
            stdout=b"Introduction\fTable 3 Precision 82.1 Recall 79.4 F1 80.7\f",
            stderr=b"",
        )

    monkeypatch.setattr(builtins, "__import__", fail_pypdf_import)
    monkeypatch.setattr(
        "deepresearch.retrieval.loaders.shutil.which",
        lambda _: "/usr/bin/pdftotext",
    )
    monkeypatch.setattr("deepresearch.retrieval.loaders.subprocess.run", fake_run)

    loaded = PDFDocumentLoader(fetcher=fake_fetcher).load(
        "https://example.com/report.pdf"
    )

    assert "F1 80.7" in loaded.content
    assert loaded.metadata["extraction_backend"] == "pdftotext"
    assert loaded.metadata["page_count"] == 2
    assert loaded.metadata["content_truncated"] is False


def test_pdf_loader_can_request_reading_order_for_multicolumn_evidence(monkeypatch):
    """Reading-order extraction omits -layout so columns do not interleave quotes."""

    original_import = builtins.__import__

    def fail_pypdf_import(name, *args, **kwargs):
        if name == "pypdf":
            raise ImportError("not installed")
        return original_import(name, *args, **kwargs)

    def fake_fetcher(url: str, timeout_seconds: float, max_bytes: int) -> FetchResult:
        del timeout_seconds, max_bytes
        return FetchResult(b"%PDF-test", "application/pdf", url)

    def fake_run(args, **kwargs):
        assert args[1:] == ["-", "-"]
        assert kwargs["input"] == b"%PDF-test"
        return SimpleNamespace(returncode=0, stdout=b"Citation recall definition", stderr=b"")

    monkeypatch.setattr(builtins, "__import__", fail_pypdf_import)
    monkeypatch.setattr(
        "deepresearch.retrieval.loaders.shutil.which",
        lambda _: "/usr/bin/pdftotext",
    )
    monkeypatch.setattr("deepresearch.retrieval.loaders.subprocess.run", fake_run)

    loaded = PDFDocumentLoader(
        fetcher=fake_fetcher,
        preserve_layout=False,
    ).load("https://example.com/report.pdf")

    assert loaded.content == "Citation recall definition"
    assert loaded.metadata["extraction_layout"] == "reading_order"


def test_tavily_provider_deduplicates_and_diversifies_domains():
    captured: dict[str, object] = {}

    def fake_transport(endpoint, headers, payload, timeout_seconds):
        captured.update(
            endpoint=endpoint,
            headers=headers,
            payload=payload,
            timeout_seconds=timeout_seconds,
        )
        return {
            "request_id": "req-123",
            "response_time": 0.42,
            "usage": {"credits": 1},
            "results": [
                {
                    "title": "A1",
                    "url": "https://a.example/report?utm_source=search",
                    "content": "snippet a1",
                    "raw_content": "full a1",
                    "score": 0.99,
                    "published_date": "2026-09-01",
                },
                {
                    "title": "A1 duplicate",
                    "url": "https://a.example/report#section",
                    "content": "duplicate",
                    "score": 0.98,
                },
                {
                    "title": "A2",
                    "url": "https://a.example/another",
                    "content": "snippet a2",
                    "score": 0.97,
                },
                {
                    "title": "B1",
                    "url": "https://b.example/report",
                    "content": "snippet b1",
                    "score": 0.90,
                    "published_date": "2026-10-01",
                },
                {
                    "title": "C1",
                    "url": "https://c.example/report",
                    "content": "snippet c1",
                    "score": 0.80,
                },
            ],
        }

    provider = TavilySearchProvider(
        api_key="test-secret-key",
        transport=fake_transport,
        diversify_domains=True,
    )

    docs, diagnostics = provider.search_with_diagnostics("test query", top_k=3)

    assert [doc.title for doc in docs] == ["A1", "B1", "C1"]
    assert docs[0].content == "full a1"
    assert docs[0].metadata["content_origin"] == "tavily_raw_content"
    assert docs[0].published_at == "2026-09-01"
    assert diagnostics["duplicate_url_count"] == 1
    assert diagnostics["unique_domain_count"] == 3
    assert diagnostics["published_date_count"] == 2
    assert diagnostics["request_id"] == "req-123"
    assert captured["payload"]["max_results"] == 6
    assert captured["payload"]["include_raw_content"] == "text"
    assert captured["headers"]["Authorization"] == "Bearer test-secret-key"
    assert "test-secret-key" not in repr(diagnostics)


def test_tavily_source_id_is_stable_after_url_normalization():
    responses = [
        {
            "results": [
                {
                    "title": "Result",
                    "url": "https://example.com/page?utm_campaign=x",
                    "content": "one",
                    "score": 0.8,
                }
            ]
        },
        {
            "results": [
                {
                    "title": "Result",
                    "url": "https://example.com/page#part",
                    "content": "two",
                    "score": 0.8,
                }
            ]
        },
    ]

    def fake_transport(endpoint, headers, payload, timeout_seconds):
        del endpoint, headers, payload, timeout_seconds
        return responses.pop(0)

    provider = TavilySearchProvider(api_key="test", transport=fake_transport)

    first = provider.search("query", top_k=1)[0]
    second = provider.search("query", top_k=1)[0]

    assert first.id == second.id
    assert first.url == second.url == "https://example.com/page"


def test_tavily_diagnostics_order_rfc2822_dates_chronologically():
    provider = TavilySearchProvider(
        api_key="test",
        transport=lambda *args: {
            "results": [
                {
                    "title": "Newer",
                    "url": "https://example.com/newer",
                    "content": "newer",
                    "published_date": "Mon, 02 Feb 2026 00:00:00 GMT",
                },
                {
                    "title": "Older",
                    "url": "https://example.com/older",
                    "content": "older",
                    "published_date": "Fri, 15 Dec 2023 00:00:00 GMT",
                },
            ]
        },
    )

    _, diagnostics = provider.search_with_diagnostics("query", top_k=2)

    assert diagnostics["oldest_published_at"] == "Fri, 15 Dec 2023 00:00:00 GMT"
    assert diagnostics["newest_published_at"] == "Mon, 02 Feb 2026 00:00:00 GMT"


def test_retriever_uses_provider_async_interface():
    provider = TavilySearchProvider(
        api_key="test",
        transport=lambda *args: {
            "results": [
                {
                    "title": "Async result",
                    "url": "https://example.com/async",
                    "content": "content",
                    "score": 0.7,
                }
            ]
        },
    )
    retriever = RetrieverAgent(provider, top_k=1)

    docs, diagnostics = asyncio.run(
        retriever.retrieve_with_diagnostics_async(SubTask(id="task", question="query"))
    )

    assert docs[0].title == "Async result"
    assert diagnostics["provider"] == "tavily"


def test_tavily_async_transport_allows_concurrent_searches():
    active = 0
    peak_active = 0

    async def fake_async_transport(endpoint, headers, payload, timeout_seconds):
        nonlocal active, peak_active
        del endpoint, headers, timeout_seconds
        active += 1
        peak_active = max(peak_active, active)
        await asyncio.sleep(0.01)
        active -= 1
        return {
            "results": [
                {
                    "title": payload["query"],
                    "url": f"https://example.com/{payload['query']}",
                    "content": "content",
                    "score": 0.8,
                }
            ]
        }

    provider = TavilySearchProvider(
        api_key="test",
        transport=lambda *args: (_ for _ in ()).throw(AssertionError("sync transport used")),
        async_transport=fake_async_transport,
    )

    async def run_both():
        return await asyncio.gather(
            provider.search_with_diagnostics_async("first", top_k=1),
            provider.search_with_diagnostics_async("second", top_k=1),
        )

    first, second = asyncio.run(run_both())

    assert first[0][0].title == "first"
    assert second[0][0].title == "second"
    assert peak_active == 2


def test_retrieval_smoke_omits_document_content():
    provider = TavilySearchProvider(
        api_key="test",
        transport=lambda *args: {
            "results": [
                {
                    "title": "Smoke source",
                    "url": "https://example.com/smoke",
                    "raw_content": "full private-to-pipeline body",
                    "score": 0.9,
                }
            ]
        },
    )

    result = asyncio.run(
        run_retrieval_smoke(
            RetrievalConfig(provider="tavily", top_k=1),
            "smoke query",
            provider=provider,
        )
    )

    assert result["documents"][0]["content_chars"] == len("full private-to-pipeline body")
    assert "content" not in result["documents"][0]


def test_tavily_factory_requires_configured_environment_variable(monkeypatch):
    monkeypatch.delenv("TEST_TAVILY_KEY", raising=False)
    config = RetrievalConfig(provider="tavily", api_key_env="TEST_TAVILY_KEY")

    with pytest.raises(ValueError, match="TEST_TAVILY_KEY"):
        create_search_provider(config)


def test_pipeline_records_web_source_metadata_and_diagnostics(tmp_path):
    provider = TavilySearchProvider(
        api_key="test",
        transport=lambda *args: {
            "request_id": "pipeline-request",
            "results": [
                {
                    "title": "Primary source",
                    "url": "https://research.example/report",
                    "raw_content": "该研究资料说明证据需要保留可追溯来源。",
                    "score": 0.91,
                    "published_date": "2026-10-01",
                }
            ],
        },
    )
    config = AppConfig(
        retrieval=RetrievalConfig(provider="tavily", top_k=1),
        runtime=RuntimeConfig(run_dir=tmp_path, max_subtasks=1),
    )

    report = ResearchPipeline(config, search_provider=provider).run(
        ResearchTask(query="如何保证研究证据可追溯？")
    )

    replay_path = next(tmp_path.glob(f"*_{report.run_id}_replay.json"))
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    node_result = next(iter(replay["node_results"].values()))
    source = node_result["docs"][0]
    diagnostics = node_result["diagnostics"]["retrieval"]

    assert source["source_type"] == "web"
    assert source["published_at"] == "2026-10-01"
    assert source["metadata"]["provider"] == "tavily"
    assert diagnostics["provider"] == "tavily"
    assert diagnostics["request_id"] == "pipeline-request"
    assert diagnostics["ranking"][0]["returned"] is True

"""Tavily Web Search provider，返回可追踪的 SourceDoc 和排序诊断。"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Awaitable, Callable
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import httpx

from deepresearch.config import RetrievalConfig
from deepresearch.retrieval.base import DocumentLoader, SearchProvider
from deepresearch.retrieval.loaders import HTMLDocumentLoader, PDFDocumentLoader, WebDocumentLoader
from deepresearch.retrieval.url_utils import (
    canonicalize_url,
    parse_published_datetime,
    url_domain,
)
from deepresearch.schema import SourceDoc


SearchTransport = Callable[[str, dict[str, str], dict[str, Any], float], dict[str, Any]]
AsyncSearchTransport = Callable[
    [str, dict[str, str], dict[str, Any], float],
    Awaitable[dict[str, Any]],
]


def _default_transport(
    endpoint: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout_seconds: float,
) -> dict[str, Any]:
    """用标准库调用搜索接口，使在线检索不增加运行时依赖。"""

    request = Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310 - endpoint 来自配置
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        # 不转储请求头，确保 Bearer token 不进入错误日志和 replay。
        body = exc.read(500).decode("utf-8", errors="replace")
        bearer = headers.get("Authorization", "").removeprefix("Bearer ")
        if bearer:
            body = body.replace(bearer, "[REDACTED]")
        raise RuntimeError(f"Tavily search failed with HTTP {exc.code}: {body}") from exc
    except (URLError, TimeoutError) as exc:
        raise RuntimeError(f"Tavily search request failed: {exc.reason if isinstance(exc, URLError) else exc}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("Tavily search returned invalid JSON.") from exc


async def _default_async_transport(
    endpoint: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout_seconds: float,
) -> dict[str, Any]:
    """使用原生异步 HTTP，避免多个 Web 检索节点互相阻塞。"""

    try:
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            response = await client.post(endpoint, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
    except httpx.HTTPStatusError as exc:
        body = exc.response.text[:500]
        bearer = headers.get("Authorization", "").removeprefix("Bearer ")
        if bearer:
            body = body.replace(bearer, "[REDACTED]")
        raise RuntimeError(
            f"Tavily search failed with HTTP {exc.response.status_code}: {body}"
        ) from exc
    except httpx.HTTPError as exc:
        raise RuntimeError(f"Tavily search request failed: {type(exc).__name__}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("Tavily search returned invalid JSON.") from exc
    if not isinstance(result, dict):
        raise RuntimeError("Tavily search returned a non-object JSON response.")
    return result


class TavilySearchProvider(SearchProvider):
    """Tavily API 适配器，包含 URL 去重、域名多样性和内容补全。"""

    def __init__(
        self,
        *,
        api_key: str,
        endpoint: str = "https://api.tavily.com/search",
        timeout_seconds: float = 30.0,
        search_depth: str = "basic",
        topic: str = "general",
        include_raw_content: bool = True,
        include_published_date: bool = True,
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
        language: str | None = None,
        diversify_domains: bool = True,
        fetch_full_content: bool = False,
        max_chars_per_doc: int = 12_000,
        transport: SearchTransport = _default_transport,
        async_transport: AsyncSearchTransport | None = None,
        document_loader: DocumentLoader | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("Tavily API key cannot be empty.")
        self._api_key = api_key
        self.endpoint = endpoint
        self.timeout_seconds = timeout_seconds
        self.search_depth = search_depth
        self.topic = topic
        self.include_raw_content = include_raw_content
        self.include_published_date = include_published_date
        self.include_domains = include_domains or []
        self.exclude_domains = exclude_domains or []
        self.language = language
        self.diversify_domains = diversify_domains
        self.fetch_full_content = fetch_full_content
        self.max_chars_per_doc = max_chars_per_doc
        self.transport = transport
        # 测试自定义同步 transport 时默认复用它；正常运行使用真正异步的 HTTP transport。
        self.async_transport = async_transport or (
            _default_async_transport if transport is _default_transport else None
        )
        self.document_loader = document_loader or WebDocumentLoader(
            html=HTMLDocumentLoader(
                timeout_seconds=timeout_seconds,
                max_chars=max_chars_per_doc,
            ),
            pdf=PDFDocumentLoader(
                timeout_seconds=timeout_seconds,
                max_chars=max_chars_per_doc,
            ),
        )

    @classmethod
    def from_config(cls, config: RetrievalConfig) -> "TavilySearchProvider":
        """从环境变量读取凭据，配置和异常中都不携带真实值。"""

        api_key = os.environ.get(config.api_key_env, "")
        if not api_key:
            raise ValueError(
                f"Retrieval provider 'tavily' requires environment variable {config.api_key_env}."
            )
        return cls(
            api_key=api_key,
            endpoint=config.endpoint,
            timeout_seconds=config.timeout_seconds,
            search_depth=config.search_depth,
            topic=config.topic,
            include_raw_content=config.include_raw_content,
            include_published_date=config.include_published_date,
            include_domains=config.include_domains,
            exclude_domains=config.exclude_domains,
            language=config.language,
            diversify_domains=config.diversify_domains,
            fetch_full_content=config.fetch_full_content,
            max_chars_per_doc=config.max_chars_per_doc,
        )

    def search(self, query: str, top_k: int = 5) -> list[SourceDoc]:
        docs, _ = self.search_with_diagnostics(query, top_k=top_k)
        return docs

    def search_with_diagnostics(
        self,
        query: str,
        top_k: int = 5,
    ) -> tuple[list[SourceDoc], dict[str, object]]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        payload = self._build_payload(query, top_k)
        response = self.transport(
            self.endpoint,
            self._headers(),
            payload,
            self.timeout_seconds,
        )
        return self._parse_response(query, top_k, response)

    async def search_with_diagnostics_async(
        self,
        query: str,
        top_k: int = 5,
    ) -> tuple[list[SourceDoc], dict[str, object]]:
        """Use native async HTTP in production while preserving injectable sync test transports."""

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        if self.async_transport is None:
            return self.search_with_diagnostics(query, top_k)
        response = await self.async_transport(
            self.endpoint,
            self._headers(),
            self._build_payload(query, top_k),
            self.timeout_seconds,
        )
        return self._parse_response(query, top_k, response)

    def _parse_response(
        self,
        query: str,
        top_k: int,
        response: dict[str, Any],
    ) -> tuple[list[SourceDoc], dict[str, object]]:
        """Normalize one provider response identically for sync and async transports."""

        raw_results = response.get("results", [])
        if not isinstance(raw_results, list):
            raise RuntimeError("Tavily search response field 'results' must be a list.")

        candidates: list[SourceDoc] = []
        ranking: list[dict[str, object]] = []
        seen_urls: set[str] = set()
        duplicate_count = 0
        load_errors: list[dict[str, str]] = []
        for provider_rank, item in enumerate(raw_results, start=1):
            if not isinstance(item, dict):
                continue
            raw_url = str(item.get("url") or "").strip()
            try:
                url = canonicalize_url(raw_url)
            except ValueError:
                ranking.append(
                    {
                        "rank": provider_rank,
                        "provider_rank": provider_rank,
                        "source_id": None,
                        "source_title": str(item.get("title") or ""),
                        "url": raw_url,
                        "domain": "",
                        "score": self._as_float(item.get("score")),
                        "published_at": item.get("published_date"),
                        "duplicate": False,
                        "invalid_url": True,
                        "returned": False,
                    }
                )
                continue
            duplicate = url in seen_urls
            if duplicate:
                duplicate_count += 1
            else:
                seen_urls.add(url)
                content = str(item.get("raw_content") or item.get("content") or "").strip()
                content_origin = "tavily_raw_content" if item.get("raw_content") else "tavily_snippet"
                loaded_metadata: dict[str, Any] = {}
                title = str(item.get("title") or url)
                # PDF 搜索结果的 raw_content 经常只有摘要；显式开启全文加载后，
                # PDF 无论是否带 raw_content 都尝试本地解析，失败时仍保留 provider 文本。
                is_pdf = urlsplit(url).path.lower().endswith(".pdf")
                if self.fetch_full_content and (is_pdf or not item.get("raw_content")):
                    try:
                        loaded = self.document_loader.load(url)
                        if loaded.content:
                            content = loaded.content
                            title = loaded.title or title
                            content_origin = "document_loader"
                            loaded_metadata = loaded.metadata
                    except Exception as exc:  # 单个来源加载失败不应中断整个搜索。
                        load_errors.append({"url": url, "error": f"{type(exc).__name__}: {exc}"})
                score = self._as_float(item.get("score"))
                source_id = self._source_id(url)
                candidates.append(
                    SourceDoc(
                        id=source_id,
                        title=title,
                        url=url,
                        content=content[: self.max_chars_per_doc],
                        source_type="web",
                        published_at=self._optional_string(item.get("published_date")),
                        score=score,
                        metadata={
                            "provider": "tavily",
                            "provider_rank": provider_rank,
                            "domain": url_domain(url),
                            "content_origin": content_origin,
                            **loaded_metadata,
                        },
                    )
                )
            ranking.append(
                {
                    "rank": provider_rank,
                    "provider_rank": provider_rank,
                    "source_id": None if duplicate else self._source_id(url),
                    "source_title": str(item.get("title") or url),
                    "url": url,
                    "domain": url_domain(url),
                    "score": self._as_float(item.get("score")),
                    "published_at": item.get("published_date"),
                    "duplicate": duplicate,
                    "invalid_url": False,
                    "returned": False,
                }
            )

        selected = self._select_candidates(candidates, top_k)
        selected_ids = {doc.id for doc in selected}
        for item in ranking:
            item["returned"] = item.get("source_id") in selected_ids
            item["selected_rank"] = next(
                (
                    selected_rank
                    for selected_rank, doc in enumerate(selected, start=1)
                    if doc.id == item.get("source_id")
                ),
                None,
            )

        published_dates = [doc.published_at for doc in candidates if doc.published_at]
        parsed_dates = [
            (parsed, value)
            for value in published_dates
            if (parsed := parse_published_datetime(value)) is not None
        ]
        diagnostics: dict[str, object] = {
            "provider": "tavily",
            "query": query,
            "search_depth": self.search_depth,
            "topic": self.topic,
            "candidate_docs_count": len(candidates),
            "returned_docs_count": len(selected),
            "raw_result_count": len(raw_results),
            "duplicate_url_count": duplicate_count,
            "unique_domain_count": len({url_domain(doc.url) for doc in candidates}),
            "published_date_count": len(published_dates),
            "oldest_published_at": min(parsed_dates)[1] if parsed_dates else None,
            "newest_published_at": max(parsed_dates)[1] if parsed_dates else None,
            "domain_diversity_enabled": self.diversify_domains,
            "ranking": ranking,
            "load_errors": load_errors,
            "request_id": response.get("request_id"),
            "response_time_seconds": response.get("response_time"),
            "usage": response.get("usage"),
        }
        return selected, diagnostics

    def _headers(self) -> dict[str, str]:
        """Build authorization headers without exposing them through diagnostics."""

        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def _build_payload(self, query: str, top_k: int) -> dict[str, Any]:
        # 多取一些候选，给 URL 去重和域名多样性留出空间；Tavily 当前上限为 20。
        payload: dict[str, Any] = {
            "query": query,
            "search_depth": self.search_depth,
            "topic": self.topic,
            "max_results": min(20, max(top_k, top_k * 2)),
            "include_answer": False,
            "include_raw_content": "text" if self.include_raw_content else False,
            "include_usage": True,
            "include_published_date": self.include_published_date,
        }
        if self.include_domains:
            payload["include_domains"] = self.include_domains
        if self.exclude_domains:
            payload["exclude_domains"] = self.exclude_domains
        if self.language:
            payload["language"] = self.language
        return payload

    def _select_candidates(self, candidates: list[SourceDoc], top_k: int) -> list[SourceDoc]:
        if not self.diversify_domains:
            return candidates[:top_k]
        selected: list[SourceDoc] = []
        selected_ids: set[str] = set()
        seen_domains: set[str] = set()
        # 第一轮优先保留不同域名，第二轮按 provider 相关性顺序补足数量。
        for doc in candidates:
            domain = url_domain(doc.url)
            if domain in seen_domains:
                continue
            selected.append(doc)
            selected_ids.add(doc.id)
            seen_domains.add(domain)
            if len(selected) == top_k:
                return selected
        for doc in candidates:
            if doc.id in selected_ids:
                continue
            selected.append(doc)
            if len(selected) == top_k:
                break
        return selected

    @staticmethod
    def _source_id(url: str) -> str:
        return f"web_{hashlib.sha256(url.encode('utf-8')).hexdigest()[:16]}"

    @staticmethod
    def _as_float(value: object) -> float | None:
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _optional_string(value: object) -> str | None:
        return str(value) if value not in (None, "") else None

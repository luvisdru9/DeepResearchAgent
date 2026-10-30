"""轻量 HTML/PDF 文档加载器；PDF 支持 pypdf 或系统 pdftotext。"""

from __future__ import annotations

import io
import ipaddress
import shutil
import subprocess
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Callable
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from deepresearch.retrieval.base import DocumentLoader, LoadedDocument


@dataclass(slots=True)
class FetchResult:
    """将网络读取与正文解析解耦，便于离线测试。"""

    body: bytes
    content_type: str
    final_url: str


Fetcher = Callable[[str, float, int], FetchResult]


class _ReadableHTMLParser(HTMLParser):
    """提取可见文本，并忽略脚本、样式和导航等噪声区域。"""

    _ignored_tags = {"script", "style", "noscript", "svg", "nav"}
    _block_tags = {"article", "br", "div", "h1", "h2", "h3", "li", "p", "section", "tr"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._ignored_depth = 0
        self._in_title = False
        self._title_parts: list[str] = []
        self._text_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        normalized = tag.lower()
        if normalized in self._ignored_tags:
            self._ignored_depth += 1
        if normalized == "title":
            self._in_title = True
        if normalized in self._block_tags and self._ignored_depth == 0:
            self._text_parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        normalized = tag.lower()
        if normalized == "title":
            self._in_title = False
        if normalized in self._ignored_tags and self._ignored_depth:
            self._ignored_depth -= 1
        if normalized in self._block_tags and self._ignored_depth == 0:
            self._text_parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._ignored_depth:
            return
        if self._in_title:
            self._title_parts.append(data)
        self._text_parts.append(data)

    def result(self, max_chars: int) -> LoadedDocument:
        title = " ".join(" ".join(self._title_parts).split()) or None
        lines = [" ".join(line.split()) for line in "".join(self._text_parts).splitlines()]
        content = "\n".join(line for line in lines if line).strip()[:max_chars]
        return LoadedDocument(content=content, title=title, content_type="text/html")


def _validate_remote_url(url: str) -> None:
    """拒绝明显的本机/私网 URL；直接抓取默认关闭，此处再做一层保护。"""

    parsed = urlsplit(url)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Document URL must use http or https.")
    hostname = parsed.hostname.lower()
    if hostname == "localhost" or hostname.endswith(".localhost"):
        raise ValueError("Local document URLs are not allowed.")
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        return
    if not address.is_global:
        raise ValueError("Private or local document URLs are not allowed.")


def default_fetcher(url: str, timeout_seconds: float, max_bytes: int) -> FetchResult:
    """使用标准库读取有限大小的远程文档，避免引入新的硬依赖。"""

    _validate_remote_url(url)
    request = Request(url, headers={"User-Agent": "DeepResearchAgent/0.1"})
    with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310 - URL 已校验
        body = response.read(max_bytes + 1)
        if len(body) > max_bytes:
            raise ValueError(f"Document exceeds the {max_bytes}-byte download limit.")
        content_type = response.headers.get_content_type()
        return FetchResult(
            body=body,
            content_type=content_type,
            final_url=response.geturl(),
        )


class HTMLDocumentLoader(DocumentLoader):
    """下载 HTML 并转换成适合 Reader 的纯文本。"""

    def __init__(
        self,
        *,
        timeout_seconds: float = 20.0,
        max_bytes: int = 5_000_000,
        max_chars: int = 12_000,
        fetcher: Fetcher = default_fetcher,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.max_bytes = max_bytes
        self.max_chars = max_chars
        self.fetcher = fetcher

    def load(self, url: str) -> LoadedDocument:
        fetched = self.fetcher(url, self.timeout_seconds, self.max_bytes)
        encoding = "utf-8"
        html = fetched.body.decode(encoding, errors="replace")
        parser = _ReadableHTMLParser()
        parser.feed(html)
        loaded = parser.result(self.max_chars)
        loaded.metadata["final_url"] = fetched.final_url
        return loaded


class PDFDocumentLoader(DocumentLoader):
    """提取 PDF 全文，并记录实际使用的解析后端。

    优先使用可选 Python ``pypdf``；未安装时回退到固定参数调用系统
    ``pdftotext``。后者不经过 shell，URL 和文档内容都不会进入命令行。
    """

    def __init__(
        self,
        *,
        timeout_seconds: float = 30.0,
        max_bytes: int = 20_000_000,
        max_chars: int = 12_000,
        preserve_layout: bool = True,
        fetcher: Fetcher = default_fetcher,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.max_bytes = max_bytes
        self.max_chars = max_chars
        # 双栏论文用 -layout 可保留视觉位置，但可能把左右栏交错成不连续的
        # 句子。证据抽取优先需要连续阅读顺序时，调用者可显式关闭版式保留。
        self.preserve_layout = preserve_layout
        self.fetcher = fetcher

    def load(self, url: str) -> LoadedDocument:
        backend = "pypdf"
        try:
            from pypdf import PdfReader
        except ImportError:
            PdfReader = None  # type: ignore[assignment,misc]
            backend = "pdftotext"
        pdftotext = shutil.which("pdftotext") if PdfReader is None else None
        if PdfReader is None and pdftotext is None:
            raise RuntimeError(
                "PDF extraction requires either the optional 'pypdf' package "
                "or the system 'pdftotext' executable."
            )
        fetched = self.fetcher(url, self.timeout_seconds, self.max_bytes)
        if PdfReader is not None:
            reader = PdfReader(io.BytesIO(fetched.body))
            content = "\n".join(
                (page.extract_text() or "").strip() for page in reader.pages
            ).strip()
            page_count = len(reader.pages)
        else:
            assert pdftotext is not None
            arguments = [pdftotext]
            if self.preserve_layout:
                arguments.append("-layout")
            arguments.extend(["-", "-"])
            completed = subprocess.run(  # noqa: S603 - fixed executable and arguments.
                arguments,
                input=fetched.body,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.timeout_seconds,
                check=False,
            )
            if completed.returncode != 0:
                detail = completed.stderr.decode("utf-8", errors="replace")[:300].strip()
                raise RuntimeError(
                    f"pdftotext failed with exit code {completed.returncode}: {detail}"
                )
            raw_content = completed.stdout.decode("utf-8", errors="replace")
            page_count = max(1, raw_content.count("\f")) if raw_content.strip() else 0
            content = raw_content.replace("\f", "\n\n").strip()
        if not content:
            raise RuntimeError("PDF parser returned no extractable text.")
        return LoadedDocument(
            content=content[: self.max_chars],
            content_type="application/pdf",
            metadata={
                "final_url": fetched.final_url,
                "page_count": page_count,
                "extraction_backend": backend,
                "extraction_layout": (
                    "physical" if backend == "pdftotext" and self.preserve_layout
                    else "reading_order"
                ),
                "extracted_chars": len(content),
                "content_truncated": len(content) > self.max_chars,
            },
        )


class WebDocumentLoader(DocumentLoader):
    """按 URL 后缀分发 HTML/PDF；搜索 provider 可把它作为内容补全器。"""

    def __init__(self, *, html: HTMLDocumentLoader, pdf: PDFDocumentLoader) -> None:
        self.html = html
        self.pdf = pdf

    def load(self, url: str) -> LoadedDocument:
        if urlsplit(url).path.lower().endswith(".pdf"):
            return self.pdf.load(url)
        return self.html.load(url)

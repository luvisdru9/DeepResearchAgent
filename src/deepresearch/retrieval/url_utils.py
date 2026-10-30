"""Web 检索使用的 URL 规范化与域名提取工具。"""

from __future__ import annotations

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


_TRACKING_PARAMETERS = {
    "access_token",
    "api_key",
    "fbclid",
    "gclid",
    "key",
    "mc_cid",
    "mc_eid",
    "ref",
    "signature",
    "source",
    "token",
}


def canonicalize_url(url: str) -> str:
    """移除 fragment 和常见追踪参数，使同一页面可稳定去重。"""

    parsed = urlsplit(url.strip())
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"Unsupported or invalid URL: {url!r}")

    scheme = parsed.scheme.lower()
    host = parsed.hostname.lower()
    port = parsed.port
    if port and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
        host = f"{host}:{port}"

    path = parsed.path or "/"
    if path != "/":
        path = path.rstrip("/")
    query_items = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if not key.lower().startswith("utm_") and key.lower() not in _TRACKING_PARAMETERS
    ]
    return urlunsplit((scheme, host, path, urlencode(sorted(query_items)), ""))


def url_domain(url: str) -> str:
    """返回用于多样性诊断的规范化主机名。"""

    return (urlsplit(url).hostname or "").lower()


def parse_published_datetime(value: str | None) -> datetime | None:
    """Parse ISO or HTTP/RFC 2822 dates returned by search providers."""

    if not value:
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        try:
            parsed = parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)

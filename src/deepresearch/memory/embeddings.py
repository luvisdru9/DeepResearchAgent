"""Replaceable embedding interfaces with a deterministic offline baseline."""

from __future__ import annotations

from abc import ABC, abstractmethod
import hashlib
import os
from typing import Any

import numpy as np

from deepresearch.text import tokenize_mixed_text


class EmbeddingProvider(ABC):
    """Minimal interface shared by local hashing and future learned embedding adapters."""

    provider_name: str
    model: str
    version: str
    dimensions: int

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return one normalized float32 vector per input text."""


class HashingEmbeddingProvider(EmbeddingProvider):
    """Dependency-light baseline using stable signed hashing over mixed-language terms."""

    provider_name = "local_hashing"
    model = "mixed-ngram-sha256"
    version = "1"

    def __init__(self, *, dimensions: int = 512) -> None:
        if dimensions < 64:
            raise ValueError("dimensions must be at least 64.")
        self.dimensions = dimensions

    def embed(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for row, text in enumerate(texts):
            for term in tokenize_mixed_text(text):
                digest = hashlib.sha256(term.encode("utf-8")).digest()
                bucket = int.from_bytes(digest[:8], "big") % self.dimensions
                matrix[row, bucket] += 1.0 if digest[8] % 2 == 0 else -1.0
            norm = float(np.linalg.norm(matrix[row]))
            if norm:
                matrix[row] /= norm
        return matrix


class OpenAICompatibleEmbeddingProvider(EmbeddingProvider):
    """Learned embedding adapter for OpenAI-compatible `/embeddings` endpoints."""

    provider_name = "openai_compatible"
    version = "1"

    def __init__(
        self,
        *,
        model: str,
        dimensions: int,
        api_key_env: str,
        base_url: str | None = None,
        timeout_seconds: float = 30.0,
        send_dimensions: bool = True,
        sdk_client: Any | None = None,
    ) -> None:
        if dimensions < 1:
            raise ValueError("dimensions must be positive.")
        self.model = model
        self.dimensions = dimensions
        self.api_key_env = api_key_env
        self.base_url = base_url
        self.timeout_seconds = timeout_seconds
        self.send_dimensions = send_dimensions
        self._client = sdk_client
        self._client_options: dict[str, Any] | None = None
        if sdk_client is None:
            api_key = os.environ.get(api_key_env)
            if not api_key:
                raise ValueError(
                    f"Environment variable '{api_key_env}' is required for embedding provider."
                )
            self._client_options = {
                "api_key": api_key,
                "timeout": timeout_seconds,
                "max_retries": 2,
            }
            if base_url:
                self._client_options["base_url"] = base_url

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimensions), dtype=np.float32)
        request: dict[str, Any] = {"model": self.model, "input": texts}
        if self.send_dimensions:
            request["dimensions"] = self.dimensions
        try:
            if self._client is not None:
                response = self._client.embeddings.create(**request)
            else:
                from openai import OpenAI

                with OpenAI(**(self._client_options or {})) as client:
                    response = client.embeddings.create(**request)
        except Exception as exc:
            secret = os.environ.get(self.api_key_env)
            message = str(exc).replace(secret, "[REDACTED]") if secret else str(exc)
            raise RuntimeError(f"Embedding request failed: {message}") from exc

        ordered = sorted(response.data, key=lambda item: int(getattr(item, "index", 0)))
        matrix = np.asarray([item.embedding for item in ordered], dtype=np.float32)
        if matrix.shape != (len(texts), self.dimensions):
            raise ValueError(
                f"Embedding endpoint returned shape {matrix.shape}; expected "
                f"{(len(texts), self.dimensions)}."
            )
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        # 零向量保持为零，正常向量统一归一化以便后续直接点积计算余弦相似度。
        return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms != 0)


def create_embedding_provider(
    name: str,
    *,
    dimensions: int,
    model: str = "text-embedding-3-small",
    base_url: str | None = None,
    api_key_env: str = "OPENAI_API_KEY",
    timeout_seconds: float = 30.0,
    send_dimensions: bool = True,
) -> EmbeddingProvider:
    """Create one configured embedding backend without coupling callers to its implementation."""

    if name == "hashing":
        return HashingEmbeddingProvider(dimensions=dimensions)
    if name in {"openai", "openai_compatible"}:
        return OpenAICompatibleEmbeddingProvider(
            model=model,
            dimensions=dimensions,
            api_key_env=api_key_env,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            send_dimensions=send_dimensions,
        )
    raise ValueError(
        f"Unsupported embedding provider '{name}'. Supported providers: "
        "hashing, openai, openai_compatible."
    )

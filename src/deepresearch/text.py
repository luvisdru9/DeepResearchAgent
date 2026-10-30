"""Dependency-free text helpers shared by retrieval and memory modules."""

from __future__ import annotations

import re


def tokenize_mixed_text(text: str) -> set[str]:
    """Tokenize Latin terms and short CJK n-grams without a heavyweight NLP dependency."""

    tokens: set[str] = set()
    for part in re.findall(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]+", text.lower()):
        tokens.add(part)
        if re.fullmatch(r"[\u4e00-\u9fff]+", part):
            for size in (2, 3, 4):
                if len(part) < size:
                    continue
                tokens.update(part[index : index + size] for index in range(len(part) - size + 1))
    return tokens

"""JSON extraction and Pydantic validation used by json_chat()."""

from __future__ import annotations

import json
import re
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from deepresearch.llm.errors import LLMResponseParseError, LLMResponseValidationError

T = TypeVar("T", bound=BaseModel)


def parse_json_model(content: str, response_model: type[T]) -> T:
    """Extract the first JSON value and validate it against a Pydantic model."""

    value = _extract_json_value(content)
    try:
        return response_model.model_validate(value)
    except ValidationError as exc:
        raise LLMResponseValidationError(
            f"Response JSON does not match {response_model.__name__}: {exc}"
        ) from exc


def _extract_json_value(content: str) -> object:
    """Parse plain or fenced JSON, with a bounded fallback for surrounding prose."""

    stripped = content.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
    candidate = fenced.group(1).strip() if fenced else stripped
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for index, character in enumerate(candidate):
        if character not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(candidate[index:])
            return value
        except json.JSONDecodeError:
            continue
    raise LLMResponseParseError("Model response does not contain valid JSON.")

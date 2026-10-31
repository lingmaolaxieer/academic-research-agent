"""Validate structured model output and retry only malformed output, boundedly."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Generic, TypeVar

from pydantic import BaseModel

from research_agent.config import settings

T = TypeVar("T", bound=BaseModel)
logger = logging.getLogger(__name__)


class StructuredOutputError(ValueError):
    def __init__(self, schema: str, attempts: int, finish_reason: str | None):
        self.attempts = attempts
        self.finish_reason = finish_reason
        super().__init__(
            f"Invalid {schema} structured output after {attempts} attempts "
            f"(finish_reason={finish_reason or 'unknown'})"
        )


@dataclass(frozen=True)
class StructuredResult(Generic[T]):
    value: T
    calls: int


def invoke_structured(model, schema: type[T], prompt: str, *, attempts: int | None = None):
    limit = attempts if attempts is not None else settings.structured_output_attempts
    if limit < 1:
        raise ValueError("attempts must be positive")
    runnable = model.with_structured_output(
        schema, method="function_calling", include_raw=True,
    )
    reason = None
    current_prompt = prompt
    for attempt in range(1, limit + 1):
        # Transport/authentication errors propagate to the classified graph policy.
        result = runnable.invoke(current_prompt)
        parsed = result.get("parsed") if isinstance(result, dict) else None
        if isinstance(parsed, schema):
            return StructuredResult(parsed, attempt)
        raw = result.get("raw") if isinstance(result, dict) else None
        metadata = getattr(raw, "response_metadata", {}) or {}
        reason = metadata.get("finish_reason")
        parsing_error = result.get("parsing_error") if isinstance(result, dict) else None
        tool_calls = getattr(raw, "tool_calls", ()) or ()
        tool_keys = [sorted(call["args"].keys()) for call in tool_calls
                     if isinstance(call, dict) and isinstance(call.get("args"), dict)]
        logger.warning(
            "structured_output_invalid schema=%s attempt=%d finish_reason=%s "
            "error_type=%s tool_calls=%d argument_keys=%s",
            schema.__name__, attempt, reason,
            type(parsing_error).__name__ if parsing_error else "none",
            len(tool_calls), tool_keys,
        )
        # Never log raw provider text, research documents, or credentials.
        current_prompt = prompt + (
            f"\nReturn the requested {schema.__name__} via the provided function. "
            "Include all required fields. Keep text concise to avoid truncation."
        )
    raise StructuredOutputError(schema.__name__, limit, reason)

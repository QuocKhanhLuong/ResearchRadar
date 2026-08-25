"""Usage telemetry for remote language-model calls.

Records token consumption per operation without ever storing credentials.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict

LLMOperation = Literal["paper_card_extraction", "ask_synthesis", "critic_review", "other"]


class LLMUsage(BaseModel):
    """One recorded language-model call with its token accounting."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: str
    model: str
    operation: LLMOperation
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    timestamp: datetime


class UsageSink(Protocol):
    """Receive completed language-model usage records."""

    def record(self, usage: LLMUsage) -> None:
        """Persist or aggregate one usage record."""


class InMemoryUsageSink:
    """Bounded in-process usage log for a single-user daemon."""

    def __init__(self, max_entries: int = 500) -> None:
        if max_entries < 0:
            raise ValueError("max_entries must not be negative")
        self._entries: deque[LLMUsage] = deque(maxlen=max_entries)

    def record(self, usage: LLMUsage) -> None:
        """Append a usage record, evicting the oldest entry past the bound."""

        self._entries.append(usage)

    @property
    def entries(self) -> list[LLMUsage]:
        """Return the currently retained usage records, oldest first."""

        return list(self._entries)

    def totals(self) -> dict[str, int]:
        """Sum retained input, output, and total token counts."""

        return {
            "input_tokens": sum(
                entry.input_tokens for entry in self._entries if entry.input_tokens is not None
            ),
            "output_tokens": sum(
                entry.output_tokens for entry in self._entries if entry.output_tokens is not None
            ),
            "total_tokens": sum(
                entry.total_tokens for entry in self._entries if entry.total_tokens is not None
            ),
        }


def parse_usage(
    payload: object,
    *,
    provider: str,
    model: str,
    operation: LLMOperation,
) -> LLMUsage | None:
    """Extract an :class:`LLMUsage` from a chat-completions response payload."""

    if not isinstance(payload, Mapping):
        return None
    usage = payload.get("usage")
    if not isinstance(usage, Mapping):
        return None
    return LLMUsage(
        provider=provider,
        model=model,
        operation=operation,
        input_tokens=_first_int(usage, "prompt_tokens", "input_tokens"),
        output_tokens=_first_int(usage, "completion_tokens", "output_tokens"),
        total_tokens=_first_int(usage, "total_tokens"),
        timestamp=datetime.now(UTC),
    )


def _first_int(mapping: Mapping[str, object], *keys: str) -> int | None:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, bool) or not isinstance(value, int):
            continue
        return value
    return None

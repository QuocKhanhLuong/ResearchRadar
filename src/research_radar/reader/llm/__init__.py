"""Provider-neutral structured LLM adapters for the reader."""

from research_radar.reader.llm.base import LLMMessage, LLMProvider, LLMResponseError
from research_radar.reader.llm.mock import MockLLMProvider
from research_radar.reader.llm.remote import RemoteLLMProvider

__all__ = [
    "LLMMessage",
    "LLMProvider",
    "LLMResponseError",
    "MockLLMProvider",
    "RemoteLLMProvider",
]

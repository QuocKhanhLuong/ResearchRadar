"""Tests for the chat-on-mention on_message wiring using fakes, never a gateway."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from types import SimpleNamespace

from research_radar.bot.client import (
    _CHUNK_LIMIT,
    _MENTION_USAGE_HINT,
    ResearchRadarBot,
    _application_intents,
    _chunk_message_text,
    create_bot,
)
from research_radar.config import Settings

BOT_USER_ID = 900_000_000_000_001
HUMAN_ID = 111_222_333_444_555
CHANNEL_ID = 700_100_200_300_400
MESSAGE_ID = 5_500_000


class FakeUser:
    def __init__(self, user_id: int, *, bot: bool = False) -> None:
        self.id = user_id
        self.bot = bot


class FakeChannel:
    def __init__(self, channel_id: int) -> None:
        self.id = channel_id
        self.sent: list[str] = []
        self.typing_entered = 0

    @asynccontextmanager
    async def typing(self):
        self.typing_entered += 1
        yield

    async def send(self, content: str, **_: object) -> None:
        self.sent.append(content)


class FakeMessage:
    def __init__(
        self,
        content: str,
        *,
        author: FakeUser | None = None,
        channel: FakeChannel | None = None,
        guild: object | None = object(),
    ) -> None:
        self.id = MESSAGE_ID
        self.content = content
        self.author = author if author is not None else FakeUser(HUMAN_ID)
        self.channel = channel if channel is not None else FakeChannel(CHANNEL_ID)
        self.guild = guild
        self.mentions: list[object] = []
        self.replied: list[str] = []

    async def reply(self, content: str, **_: object) -> None:
        self.replied.append(content)


@dataclass
class FakeChatService:
    """Records requests; replies with canned text or raises."""

    response_text: str = "an answer"
    error: Exception | None = None
    calls: list[object] = field(default_factory=list)

    async def chat(self, request: object) -> object:
        self.calls.append(request)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(text=self.response_text)


def build_bot(chat_service: FakeChatService | None) -> ResearchRadarBot:
    """Create an offline bot wired exactly as production composition would."""

    bot = create_bot(Settings(_env_file=None), chat_service=chat_service)
    bot._connection.user = FakeUser(BOT_USER_ID)
    return bot


def accepted_message(text: str = "what is new in retrieval?") -> tuple[FakeMessage, FakeChannel]:
    """Build a guild message that explicitly mentions the bot."""

    channel = FakeChannel(CHANNEL_ID)
    message = FakeMessage(f"<@{BOT_USER_ID}> {text}", channel=channel)
    message.mentions.append(FakeUser(BOT_USER_ID))
    return message, channel


async def test_rejected_message_triggers_zero_chat_calls_and_zero_replies() -> None:
    service = FakeChatService()
    bot = build_bot(service)
    rejected = FakeMessage("no mention here")

    await bot.on_message(rejected)
    await bot.close()

    assert service.calls == []
    assert rejected.replied == []
    assert rejected.channel.sent == []


async def test_empty_mention_text_replies_usage_hint_without_backend_call() -> None:
    service = FakeChatService()
    bot = build_bot(service)
    channel = FakeChannel(CHANNEL_ID)
    message = FakeMessage(f"<@{BOT_USER_ID}>", channel=channel)
    message.mentions.append(FakeUser(BOT_USER_ID))

    await bot.on_message(message)
    await bot.close()

    assert service.calls == []
    assert message.replied == [_MENTION_USAGE_HINT]
    assert channel.sent == []


async def test_accepted_mention_calls_chat_once_and_sends_single_reply() -> None:
    service = FakeChatService(response_text="here is the answer")
    bot = build_bot(service)
    message, channel = accepted_message()

    await bot.on_message(message)
    await bot.close()

    assert len(service.calls) == 1
    request = service.calls[0]
    assert request.text == "what is new in retrieval?"
    assert request.discord_user_id == str(HUMAN_ID)
    assert request.channel_id == str(CHANNEL_ID)
    assert request.message_id == str(MESSAGE_ID)
    assert channel.typing_entered == 1
    assert channel.sent == ["here is the answer"]
    assert message.replied == []


async def test_chat_failure_replies_safe_error_exactly_once() -> None:
    leaked_input = "user secret echoed back"
    error = RuntimeError(f"boom {leaked_input}")
    failing = FakeChatService(error=error)
    bot = build_bot(failing)
    message, channel = accepted_message()

    await bot.on_message(message)
    await bot.close()

    assert len(message.replied) == 1
    assert "Sorry" in message.replied[0]
    assert leaked_input not in message.replied[0]
    assert channel.sent == []


async def test_long_response_is_chunked_under_limit_in_order() -> None:
    paragraphs = [f"paragraph {index} " + "x" * 120 for index in range(30)]
    full_text = "\n\n".join(paragraphs)
    assert len(full_text) > _CHUNK_LIMIT
    service = FakeChatService(response_text=full_text)
    bot = build_bot(service)
    message, channel = accepted_message()

    await bot.on_message(message)
    await bot.close()

    assert len(channel.sent) > 1
    assert all(len(chunk) <= _CHUNK_LIMIT for chunk in channel.sent)
    assert "\n\n".join(channel.sent) == full_text


def test_chunk_helper_hard_cuts_text_without_boundaries() -> None:
    wall = "y" * (_CHUNK_LIMIT * 2 + 50)

    chunks = _chunk_message_text(wall)

    assert all(len(chunk) <= _CHUNK_LIMIT for chunk in chunks)
    assert "".join(chunks) == wall


def test_on_message_registered_only_with_a_chat_service() -> None:
    with_service = build_bot(FakeChatService())
    without_service = build_bot(None)

    assert with_service.on_message.__name__ == "_on_message"
    assert not hasattr(without_service, "on_message")


def test_intents_request_dm_messages_only_when_enabled() -> None:
    intents_on = _application_intents(SimpleNamespace(discord_dm_chat=True))
    intents_off = _application_intents(SimpleNamespace(discord_dm_chat=False))

    assert intents_on.guilds is True
    assert intents_on.guild_messages is True
    assert intents_on.dm_messages is True
    assert intents_on.message_content is False
    assert intents_off.dm_messages is False
    assert intents_off.message_content is False


def test_intents_default_to_dm_enabled_for_settings_missing_the_new_field() -> None:
    intents = _application_intents(Settings(_env_file=None))

    assert intents.dm_messages is True

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from research_radar.bot.notifications import DiscordNotificationSink
from research_radar.digest import ResearchDigest
from research_radar.storage import PaperSource, PendingNotification, StoredPaper, WatchTopic


def _topic() -> WatchTopic:
    now = datetime.now(UTC).replace(tzinfo=None)
    return WatchTopic("topic-1", "Medical imaging", "medical imaging", True, now, None, None)


def _notification() -> PendingNotification:
    now = datetime.now(UTC).replace(tzinfo=None)
    paper = StoredPaper(
        id="paper-1",
        canonical_key="doi:10.1/example",
        title="A useful paper",
        abstract=None,
        authors=[],
        publication_year=2026,
        venue=None,
        doi=None,
        url="https://papers.example/one",
        citation_count=None,
        primary_source="openalex",
        sources=(PaperSource("openalex", "W1", None, now),),
        first_discovered_at=now,
        created_at=now,
        updated_at=now,
    )
    return PendingNotification("topic-1", paper, 0.9, now, now)


@pytest.mark.asyncio
async def test_discord_notification_sink_sends_bounded_watch_update_without_mentions() -> None:
    channel = SimpleNamespace(send=AsyncMock())
    client = SimpleNamespace(get_channel=lambda channel_id: channel)
    sink = DiscordNotificationSink(42)
    sink.bind_client(client)  # type: ignore[arg-type]

    await sink.notify(_topic(), [_notification()])

    kwargs = channel.send.await_args.kwargs
    assert "ResearchRadar watch update" in kwargs["content"]
    assert "A useful paper" in kwargs["content"]
    assert kwargs["allowed_mentions"].everyone is False


@pytest.mark.asyncio
async def test_discord_notification_sink_fetches_missing_cached_channel_and_sends_digest() -> None:
    channel = SimpleNamespace(send=AsyncMock())
    client = SimpleNamespace(
        get_channel=lambda channel_id: None,
        fetch_channel=AsyncMock(return_value=channel),
    )
    sink = DiscordNotificationSink(42)
    sink.bind_client(client)  # type: ignore[arg-type]
    now = datetime.now(UTC).replace(tzinfo=None)
    digest = ResearchDigest(now, now, 0, (), ())

    await sink.notify_digest(digest)

    client.fetch_channel.assert_awaited_once_with(42)
    assert channel.send.await_args.kwargs["content"].startswith("ResearchRadar Daily Digest")


@pytest.mark.asyncio
async def test_unbound_notification_sink_refuses_delivery() -> None:
    sink = DiscordNotificationSink(42)

    with pytest.raises(RuntimeError, match="not bound"):
        await sink.notify(_topic(), [_notification()])

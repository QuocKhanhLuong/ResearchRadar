"""Discord delivery adapters for watch and digest domain notifications."""

from __future__ import annotations

from collections.abc import Sequence

import discord

from research_radar.digest import ResearchDigest
from research_radar.storage import PendingNotification, WatchTopic

_NO_MENTIONS = discord.AllowedMentions.none()
_MAX_CONTENT_CHARS = 2_000


class DiscordNotificationSink:
    """Deliver bounded private watch/digest updates to one configured channel.

    The domain services depend only on their notification protocols. This class
    is the only bridge that knows Discord channel APIs and can be bound after
    the bot factory has constructed its client.
    """

    def __init__(self, channel_id: int) -> None:
        self._channel_id = channel_id
        self._client: discord.Client | None = None

    def bind_client(self, client: discord.Client) -> None:
        """Bind the running Discord client before schedulers are started."""

        self._client = client

    async def notify(
        self,
        topic: WatchTopic,
        papers: Sequence[PendingNotification],
    ) -> None:
        """Send a compact bounded watch update after new papers are persisted."""

        lines = [f"ResearchRadar watch update: {_truncate(topic.name, 120)}"]
        for index, notification in enumerate(papers, start=1):
            paper = notification.paper
            year = f" ({paper.publication_year})" if paper.publication_year else ""
            lines.append(f"{index}. {_truncate(paper.title, 180)}{year}")
            if paper.url:
                lines.append(f"   {_truncate(paper.url, 220)}")
        await self._send("\n".join(lines))

    async def notify_digest(self, digest: ResearchDigest) -> None:
        """Send the shared bounded persisted-data digest renderer output."""

        await self._send(digest.render_text(max_characters=1_800))

    async def _send(self, content: str) -> None:
        client = self._client
        if client is None:
            raise RuntimeError("Discord notification sink is not bound to a client.")
        channel = client.get_channel(self._channel_id)
        if channel is None:
            channel = await client.fetch_channel(self._channel_id)
        send = getattr(channel, "send", None)
        if send is None:
            raise RuntimeError("Configured Discord notification destination is not messageable.")
        await send(
            content=_truncate(content, _MAX_CONTENT_CHARS),
            allowed_mentions=_NO_MENTIONS,
        )


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return f"{value[: max(1, limit - 1)].rstrip()}…"

"""Mention parsing and admission filtering for chat-on-mention Discord messages."""

from __future__ import annotations

import re
from dataclasses import dataclass

import discord

from research_radar.config import Settings

_WHITESPACE_RUN = re.compile(r"\s+")

REASON_SELF_MESSAGE = "self_message"
REASON_BOT_AUTHOR = "bot_author"
REASON_DM_DISABLED = "dm_disabled"
REASON_MENTION_DISABLED = "mention_disabled"
REASON_NO_MENTION = "no_mention"
REASON_CHANNEL_NOT_ALLOWED = "channel_not_allowed"
REASON_OWNER_ONLY = "owner_only"


@dataclass(frozen=True, slots=True)
class MentionAdmission:
    """Outcome of evaluating one inbound Discord message against the policy."""

    accepted: bool
    reason: str
    text: str = ""
    is_dm: bool = False


class MentionPolicy:
    """Decides which inbound Discord messages may reach the chat pipeline.

    Evaluation order is fixed and first-match-wins: ``self_message``,
    ``bot_author``, ``dm_disabled``, ``mention_disabled``, ``no_mention``,
    ``channel_not_allowed``, ``owner_only``. Direct messages bypass the mention
    requirement but still respect the owner gate. Settings introduced by this
    phase are read with their contract defaults so the policy composes against
    any ``Settings`` build in the parallel-worker integration.
    """

    def __init__(self, settings: Settings) -> None:
        """Store the settings consulted by the DM, mention, channel, and owner gates."""

        self._settings = settings

    def admit(self, message: discord.Message, *, bot_user_id: int) -> MentionAdmission:
        """Return whether the message is admitted, with cleaned text when accepted."""

        author = message.author
        is_dm = message.guild is None

        if author.id == bot_user_id:
            return MentionAdmission(False, REASON_SELF_MESSAGE, is_dm=is_dm)
        if author.bot:
            return MentionAdmission(False, REASON_BOT_AUTHOR, is_dm=is_dm)

        if is_dm:
            if not getattr(self._settings, "discord_dm_chat", True):
                return MentionAdmission(False, REASON_DM_DISABLED, is_dm=True)
        else:
            if not getattr(self._settings, "discord_chat_on_mention", True):
                return MentionAdmission(False, REASON_MENTION_DISABLED, is_dm=is_dm)
            if not _explicitly_mentioned(message, bot_user_id):
                return MentionAdmission(False, REASON_NO_MENTION, is_dm=is_dm)
            allowed_channels = tuple(
                getattr(self._settings, "discord_allowed_channel_ids", ()) or ()
            )
            if allowed_channels and message.channel.id not in allowed_channels:
                return MentionAdmission(False, REASON_CHANNEL_NOT_ALLOWED, is_dm=is_dm)

        owner_id = getattr(self._settings, "discord_owner_user_id", None)
        if owner_id is not None and author.id != owner_id:
            return MentionAdmission(False, REASON_OWNER_ONLY, is_dm=is_dm)

        return MentionAdmission(
            True,
            "accepted",
            text=_strip_bot_mentions(message, bot_user_id),
            is_dm=is_dm,
        )


def _explicitly_mentioned(message: discord.Message, bot_user_id: int) -> bool:
    """Return True only when the bot user itself is mentioned.

    ``@everyone``, ``@here``, and role mentions never count, even when the
    resolved role happens to include the bot.
    """

    for member in getattr(message, "mentions", ()) or ():
        if getattr(member, "id", None) == bot_user_id:
            return True

    content = message.content or ""
    plain_form = f"<@{bot_user_id}>"
    nickname_form = f"<@!{bot_user_id}>"
    return plain_form in content or nickname_form in content


def _strip_bot_mentions(message: discord.Message, bot_user_id: int) -> str:
    """Remove every bot mention form, collapse whitespace, and trim."""

    content = message.content or ""
    without_nickname_form = content.replace(f"<@!{bot_user_id}>", " ")
    without_plain_form = without_nickname_form.replace(f"<@{bot_user_id}>", " ")
    collapsed = _WHITESPACE_RUN.sub(" ", without_plain_form)
    return collapsed.strip()

"""Tests for MentionPolicy admission order, mention parsing, and text stripping."""

from __future__ import annotations

from dataclasses import dataclass

from research_radar.bot.mention import MentionPolicy

BOT_USER_ID = 900_000_000_000_001
OTHER_BOT_ID = 900_000_000_000_002
HUMAN_ID = 111_222_333_444_555
OWNER_ID = 123_456_789_012_345
CHANNEL_ID = 700_100_200_300_400


@dataclass(frozen=True)
class FakeSettings:
    """Duck-typed Settings carrying exactly the fields the policy reads."""

    discord_dm_chat: bool = True
    discord_chat_on_mention: bool = True
    discord_allowed_channel_ids: tuple[int, ...] = ()
    discord_owner_user_id: int | None = None


class FakeAuthor:
    def __init__(self, *, user_id: int, bot: bool = False) -> None:
        self.id = user_id
        self.bot = bot


class FakeChannel:
    def __init__(self, channel_id: int) -> None:
        self.id = channel_id


class FakeMessage:
    def __init__(
        self,
        *,
        content: str = "",
        author: FakeAuthor,
        guild: object | None = object(),
        channel: FakeChannel | None = None,
        mentions: list[object] | None = None,
    ) -> None:
        self.content = content
        self.author = author
        self.guild = guild
        self.channel = channel if channel is not None else FakeChannel(CHANNEL_ID)
        self.mentions = list(mentions if mentions is not None else [])


def make_policy(settings: FakeSettings | None = None) -> MentionPolicy:
    """Build a policy over fake settings without importing the real config."""

    return MentionPolicy(settings if settings is not None else FakeSettings())  # type: ignore[arg-type]


def human_message(content: str = "hello there", **kwargs: object) -> FakeMessage:
    """Build a non-bot author message with per-test overrides."""

    kwargs.setdefault("author", FakeAuthor(user_id=HUMAN_ID))
    return FakeMessage(content=content, **kwargs)  # type: ignore[arg-type]


def test_self_message_is_rejected_first() -> None:
    message = FakeMessage(
        content=f"<@{BOT_USER_ID}> hi",
        author=FakeAuthor(user_id=BOT_USER_ID),
    )

    result = make_policy().admit(message, bot_user_id=BOT_USER_ID)  # type: ignore[arg-type]

    assert result.accepted is False
    assert result.reason == "self_message"


def test_bot_author_message_is_rejected() -> None:
    message = human_message(
        f"<@{BOT_USER_ID}> hi",
        author=FakeAuthor(user_id=OTHER_BOT_ID, bot=True),
    )

    result = make_policy().admit(message, bot_user_id=BOT_USER_ID)  # type: ignore[arg-type]

    assert result.accepted is False
    assert result.reason == "bot_author"


def test_dm_when_dm_chat_disabled_is_rejected() -> None:
    message = human_message("plain question", guild=None)

    result = make_policy(FakeSettings(discord_dm_chat=False)).admit(
        message, bot_user_id=BOT_USER_ID
    )  # type: ignore[arg-type]

    assert result.accepted is False
    assert result.reason == "dm_disabled"
    assert result.is_dm is True


def test_dm_bypasses_mention_requirement_and_is_accepted() -> None:
    message = human_message("plain question", guild=None)

    result = make_policy().admit(message, bot_user_id=BOT_USER_ID)  # type: ignore[arg-type]

    assert result.accepted is True
    assert result.is_dm is True
    assert result.text == "plain question"


def test_guild_mention_disabled_wins_over_no_mention() -> None:
    message = human_message(f"<@{BOT_USER_ID}> hi")

    result = make_policy(FakeSettings(discord_chat_on_mention=False)).admit(
        message, bot_user_id=BOT_USER_ID
    )  # type: ignore[arg-type]

    assert result.accepted is False
    assert result.reason == "mention_disabled"
    assert result.is_dm is False


def test_guild_message_without_bot_mention_is_rejected() -> None:
    message = human_message("just chatting, no mention")

    result = make_policy().admit(message, bot_user_id=BOT_USER_ID)  # type: ignore[arg-type]

    assert result.accepted is False
    assert result.reason == "no_mention"


def test_everyone_here_and_role_mentions_do_not_count() -> None:
    everyone = human_message("@everyone what do you think")
    here = human_message("@here status update please")
    role_only = human_message("<@&987654321> ping the role", mentions=[{"id": 987654321}])
    role_including_bot_via_content = human_message(f"<@&{BOT_USER_ID}> pseudo role mention")

    policy = make_policy()
    for message in (everyone, here, role_only, role_including_bot_via_content):
        result = policy.admit(message, bot_user_id=BOT_USER_ID)  # type: ignore[arg-type]
        assert result.accepted is False, message.content
        assert result.reason == "no_mention"


def test_nickname_mention_form_counts_as_explicit() -> None:
    message = human_message(f"<@!{BOT_USER_ID}> hello bot")

    result = make_policy().admit(message, bot_user_id=BOT_USER_ID)  # type: ignore[arg-type]

    assert result.accepted is True
    assert result.text == "hello bot"


def test_mentions_list_entry_with_bot_id_counts_as_explicit() -> None:
    class MentionedUser:
        id = BOT_USER_ID

    message = human_message("please help", mentions=[MentionedUser()])

    result = make_policy().admit(message, bot_user_id=BOT_USER_ID)  # type: ignore[arg-type]

    assert result.accepted is True
    assert result.text == "please help"


def test_channel_not_allowed_rejects_mention_outside_allowlist() -> None:
    settings = FakeSettings(discord_allowed_channel_ids=(1, 2, CHANNEL_ID))
    outside = human_message(f"<@{BOT_USER_ID}> hi", channel=FakeChannel(channel_id=999))

    rejected = make_policy(settings).admit(outside, bot_user_id=BOT_USER_ID)  # type: ignore[arg-type]
    inside = human_message(f"<@{BOT_USER_ID}> hi")
    accepted = make_policy(settings).admit(inside, bot_user_id=BOT_USER_ID)  # type: ignore[arg-type]

    assert rejected.accepted is False
    assert rejected.reason == "channel_not_allowed"
    assert accepted.accepted is True


def test_no_mention_wins_over_channel_not_allowed_by_order() -> None:
    settings = FakeSettings(discord_allowed_channel_ids=(1, 2))
    message = human_message("no mention here", channel=FakeChannel(channel_id=999))

    result = make_policy(settings).admit(message, bot_user_id=BOT_USER_ID)  # type: ignore[arg-type]

    assert result.accepted is False
    assert result.reason == "no_mention"


def test_owner_only_applies_to_dm_even_without_mention_requirement() -> None:
    settings = FakeSettings(discord_owner_user_id=OWNER_ID)
    stranger_dm = human_message("plain question", guild=None)
    owner_dm = human_message("plain question", guild=None, author=FakeAuthor(user_id=OWNER_ID))

    rejected = make_policy(settings).admit(stranger_dm, bot_user_id=BOT_USER_ID)  # type: ignore[arg-type]
    accepted = make_policy(settings).admit(owner_dm, bot_user_id=BOT_USER_ID)  # type: ignore[arg-type]

    assert rejected.accepted is False
    assert rejected.reason == "owner_only"
    assert accepted.accepted is True
    assert accepted.text == "plain question"


def test_owner_only_applies_to_allowed_guild_mention() -> None:
    settings = FakeSettings(discord_owner_user_id=OWNER_ID)
    message = human_message(f"<@{BOT_USER_ID}> hi")

    result = make_policy(settings).admit(message, bot_user_id=BOT_USER_ID)  # type: ignore[arg-type]

    assert result.accepted is False
    assert result.reason == "owner_only"


def test_stripping_removes_both_forms_and_collapses_whitespace() -> None:
    content = f"  <@!{BOT_USER_ID}> What\tis \n\n solid-state   cooling? <@{BOT_USER_ID}> "
    message = human_message(content)

    result = make_policy().admit(message, bot_user_id=BOT_USER_ID)  # type: ignore[arg-type]

    assert result.accepted is True
    assert result.text == "What is solid-state cooling?"


def test_accepted_but_empty_text_yields_empty_string_text() -> None:
    only_mention_contents = (
        f"<@{BOT_USER_ID}>",
        f"<@!{BOT_USER_ID}>",
        f"<@{BOT_USER_ID}> <@!{BOT_USER_ID}>",
    )
    for content in only_mention_contents:
        message = human_message(content)

        result = make_policy().admit(message, bot_user_id=BOT_USER_ID)  # type: ignore[arg-type]

        assert result.accepted is True, content
        assert result.text == "", repr(content)


def test_other_users_mentions_are_left_in_text() -> None:
    other_id = BOT_USER_ID + 7
    message = human_message(f"<@{BOT_USER_ID}> ask <@{other_id}> too")

    result = make_policy().admit(message, bot_user_id=BOT_USER_ID)  # type: ignore[arg-type]

    assert result.accepted is True
    assert result.text == f"ask <@{other_id}> too"


def test_dm_bypasses_channel_allowlist() -> None:
    settings = FakeSettings(discord_allowed_channel_ids=(123, 456))
    dm_message = human_message("dm query", guild=None, channel=FakeChannel(channel_id=999))

    result = make_policy(settings).admit(dm_message, bot_user_id=BOT_USER_ID)  # type: ignore[arg-type]

    assert result.accepted is True
    assert result.is_dm is True
    assert result.text == "dm query"


def test_dm_empty_content_is_accepted_with_empty_text() -> None:
    dm_message = human_message("", guild=None)

    result = make_policy().admit(dm_message, bot_user_id=BOT_USER_ID)  # type: ignore[arg-type]

    assert result.accepted is True
    assert result.is_dm is True
    assert result.text == ""


def test_self_message_in_dm_is_rejected_as_self_message() -> None:
    message = FakeMessage(
        content="self message in DM",
        author=FakeAuthor(user_id=BOT_USER_ID),
        guild=None,
    )

    result = make_policy().admit(message, bot_user_id=BOT_USER_ID)  # type: ignore[arg-type]

    assert result.accepted is False
    assert result.reason == "self_message"
    assert result.is_dm is True


def test_bot_author_in_dm_is_rejected_as_bot_author() -> None:
    message = FakeMessage(
        content="other bot in DM",
        author=FakeAuthor(user_id=OTHER_BOT_ID, bot=True),
        guild=None,
    )

    result = make_policy().admit(message, bot_user_id=BOT_USER_ID)  # type: ignore[arg-type]

    assert result.accepted is False
    assert result.reason == "bot_author"
    assert result.is_dm is True


def test_owner_mention_in_allowed_guild_channel_is_accepted() -> None:
    settings = FakeSettings(
        discord_owner_user_id=OWNER_ID,
        discord_allowed_channel_ids=(CHANNEL_ID,),
    )
    message = human_message(
        f"<@{BOT_USER_ID}> query from owner",
        author=FakeAuthor(user_id=OWNER_ID),
        channel=FakeChannel(CHANNEL_ID),
    )

    result = make_policy(settings).admit(message, bot_user_id=BOT_USER_ID)  # type: ignore[arg-type]

    assert result.accepted is True
    assert result.text == "query from owner"
    assert result.is_dm is False


def test_empty_content_with_mention_in_mentions_list_is_accepted_empty() -> None:
    class MentionedUser:
        id = BOT_USER_ID

    message = FakeMessage(
        content="",
        author=FakeAuthor(user_id=HUMAN_ID),
        mentions=[MentionedUser()],
    )

    result = make_policy().admit(message, bot_user_id=BOT_USER_ID)  # type: ignore[arg-type]

    assert result.accepted is True
    assert result.text == ""


def test_mentions_list_with_nonmatching_or_malformed_members() -> None:
    class OtherUser:
        id = 999_888_777

    class MalformedMember:
        pass

    message = human_message(
        "regular message with no bot mention",
        mentions=[OtherUser(), MalformedMember()],
    )

    result = make_policy().admit(message, bot_user_id=BOT_USER_ID)  # type: ignore[arg-type]

    assert result.accepted is False
    assert result.reason == "no_mention"


def test_multiple_bot_mentions_and_surrounding_whitespace() -> None:
    content = f"<@{BOT_USER_ID}>\n\n  first part  <@!{BOT_USER_ID}>\tsecond part\n<@{BOT_USER_ID}>"
    message = human_message(content)

    result = make_policy().admit(message, bot_user_id=BOT_USER_ID)  # type: ignore[arg-type]

    assert result.accepted is True
    assert result.text == "first part second part"


def test_multiple_bot_mentions_and_mixed_forms_stripped() -> None:
    content = f"<@{BOT_USER_ID}>   alpha   <@!{BOT_USER_ID}>   beta  <@{BOT_USER_ID}>"
    message = human_message(content)

    result = make_policy().admit(message, bot_user_id=BOT_USER_ID)  # type: ignore[arg-type]

    assert result.accepted is True
    assert result.text == "alpha beta"


def test_policy_defaults_with_empty_settings_object() -> None:
    class BareSettings:
        pass

    policy = MentionPolicy(BareSettings())  # type: ignore[arg-type]
    guild_msg = human_message(f"<@{BOT_USER_ID}> hello")
    dm_msg = human_message("hello dm", guild=None)

    assert policy.admit(guild_msg, bot_user_id=BOT_USER_ID).accepted is True  # type: ignore[arg-type]
    assert policy.admit(dm_msg, bot_user_id=BOT_USER_ID).accepted is True  # type: ignore[arg-type]

"""Tests for research_radar.memory.secrets: detection and redaction."""

from __future__ import annotations

import base64

from research_radar.memory.secrets import REDACTED_PLACEHOLDER, contains_secret, redact_secrets


def _fake_discord_token() -> str:
    """Build a structurally valid but entirely synthetic Discord-style token."""
    first = base64.urlsafe_b64encode(b"555666777888999000").decode().rstrip("=")
    middle = "QmFzZTY0"
    third = "A1b2C3d4E5f6G7h8I9j0K1l2M3n4OpQr5S"
    return f"{first}.{middle}.{third}"


def _fake_key(prefix: str, body: str) -> str:
    """Attach a synthetic body to a vendor key prefix."""
    return prefix + body


class TestDiscordTokens:
    """Discord bot-token shapes."""

    def test_snowflake_decodable_token_detected(self) -> None:
        token = _fake_discord_token()
        assert contains_secret(f"the token is {token} please rotate")

    def test_mt_prefix_variant_detected(self) -> None:
        token = "MTAAAAAAAAAAAAAAAAAAAAAA.QmFzZTY0.A1b2C3d4E5f6G7h8I9j0K1l2M3n4OpQr5S"
        assert token.split(".")[0].startswith("MT")
        assert contains_secret(f"bot token: {token}")

    def test_redaction_removes_every_segment(self) -> None:
        token = _fake_discord_token()
        text = f"use {token} carefully"
        redacted = redact_secrets(text)
        assert token not in redacted
        for segment in token.split("."):
            assert segment not in redacted
        assert REDACTED_PLACEHOLDER in redacted
        assert "carefully" in redacted


class TestVendorApiKeys:
    """Vendor-prefixed API keys."""

    def test_openai_legacy_key(self) -> None:
        key = _fake_key("sk-", "fake000000000000000000key")
        assert contains_secret(f"key={key}")

    def test_anthropic_key(self) -> None:
        key = _fake_key("sk-ant-api03-", "fake000000000000000000key")
        assert contains_secret(key)

    def test_openai_project_key(self) -> None:
        key = _fake_key("sk-proj-", "fake000000000000000000key")
        assert contains_secret(key)

    def test_pinecone_key(self) -> None:
        key = _fake_key("pcsk_", "fake00000000000000key")
        assert contains_secret(key)

    def test_perplexity_key(self) -> None:
        key = _fake_key("pplx-", "fake00000000000000key")
        assert contains_secret(key)

    def test_groq_key(self) -> None:
        key = _fake_key("gsk_", "fake00000000000000key")
        assert contains_secret(key)

    def test_google_api_key(self) -> None:
        key = "AIzaFake0000000000000000000000000000000"
        assert contains_secret(key)

    def test_github_classic_pat(self) -> None:
        key = _fake_key("ghp_", "Fake0000000000000000000000000000000036")
        assert contains_secret(key)

    def test_github_fine_grained_pat(self) -> None:
        key = _fake_key("github_pat_", "Fake0000000_000000000_0000000")
        assert contains_secret(key)

    def test_slack_bot_token(self) -> None:
        key = _fake_key("xoxb-", "000000000000-fake-fake-fake")
        assert contains_secret(key)

    def test_pinecone_key_with_underscores_and_hyphens(self) -> None:
        key = _fake_key("pcsk_", "fake_key_0000_1111-2222")
        assert contains_secret(key)
        expected = f"pinecone key {REDACTED_PLACEHOLDER} active"
        assert redact_secrets(f"pinecone key {key} active") == expected

    def test_perplexity_key_with_hyphens(self) -> None:
        key = _fake_key("pplx-", "fake-key-0000-1111-2222")
        assert contains_secret(key)

    def test_groq_key_with_underscores(self) -> None:
        key = _fake_key("gsk_", "fake_key_0000_1111_2222")
        assert contains_secret(key)


class TestBearerHeaders:
    """Authorization headers and bare bearer tokens."""

    def test_authorization_header(self) -> None:
        assert contains_secret("Authorization: Bearer abcdefgh12345678")

    def test_lowercase_bare_bearer(self) -> None:
        assert contains_secret("bearer Zx9yWvUtSrQpNmLkJ8hG2fD5sA")

    def test_redacts_whole_header(self) -> None:
        redacted = redact_secrets("curl -H Authorization: Bearer abcdefgh12345678 /ping")
        assert "abcdefgh12345678" not in redacted


class TestAssignments:
    """Secret-shaped key/value assignments."""

    def test_api_key_equals(self) -> None:
        assert contains_secret("api_key=fakevalue12345")

    def test_uppercase_api_key_colon(self) -> None:
        assert contains_secret("API_KEY: fakevalue12345")

    def test_password_equals_with_spaces(self) -> None:
        assert contains_secret("password = fakevalue12345")

    def test_passwd_assignment(self) -> None:
        assert contains_secret("passwd='fakevalue12345'")

    def test_pwd_assignment(self) -> None:
        assert contains_secret('pwd="fakevalue12345"')

    def test_token_fat_arrow(self) -> None:
        assert contains_secret("auth config: token => fakevalue12345")

    def test_client_secret_assignment(self) -> None:
        assert contains_secret("client_secret=fakevalue12345")

    def test_access_key_assignment(self) -> None:
        assert contains_secret("access-key: fakevalue12345")

    def test_aws_secret_access_key_env_shape(self) -> None:
        assert contains_secret("AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLExx")

    def test_quoted_assignment_in_code_block(self) -> None:
        text = 'config = {"api_key": "supersecretkey123"}'
        assert contains_secret(text)
        assert "supersecretkey123" not in redact_secrets(text)

    def test_single_quoted_assignment(self) -> None:
        text = "export client_secret='verysecrettoken123'"
        assert contains_secret(text)
        assert "verysecrettoken123" not in redact_secrets(text)


class TestAwsCredentials:
    """AWS access key ids and secret values."""

    def test_akia_access_key_id(self) -> None:
        assert contains_secret("AKIAIOSFODNN7EXAMPLE is exposed")

    def test_asia_temporary_credentials(self) -> None:
        assert contains_secret("ASIAIOSFODNN7EXAMPLE in the logs")

    def test_aws_secret_value_detected(self) -> None:
        secret = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
        assert len(secret) == 40
        assert contains_secret(secret)

    def test_lowercase_hex_sha1_not_flagged(self) -> None:
        sha1 = "da39a3ee5e6b4b0d3255bfef95601890afd80709"
        assert not contains_secret(f"fixed in commit {sha1}")


class TestHighEntropyStrings:
    """Generic opaque high-entropy runs."""

    def test_random_base62_blob_detected(self) -> None:
        blob = "o3Kf9wPzqLm2XvR8sT5uBy1hNc4Ja6Dd"
        assert contains_secret(f"session {blob} expired")

    def test_prose_not_flagged(self) -> None:
        text = "I prefer concise answers with citations to the original papers."
        assert not contains_secret(text)

    def test_doi_not_flagged(self) -> None:
        text = "the DOI is https://doi.org/10.1038/s41586-024-07867-x for the paper"
        assert not contains_secret(text)

    def test_arxiv_ids_not_flagged(self) -> None:
        text = "compare arXiv:2401.12345v2 with arXiv:1706.03762v7 results"
        assert not contains_secret(text)

    def test_long_url_not_flagged(self) -> None:
        text = (
            "see https://example.com/publications/2026/long-path/to/paper.pdf"
            "?utm_source=newsletter&id=98765&read=full for the full text"
        )
        assert not contains_secret(text)

    def test_paper_title_not_flagged(self) -> None:
        text = (
            "Attention Is All You Need: Scaling Laws and Emergent Abilities "
            "of Large Language Models in Low-Resource Settings"
        )
        assert not contains_secret(text)


class TestRedactionBehaviour:
    """Redaction output guarantees."""

    def test_multiple_secrets_all_removed(self) -> None:
        token = _fake_discord_token()
        key = _fake_key("sk-", "fake000000000000000000key")
        text = f"token {token} and key {key} both leaked"
        redacted = redact_secrets(text)
        assert token not in redacted
        assert key not in redacted
        assert redacted.count(REDACTED_PLACEHOLDER) == 2
        assert "both leaked" in redacted

    def test_redaction_is_idempotent(self) -> None:
        text = "bearer abcdefgh12345678 and api_key=fakevalue12345"
        once = redact_secrets(text)
        twice = redact_secrets(once)
        assert once == twice

    def test_clean_text_unchanged(self) -> None:
        text = "I like turtles, especially hatchlings."
        assert redact_secrets(text) == text

    def test_empty_text(self) -> None:
        assert not contains_secret("")
        assert redact_secrets("") == ""

    def test_placeholder_itself_is_not_a_secret(self) -> None:
        assert not contains_secret(REDACTED_PLACEHOLDER)

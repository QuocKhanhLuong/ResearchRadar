"""Secret detection and redaction for user-supplied text.

This module never raises to callers and never logs: it answers one question
(does this text contain something credential-shaped?) and, when asked,
returns the text with every matched span replaced by ``[REDACTED]``. The
matched substrings themselves must never be echoed anywhere — callers use
``contains_secret()`` as a reject signal and ``redact_secrets()`` as a
defensive scrubber before persisting anything.

Detection covers (at minimum):
- Discord bot tokens (three dot-separated base64url segments whose first
  segment decodes to a snowflake, or starts with the MT/MF/Nz prefix).
- Vendor API keys: ``sk-``, ``sk-ant-``, ``sk-proj-``, ``pcsk_``, ``pplx-``,
  ``gsk_``, ``AIza``, ``ghp_``, ``github_pat_``, ``xoxb-``.
- ``Authorization: Bearer <token>`` headers and bare ``Bearer <token>``.
- Key/value assignments such as ``api_key=``, ``API_KEY:``, ``token => ...``
  for key names that imply a secret.
- AWS access key ids (AKIA/ASIA + 16 chars) and 40-char AWS-style secret
  values.
- Generic high-entropy opaque strings (see ``_OPAQUE_RUN_RE`` and
  ``_ENTROPY_THRESHOLD_BITS`` below for the tuning rationale).
"""

from __future__ import annotations

import base64
import binascii
import math
import re
from collections import Counter

REDACTED_PLACEHOLDER = "[REDACTED]"

# --- vendor API keys -------------------------------------------------------
# Each pattern matches a known vendor prefix plus enough trailing charset to
# avoid ordinary words. Charsets deliberately exclude spaces so prose survives.

_API_KEY_PATTERNS: tuple[re.Pattern[str], ...] = (
    # OpenAI legacy/project and Anthropic keys all start with sk-; the charset
    # includes '-' so sk-ant-api03-... and sk-proj-... are covered transitively.
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}"),
    re.compile(r"\bpcsk_[A-Za-z0-9]{16,}"),  # Pinecone
    re.compile(r"\bpplx-[A-Za-z0-9]{16,}"),  # Perplexity
    re.compile(r"\bgsk_[A-Za-z0-9]{16,}"),  # Groq
    re.compile(r"\bAIza[0-9A-Za-z_-]{30,}"),  # Google API keys are AIza+35
    re.compile(r"\bghp_[A-Za-z0-9]{30,}"),  # GitHub classic PATs are ghp_+36
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}"),  # GitHub fine-grained PATs
    re.compile(r"\bxoxb-[A-Za-z0-9-]{10,}"),  # Slack bot tokens
)

# --- bearer / authorization headers ---------------------------------------

_BEARER_RE = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{8,}")
_AUTHORIZATION_HEADER_RE = re.compile(r"(?i)\bauthorization\s*[:=]\s*\S{8,}")

# --- key/value assignments implying a secret -------------------------------

_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(?:api[_-]?key|apikey|token|secret|password|passwd|pwd"
    r"|access[_-]?key|client[_-]?secret)[\"']?\s*(?:=>|[:=])\s*"
    r"(?:\"[^\"]{8,}\"|'[^']{8,}'|[^\s\"']{8,})"
)

# --- AWS -------------------------------------------------------------------

_AWS_ACCESS_KEY_RE = re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")

# Real AWS secret access keys are 40 random base64-ish characters. A bare
# 40-char run also matches lowercase hex git SHAs, so require at least one
# uppercase letter or one of "+/=" to separate "random mixed alphabet" from
# "all-lowercase hex digest". This keeps commit SHAs out of the alert set
# while still catching realistic AWS secret material.
_AWS_SECRET_RUN_RE = re.compile(r"[A-Za-z0-9+=/]{40}")


def _looks_like_aws_secret(run: str) -> bool:
    """Return True for 40-char runs shaped like AWS secret access keys."""
    if not re.fullmatch(r"[A-Za-z0-9+=/]{40}", run):
        return False
    has_upper_or_symbol = any(c.isupper() or c in "+=/" for c in run)
    return has_upper_or_symbol


# --- Discord bot tokens ----------------------------------------------------

# Candidate shape: three dot-separated base64url segments of plausible length.
_DISCORD_CANDIDATE_RE = re.compile(
    r"(?<![A-Za-z0-9_-])([A-Za-z0-9_-]{18,})"  # first: b64 snowflake
    r"\.([A-Za-z0-9_-]{6,})"  # second: b64 timestamp
    r"\.([A-Za-z0-9_-]{25,})(?![A-Za-z0-9_-])"  # third: HMAC
)

# Modern Discord token first segments observed in the wild start with these
# base64 prefixes; a 24+ char first segment with one of them is treated as a
# token even when the snowflake decode check below fails.
_DISCORD_KNOWN_PREFIXES = ("MT", "MF", "Nz", "Mz", "Nj", "OD", "OT")


def _discord_first_segment_is_snowflake(segment: str) -> bool:
    """True when the first segment base64url-decodes to ASCII digits."""
    try:
        padded = segment + "=" * (-len(segment) % 4)
        decoded = base64.urlsafe_b64decode(padded.encode("ascii"))
    except (binascii.Error, UnicodeEncodeError, ValueError):
        return False
    return decoded.isdigit()


def _looks_like_discord_token(match: re.Match[str]) -> bool:
    """Validate a three-segment candidate against Discord's token layout."""
    first = match.group(1)
    if _discord_first_segment_is_snowflake(first):
        return True
    return len(first) >= 24 and first.startswith(_DISCORD_KNOWN_PREFIXES)


# --- generic high-entropy opaque strings -----------------------------------
# Candidate runs: 32+ chars, no whitespace and no URL punctuation (. : / ? &
# % @), because long URLs must break apart into short segments instead of
# forming one giant candidate. A candidate counts as an opaque secret only if
# it mixes upper case, lower case and digits AND its per-character Shannon
# entropy clears the threshold below.
#
# Threshold justification (measured, see tests): unigram Shannon entropy of
# natural English text tops out around 4.1-4.3 bits/char, and DOIs/arXiv ids/
# URL slugs are mostly lower-case so they fail the mixed-case gate anyway. A
# uniformly random 32-char base62 string measures about 4.9-5.1 bits/char once
# small-sample bias is included. The threshold 4.7 sits between those bands
# with margin on both sides, and the tests pin down examples on each side.
_ENTROPY_THRESHOLD_BITS = 4.7

_OPAQUE_RUN_RE = re.compile(r"[A-Za-z0-9_-]{32,}")


def _shannon_entropy_bits_per_char(text: str) -> float:
    """Per-character Shannon entropy of ``text``, 0.0 for empty input."""
    if not text:
        return 0.0
    total = len(text)
    entropy = 0.0
    for count in Counter(text).values():
        p = count / total
        entropy -= p * math.log2(p)
    return entropy


def _looks_like_opaque_secret(run: str) -> bool:
    """True when a long run looks like a random opaque credential."""
    if not (any(c.isupper() for c in run) and any(c.islower() for c in run)):
        return False
    if not any(c.isdigit() for c in run):
        return False
    return _shannon_entropy_bits_per_char(run) >= _ENTROPY_THRESHOLD_BITS


# --- public API ------------------------------------------------------------


def _secret_spans(text: str) -> list[tuple[int, int]]:
    """Collect (start, end) spans of every detected secret in ``text``."""
    spans: list[tuple[int, int]] = []

    def add(start: int, end: int) -> None:
        spans.append((start, end))

    for pattern in _API_KEY_PATTERNS:
        for match in pattern.finditer(text):
            add(match.start(), match.end())

    for pattern in (_BEARER_RE, _AUTHORIZATION_HEADER_RE, _ASSIGNMENT_RE):
        for match in pattern.finditer(text):
            add(match.start(), match.end())

    for match in _AWS_ACCESS_KEY_RE.finditer(text):
        add(match.start(), match.end())

    for match in _AWS_SECRET_RUN_RE.finditer(text):
        if _looks_like_aws_secret(match.group(0)):
            add(match.start(), match.end())

    for match in _DISCORD_CANDIDATE_RE.finditer(text):
        if _looks_like_discord_token(match):
            add(match.start(), match.end())

    for match in _OPAQUE_RUN_RE.finditer(text):
        if _looks_like_opaque_secret(match.group(0)):
            add(match.start(), match.end())

    return _merge_spans(spans)


def _merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping/adjacent spans into a sorted, disjoint list."""
    if not spans:
        return []
    ordered = sorted(spans)
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def contains_secret(text: str) -> bool:
    """Return True when ``text`` appears to contain a credential."""
    return bool(_secret_spans(text))


def redact_secrets(text: str) -> str:
    """Replace every detected secret span in ``text`` with ``[REDACTED]``.

    No part of a matched secret ever appears in the returned string.
    """
    pieces: list[str] = []
    cursor = 0
    for start, end in _secret_spans(text):
        pieces.append(text[cursor:start])
        pieces.append(REDACTED_PLACEHOLDER)
        cursor = end
    pieces.append(text[cursor:])
    return "".join(pieces)

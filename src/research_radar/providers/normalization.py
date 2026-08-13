"""Small provider-boundary normalization helpers."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit

DOI_PREFIX_PATTERN = re.compile(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", re.IGNORECASE)
ARXIV_VERSION_PATTERN = re.compile(r"v\d+$", re.IGNORECASE)
ARXIV_PREFIX_PATTERN = re.compile(r"^arxiv:\s*", re.IGNORECASE)
LEGACY_ARXIV_PATTERN = re.compile(
    r"^(?P<archive>[a-z][a-z.-]+)/(?P<number>\d{7})(?:v\d+)?$",
    re.IGNORECASE,
)


def normalize_doi(value: object) -> str | None:
    """Return a compact, case-insensitive DOI identity or ``None`` when absent."""

    if not isinstance(value, str):
        return None
    cleaned = DOI_PREFIX_PATTERN.sub("", value.strip()).strip().rstrip("/.,;")
    return cleaned.casefold() or None


def normalize_arxiv_id(value: object) -> str | None:
    """Extract an arXiv identifier while preserving legacy archive namespaces."""

    if not isinstance(value, str):
        return None
    cleaned = ARXIV_PREFIX_PATTERN.sub("", value.strip()).rstrip("/")
    parsed = urlsplit(cleaned)
    if parsed.scheme and parsed.netloc:
        path = parsed.path.strip("/")
        for marker in ("abs/", "pdf/"):
            if path.startswith(marker):
                path = path.removeprefix(marker)
                break
        cleaned = path
    cleaned = cleaned.removesuffix(".pdf").strip("/")
    legacy_match = LEGACY_ARXIV_PATTERN.fullmatch(cleaned)
    if legacy_match:
        cleaned = f"{legacy_match.group('archive').casefold()}/{legacy_match.group('number')}"
        return cleaned
    cleaned = cleaned.split("/")[-1]
    cleaned = ARXIV_VERSION_PATTERN.sub("", cleaned)
    return cleaned or None


def known_external_ids(values: Mapping[str, Any] | None) -> dict[str, str]:
    """Keep only stable, scalar external IDs from an upstream identifier mapping."""

    if not values:
        return {}
    allowed = {"arxiv", "doi", "pmid", "pmcid", "mag", "pubmed", "dblp", "acl"}
    normalized: dict[str, str] = {}
    for key, value in values.items():
        name = str(key).strip().casefold()
        if name not in allowed or not isinstance(value, (str, int)):
            continue
        identifier = str(value).strip()
        if not identifier:
            continue
        if name == "doi":
            identifier = normalize_doi(identifier) or ""
        elif name == "arxiv":
            identifier = normalize_arxiv_id(identifier) or ""
        if identifier:
            normalized[name] = identifier
    return normalized


def string_or_none(value: object) -> str | None:
    """Return a trimmed non-empty string without coercing structured values."""

    if not isinstance(value, str):
        return None
    return value.strip() or None


def integer_or_none(value: object) -> int | None:
    """Return a nonnegative integer from API scalars without propagating bad data."""

    if isinstance(value, bool):
        return None
    try:
        parsed = int(value) if value is not None else None
    except (TypeError, ValueError):
        return None
    return parsed if parsed is not None and parsed >= 0 else None

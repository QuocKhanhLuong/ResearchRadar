"""Canonical paper grouping over dedup identity that preserves provider provenance."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from research_radar.models.paper import Paper
from research_radar.research.dedup import group_papers, identity_keys, merge_papers

_KNOWN_PROVIDER_KEYS = ("openalex", "semantic_scholar", "arxiv", "s2", "doi", "pmid")


@dataclass(frozen=True, slots=True)
class CanonicalPaper:
    """One merged representative paper plus every provider identity behind it."""

    paper: Paper
    provider_ids: dict[str, str]
    contributing_papers: tuple[Paper, ...]


def canonicalize(papers: Iterable[Paper]) -> list[CanonicalPaper]:
    """Group papers with the shared dedup identity while keeping provider identity.

    Grouping is delegated to ``dedup.group_papers`` so the identity hierarchy
    lives in exactly one place. Only the provenance-preserving merge is added
    here: ``deduplicate`` returns one merged paper per group and discards which
    providers contributed, which is precisely what ingestion must retain.
    """

    return [_build_canonical_paper(group) for group in group_papers(papers)]


def identity_summary(paper: Paper) -> list[str]:
    """Return the ordered identity keys used for grouping, provenance, and debugging."""

    return identity_keys(paper)


def _build_canonical_paper(group: list[Paper]) -> CanonicalPaper:
    """Merge one duplicate group and guarantee per-provider external-id entries."""

    merged = merge_papers(group)
    external_ids = dict(merged.external_ids)
    provider_ids: dict[str, str] = {}
    for contributor in group:
        bare_id = contributor.id.split(":", maxsplit=1)[-1]
        external_ids.setdefault(contributor.source, bare_id)
        provider_ids.setdefault(contributor.source, bare_id)
    for name in _KNOWN_PROVIDER_KEYS:
        if value := external_ids.get(name):
            provider_ids.setdefault(name, value)
    paper = merged.model_copy(update={"external_ids": external_ids})
    return CanonicalPaper(
        paper=paper,
        provider_ids=provider_ids,
        contributing_papers=tuple(group),
    )

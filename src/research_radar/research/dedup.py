"""Deterministic paper identity and metadata merging."""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable

from research_radar.models.paper import Paper
from research_radar.providers.normalization import normalize_arxiv_id, normalize_doi

_PUNCTUATION_PATTERN = re.compile(r"[^\w\s]", flags=re.UNICODE)
_PROVIDER_PRIORITY = {"openalex": 0, "semantic_scholar": 1, "arxiv": 2}


def normalize_title(title: str) -> str:
    """Produce a conservative deterministic title identity for exact matching."""

    normalized = unicodedata.normalize("NFKC", title).casefold()
    return " ".join(_PUNCTUATION_PATTERN.sub(" ", normalized).split())


def identity_keys(paper: Paper) -> list[str]:
    """Return all stable keys in the stated DOI/arXiv/external/title priority family."""

    keys: list[str] = []
    if doi := normalize_doi(paper.doi or paper.external_ids.get("doi")):
        keys.append(f"doi:{doi}")
    if arxiv_id := normalize_arxiv_id(paper.external_ids.get("arxiv")):
        keys.append(f"arxiv:{arxiv_id}")
    for name, value in sorted(paper.external_ids.items()):
        if name.casefold() in {"doi", "arxiv"}:
            continue
        cleaned = str(value).strip().casefold()
        if cleaned:
            keys.append(f"external:{name.casefold()}:{cleaned}")
    title = normalize_title(paper.title)
    if len(title) >= 8:
        keys.append(f"title:{title}")
    return keys


def deduplicate(papers: Iterable[Paper]) -> list[Paper]:
    """Group any linked exact identities, then merge each group deterministically."""

    records = list(papers)
    parent = list(range(len(records)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    seen: dict[str, int] = {}
    for index, paper in enumerate(records):
        for key in identity_keys(paper):
            if key in seen:
                union(index, seen[key])
            else:
                seen[key] = index

    groups: dict[int, list[Paper]] = defaultdict(list)
    for index, paper in enumerate(records):
        groups[find(index)].append(paper)
    return [merge_papers(group) for _, group in sorted(groups.items(), key=lambda item: item[0])]


def merge_papers(papers: Iterable[Paper]) -> Paper:
    """Merge one duplicate group while retaining complementary useful metadata."""

    candidates = sorted(
        papers,
        key=lambda paper: (
            -_metadata_score(paper),
            _PROVIDER_PRIORITY.get(paper.source, 99),
            paper.id,
        ),
    )
    if not candidates:
        raise ValueError("Cannot merge an empty paper group")
    primary = candidates[0]
    external_ids: dict[str, str] = {}
    for paper in candidates:
        external_ids.update({key: value for key, value in paper.external_ids.items() if value})
        external_ids.setdefault(paper.source, paper.id.split(":", maxsplit=1)[-1])
    doi = next((normalize_doi(paper.doi) for paper in candidates if normalize_doi(paper.doi)), None)
    if doi:
        external_ids["doi"] = doi
    abstract = max(
        (paper.abstract for paper in candidates if paper.abstract), key=len, default=None
    )
    authors = max((paper.authors for paper in candidates), key=len, default=[])
    citation_count = max(
        (paper.citation_count for paper in candidates if paper.citation_count is not None),
        default=None,
    )
    publication_year = next(
        (paper.publication_year for paper in candidates if paper.publication_year), None
    )
    return Paper(
        id=primary.id,
        title=primary.title,
        abstract=abstract,
        authors=authors,
        publication_year=publication_year,
        venue=next((paper.venue for paper in candidates if paper.venue), None),
        doi=doi,
        url=next((paper.url for paper in candidates if paper.url), None),
        citation_count=citation_count,
        source=primary.source,
        external_ids=external_ids,
    )


def _metadata_score(paper: Paper) -> int:
    return sum(
        value is not None and value != [] and value != {}
        for value in (
            paper.abstract,
            paper.authors,
            paper.publication_year,
            paper.venue,
            paper.doi,
            paper.url,
            paper.citation_count,
        )
    )

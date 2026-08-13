"""Transparent deterministic V1 ranking with no embedding or LLM dependency."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime

from research_radar.models.paper import Paper
from research_radar.research.dedup import normalize_title

_TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)


@dataclass(frozen=True, slots=True)
class RankedPaper:
    """A normalized paper plus transparent score components for diagnostics."""

    paper: Paper
    score: float
    title_overlap: float
    abstract_overlap: float
    recency: float
    citations: float
    completeness: float


def rank_papers(
    query: str, papers: list[Paper], *, current_year: int | None = None
) -> list[RankedPaper]:
    """Rank by lexical relevance first, then modest recency/citation/completeness signals.

    The weight split is title 0.50, abstract 0.25, recency 0.15, citations
    0.07, and metadata completeness 0.03. Each component is clamped to 0..1.
    """

    year = current_year or datetime.now(UTC).year
    query_tokens = _tokens(query)
    ranked: list[RankedPaper] = []
    for paper in papers:
        title_overlap = _overlap(query_tokens, _tokens(paper.title))
        abstract_overlap = _overlap(query_tokens, _tokens(paper.abstract or ""))
        recency = _recency(paper.publication_year, year)
        citations = min(math.log1p(paper.citation_count or 0) / math.log1p(1000), 1.0)
        completeness = _completeness(paper)
        score = (
            0.50 * title_overlap
            + 0.25 * abstract_overlap
            + 0.15 * recency
            + 0.07 * citations
            + 0.03 * completeness
        )
        ranked.append(
            RankedPaper(
                paper=paper,
                score=score,
                title_overlap=title_overlap,
                abstract_overlap=abstract_overlap,
                recency=recency,
                citations=citations,
                completeness=completeness,
            )
        )
    return sorted(
        ranked,
        key=lambda result: (
            -result.score,
            -(result.paper.publication_year or 0),
            -(result.paper.citation_count or 0),
            normalize_title(result.paper.title),
        ),
    )


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_PATTERN.findall(normalize_title(text)))


def _overlap(query_tokens: set[str], candidate_tokens: set[str]) -> float:
    if not query_tokens or not candidate_tokens:
        return 0.0
    return len(query_tokens & candidate_tokens) / len(query_tokens)


def _recency(publication_year: int | None, current_year: int) -> float:
    if publication_year is None:
        return 0.0
    return max(0.0, min(1.0, 1 - (current_year - publication_year) / 10))


def _completeness(paper: Paper) -> float:
    fields = (
        paper.abstract,
        paper.authors,
        paper.publication_year,
        paper.venue,
        paper.doi,
        paper.url,
    )
    return sum(value is not None and value != [] for value in fields) / len(fields)

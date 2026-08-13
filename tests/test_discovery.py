from __future__ import annotations

import pytest

from research_radar.errors import ProviderUnavailableError
from research_radar.models import Paper
from research_radar.research.dedup import deduplicate, identity_keys, normalize_title
from research_radar.research.ranker import rank_papers
from research_radar.research.scout import ScoutService
from research_radar.research.service import ResearchService


def _paper(**overrides: object) -> Paper:
    values: dict[str, object] = {
        "id": "openalex:1",
        "title": "Visual Anomaly Detection",
        "abstract": "visual anomaly detection for medical scans",
        "authors": ["Ada"],
        "publication_year": 2025,
        "venue": "Venue",
        "doi": None,
        "url": "https://example.test/1",
        "citation_count": 1,
        "source": "openalex",
        "external_ids": {"openalex": "1"},
    }
    values.update(overrides)
    return Paper(**values)


def test_title_normalization_is_exact_and_conservative() -> None:
    assert normalize_title(" A—Paper:  MRI! ") == "a paper mri"
    assert "title:visual anomaly detection" in identity_keys(_paper())


def test_deduplication_prefers_doi_then_merges_complementary_data() -> None:
    left = _paper(doi="https://doi.org/10.1/X", abstract=None, citation_count=2)
    right = _paper(
        id="semantic_scholar:2",
        title="Different provider title",
        source="semantic_scholar",
        doi="10.1/x",
        external_ids={"semantic_scholar": "2", "arxiv": "2401.12345v2"},
        abstract="richer abstract",
        authors=["Ada", "Grace"],
        citation_count=7,
    )

    merged = deduplicate([left, right])

    assert len(merged) == 1
    assert merged[0].doi == "10.1/x"
    assert merged[0].abstract == "richer abstract"
    assert merged[0].authors == ["Ada", "Grace"]
    assert merged[0].citation_count == 7
    assert merged[0].external_ids["arxiv"] == "2401.12345v2"


def test_deduplication_merges_by_arxiv_and_title_only_when_exact() -> None:
    first = _paper(external_ids={"arxiv": "2401.5v2"}, doi=None)
    second = _paper(
        id="arxiv:2401.5",
        title="Other title",
        source="arxiv",
        external_ids={"arxiv": "2401.5"},
    )
    unrelated = _paper(
        id="openalex:3",
        title="Visual anomaly detectors",
        external_ids={"openalex": "3"},
    )

    assert len(deduplicate([first, second, unrelated])) == 2


def test_deduplication_does_not_title_merge_conflicting_strong_ids() -> None:
    first = _paper(doi="10.1000/first", external_ids={"openalex": "one"})
    second = _paper(
        id="semantic_scholar:two",
        source="semantic_scholar",
        doi="10.1000/second",
        external_ids={"semantic_scholar": "two"},
    )

    deduplicated = deduplicate([first, second])

    assert [paper.doi for paper in deduplicated] == ["10.1000/first", "10.1000/second"]


def test_deduplication_does_not_merge_legacy_arxiv_ids_from_different_archives() -> None:
    first = _paper(doi=None, external_ids={"arxiv": "hep-th/9901001v2"})
    second = _paper(
        id="arxiv:math/9901001",
        title="Different paper",
        source="arxiv",
        doi=None,
        external_ids={"arxiv": "math/9901001v2"},
    )

    assert len(deduplicate([first, second])) == 2


def test_ranking_keeps_relevance_ahead_of_old_citation_count() -> None:
    relevant = _paper(
        title="Visual anomaly detection in MRI", citation_count=3, publication_year=2025
    )
    old = _paper(
        id="openalex:old",
        title="General computer science",
        abstract="Unrelated topic",
        citation_count=100_000,
        publication_year=2000,
    )

    ranked = rank_papers("visual anomaly detection", [old, relevant], current_year=2026)

    assert [result.paper.id for result in ranked] == ["openalex:1", "openalex:old"]
    assert ranked[0].score > ranked[1].score


class _Provider:
    def __init__(self, name: str, outcome: list[Paper] | Exception) -> None:
        self.name = name
        self._outcome = outcome

    async def search(self, query: str, limit: int) -> list[Paper]:
        if isinstance(self._outcome, Exception):
            raise self._outcome
        return self._outcome


@pytest.mark.asyncio
async def test_scout_retains_partial_success_and_warning() -> None:
    result = await ScoutService(
        [_Provider("openalex", [_paper()]), _Provider("arxiv", RuntimeError("timeout"))]
    ).search("query", 5)

    assert len(result.papers) == 1
    assert result.warnings == ["arxiv was unavailable; results may be partial."]


@pytest.mark.asyncio
async def test_scout_raises_only_when_every_provider_fails() -> None:
    with pytest.raises(ProviderUnavailableError, match="All scholarly"):
        scout = ScoutService([_Provider("one", RuntimeError()), _Provider("two", RuntimeError())])
        await scout.search("query", 5)


@pytest.mark.asyncio
async def test_research_service_validates_and_returns_ranked_bounded_results() -> None:
    provider = _Provider("openalex", [_paper(), _paper(id="openalex:2")])
    service = ResearchService(ScoutService([provider]))

    result = await service.search(" visual anomaly ", limit=1)

    assert len(result.papers) == 1
    assert result.papers[0].paper.title == "Visual Anomaly Detection"
    with pytest.raises(ValueError, match="empty"):
        await service.search(" ")
    with pytest.raises(ValueError, match="between"):
        await service.search("query", limit=11)

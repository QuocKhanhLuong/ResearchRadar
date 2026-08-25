"""Unit tests for embedding providers and vector-text builders (no network)."""

from __future__ import annotations

import importlib
import math
import sys
from dataclasses import dataclass

import pytest

from research_radar.errors import ResearchRadarError
from research_radar.models.paper_card import EvidenceClaim, PaperCard, StructuredEvidence
from research_radar.semantic.base import EmbeddingError, EmbeddingProvider
from research_radar.semantic.embedding import (
    FakeEmbeddingProvider,
    LocalEmbeddingProvider,
    embedding_fingerprint,
    paper_card_vector_text,
    paper_vector_text,
    prepare_text,
)


@dataclass(frozen=True)
class _DuckPaper:
    """Minimal stand-in for Paper/StoredPaper in duck-typed tests."""

    title: str
    abstract: str | None


def _norm(vector: list[float]) -> float:
    return math.sqrt(sum(component * component for component in vector))


def test_fake_provider_is_deterministic_per_text() -> None:
    provider = FakeEmbeddingProvider()
    first = provider.embed_texts(["attention is all you need"])
    second = provider.embed_texts(["attention is all you need"])
    assert first == second


def test_fake_provider_distinguishes_texts() -> None:
    provider = FakeEmbeddingProvider()
    vectors = provider.embed_texts(["graph neural networks", "protein folding"])
    assert vectors[0] != vectors[1]


def test_fake_provider_vectors_have_dimension_and_unit_norm() -> None:
    provider = FakeEmbeddingProvider(dimension=12)
    vectors = provider.embed_texts(["a", "b", "a longer sentence about retrieval"])
    assert len(vectors) == 3
    for vector in vectors:
        assert len(vector) == 12
        assert all(isinstance(component, float) for component in vector)
        assert _norm(vector) == pytest.approx(1.0)


def test_embed_empty_sequence_returns_empty_list() -> None:
    provider = FakeEmbeddingProvider()
    assert provider.embed_texts([]) == []
    assert LocalEmbeddingProvider().embed_texts([]) == []


def test_embed_preserves_input_order() -> None:
    provider = FakeEmbeddingProvider()
    texts = ["first text", "second text", "third text"]
    batch = provider.embed_texts(texts)
    singles = [provider.embed_texts([text])[0] for text in texts]
    assert batch == singles


def test_non_positive_dimension_raises() -> None:
    with pytest.raises(EmbeddingError):
        FakeEmbeddingProvider(dimension=0)
    with pytest.raises(EmbeddingError):
        FakeEmbeddingProvider(dimension=-3)


def test_prepare_text_collapses_whitespace_and_normalizes() -> None:
    prepared = prepare_text("  Hello\t\tworld \n\n from　the ﬁeld  ")
    assert prepared == "Hello world from the field"
    assert "\n" not in prepared
    assert "\t" not in prepared
    assert "  " not in prepared


def test_prepare_text_is_idempotent() -> None:
    messy = "A\u00a0 title \t with\n\nnewlines   and　fullwidthＡchars"
    once = prepare_text(messy)
    assert prepare_text(once) == once


@pytest.mark.parametrize("max_chars", [5, 17, 50])
def test_prepare_text_truncates_on_word_boundary(max_chars: int) -> None:
    text = "alpha beta gamma delta epsilon zeta eta theta"
    prepared = prepare_text(text, max_chars=max_chars)
    assert len(prepared) <= max_chars
    assert prepared == prepare_text(prepared, max_chars=max_chars)


def test_paper_vector_text_uses_title_and_abstract_only() -> None:
    paper = _DuckPaper(title="Sparse Attention", abstract="We study sparsity.")
    text = paper_vector_text(paper)
    assert "Sparse Attention" in text
    assert "We study sparsity." in text


def test_paper_vector_text_with_none_abstract() -> None:
    text = paper_vector_text(_DuckPaper(title="No Abstract Here", abstract=None))
    assert text == "No Abstract Here"
    assert "None" not in text


def _card(**overrides: object) -> PaperCard:
    defaults: dict[str, object] = {"paper_id": "paper-1"}
    defaults.update(overrides)
    return PaperCard.model_validate(defaults)  # type: ignore[arg-type]


def test_paper_card_vector_text_is_deterministic_and_labeled() -> None:
    card = _card(
        problem="Long documents exceed context windows.",
        contributions=["A sparse retriever", "A new benchmark"],
        methods=["BM25 hybrid ranking"],
        tasks=[StructuredEvidence(value="question answering")],
        modalities=[StructuredEvidence(value="text")],
        datasets=["NQ-open", "TriviaQA"],
        metrics=["EM", "F1"],
        main_claims=[EvidenceClaim(claim="Sparse retrieval matches dense baselines.")],
        limitations=["Only English data."],
    )
    first = paper_card_vector_text(card)
    second = paper_card_vector_text(card.model_copy())
    assert first == second
    assert "Problem:" in first
    assert "Contributions: A sparse retriever; A new benchmark" in first
    assert "Methods: BM25 hybrid ranking" in first
    assert "Tasks: question answering" in first
    assert "Modalities: text" in first
    assert "Datasets: NQ-open; TriviaQA" in first
    assert "Metrics: EM; F1" in first
    assert "Claims: Sparse retrieval matches dense baselines." in first
    assert "Limitations: Only English data." in first


def test_empty_card_yields_short_string_without_raising() -> None:
    text = paper_card_vector_text(_card())
    assert isinstance(text, str)
    assert len(text) < 50


def test_embedding_fingerprint_known_entity_types() -> None:
    provider = FakeEmbeddingProvider(model_id="fake-embedding-v1")
    assert embedding_fingerprint(provider, entity_type="paper") == (
        "paper-v1",
        "fake-embedding-v1",
        8,
    )
    assert embedding_fingerprint(provider, entity_type="papercard") == (
        "papercard-v1",
        "fake-embedding-v1",
        8,
    )


def test_embedding_fingerprint_unknown_entity_type_raises() -> None:
    provider = FakeEmbeddingProvider()
    with pytest.raises(EmbeddingError):
        embedding_fingerprint(provider, entity_type="bogus")


def test_import_does_not_pull_sentence_transformers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "research_radar.semantic.embedding", raising=False)
    monkeypatch.delitem(sys.modules, "sentence_transformers", raising=False)
    importlib.import_module("research_radar.semantic.embedding")
    assert "sentence_transformers" not in sys.modules


def test_local_provider_is_lazy_and_surfaces_embedding_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = LocalEmbeddingProvider()

    def fail(self: LocalEmbeddingProvider) -> object:
        raise EmbeddingError("model unavailable")

    monkeypatch.setattr(provider, "_ensure_model", fail.__get__(provider))
    with pytest.raises(ResearchRadarError):
        provider.embed_texts(["some text"])


def test_fake_provider_satisfies_protocol() -> None:
    assert isinstance(FakeEmbeddingProvider(), EmbeddingProvider)

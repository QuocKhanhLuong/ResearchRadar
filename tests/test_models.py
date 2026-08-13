import pytest
from pydantic import ValidationError

from research_radar.models import EvidenceClaim, Paper, PaperCard, PaperDocument


def test_paper_normalizes_authors_ids_and_canonical_link() -> None:
    paper = Paper(
        id="W123",
        title="A Paper",
        authors=[" Ada Lovelace ", "", "Grace Hopper"],
        doi="10.1000/example",
        source="openalex",
        external_ids={" OpenAlex ": " W123 ", "empty": ""},
    )

    assert paper.authors == ["Ada Lovelace", "Grace Hopper"]
    assert paper.external_ids == {"openalex": "W123"}
    assert paper.canonical_link == "https://doi.org/10.1000/example"


def test_paper_rejects_missing_required_identity() -> None:
    with pytest.raises(ValidationError):
        Paper(id="", title="A Paper", source="openalex")


def test_document_and_paper_card_defaults_are_isolated() -> None:
    first = PaperCard(paper_id="paper-1")
    second = PaperCard(paper_id="paper-2")
    document = PaperDocument(title="P", full_text="text", sections={"Abstract": "summary"})

    first.contributions.append("new contribution")

    assert second.contributions == []
    assert document.section_names == {"abstract"}


def test_evidence_claim_accepts_unknown_evidence_location() -> None:
    card = PaperCard(
        paper_id="paper-1",
        main_claims=[EvidenceClaim(claim="Improves robustness")],
    )

    assert card.main_claims[0].source_section is None

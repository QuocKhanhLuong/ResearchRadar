"""Tests for the local filesystem artifact store."""

from __future__ import annotations

from pathlib import Path

import pytest

from research_radar.artifacts.base import ArtifactStoreError, sha256_hex
from research_radar.artifacts.local import LocalArtifactStore

PAPER_ID = "openalex:W123"
PDF_BYTES = b"%PDF-1.7 fake pdf payload"
TEXT_BYTES = b"extracted plain text"


@pytest.fixture()
def store(tmp_path: Path) -> LocalArtifactStore:
    """Return a store rooted at a fresh temporary directory."""

    return LocalArtifactStore(tmp_path)


def paper_dir(root: Path, paper_id: str) -> Path:
    """Return the on-disk directory holding one paper's artifacts."""

    return root / "papers" / paper_id


def test_put_then_read_round_trips(store: LocalArtifactStore) -> None:
    """Stored bytes come back unchanged with a consistent reference."""

    ref = store.put(paper_id=PAPER_ID, content=PDF_BYTES, artifact_type="pdf")

    assert ref.sha256 == sha256_hex(PDF_BYTES)
    assert ref.byte_size == len(PDF_BYTES)
    assert ref.mime_type == "application/pdf"
    assert ref.backend == "local"
    assert ref.source_url is None
    assert (
        store.read(paper_id=PAPER_ID, sha256=ref.sha256, artifact_type="pdf")
        == PDF_BYTES
    )


def test_duplicate_put_is_idempotent(store: LocalArtifactStore, tmp_path: Path) -> None:
    """Writing identical content twice yields one stable file."""

    first = store.put(paper_id=PAPER_ID, content=PDF_BYTES, artifact_type="pdf")
    second = store.put(paper_id=PAPER_ID, content=PDF_BYTES, artifact_type="pdf")

    assert first.object_key == second.object_key
    assert first.sha256 == second.sha256
    files = sorted(p.name for p in paper_dir(tmp_path, PAPER_ID).iterdir())
    assert files == [f"{first.sha256}.pdf"]


def test_existing_artifact_with_same_size_is_not_rewritten(
    store: LocalArtifactStore, tmp_path: Path
) -> None:
    """An idempotent re-put must not touch the already-stored file."""

    ref = store.put(paper_id=PAPER_ID, content=b"A" * 16, artifact_type="pdf")
    path = paper_dir(tmp_path, PAPER_ID) / f"{ref.sha256}.pdf"
    path.write_bytes(b"B" * 16)

    store.put(paper_id=PAPER_ID, content=b"A" * 16, artifact_type="pdf")

    assert path.read_bytes() == b"B" * 16


def test_new_content_adds_second_file_and_keeps_first(
    store: LocalArtifactStore, tmp_path: Path
) -> None:
    """Distinct content coexists; earlier artifacts stay readable."""

    first = store.put(paper_id=PAPER_ID, content=b"version-one", artifact_type="pdf")
    second = store.put(paper_id=PAPER_ID, content=b"version-two", artifact_type="pdf")

    assert first.sha256 != second.sha256
    assert len(list(paper_dir(tmp_path, PAPER_ID).iterdir())) == 2
    assert (
        store.read(paper_id=PAPER_ID, sha256=first.sha256, artifact_type="pdf")
        == b"version-one"
    )


def test_exists_before_and_after_put(store: LocalArtifactStore) -> None:
    """``exists`` flips from False to True once an artifact is stored."""

    kwargs = {
        "paper_id": PAPER_ID,
        "sha256": sha256_hex(PDF_BYTES),
        "artifact_type": "pdf",
    }

    assert store.exists(**kwargs) is False

    store.put(paper_id=PAPER_ID, content=PDF_BYTES, artifact_type="pdf")

    assert store.exists(**kwargs) is True


def test_resolve_before_and_after_put(store: LocalArtifactStore) -> None:
    """``resolve`` returns None when absent and a matching ref afterwards."""

    digest = sha256_hex(PDF_BYTES)

    assert (
        store.resolve(paper_id=PAPER_ID, sha256=digest, artifact_type="pdf") is None
    )

    ref = store.put(paper_id=PAPER_ID, content=PDF_BYTES, artifact_type="pdf")
    found = store.resolve(paper_id=PAPER_ID, sha256=digest, artifact_type="pdf")

    assert found is not None
    assert found.paper_id == PAPER_ID
    assert found.sha256 == digest
    assert found.artifact_type == "pdf"
    assert found.byte_size == len(PDF_BYTES)
    assert found.object_key == ref.object_key
    assert found.source_url is None


@pytest.mark.parametrize(
    "bad_id",
    ["../escape", "a/b", "a\\b", ".hidden", "", "x" * 129],
)
def test_unsafe_paper_ids_rejected(
    store: LocalArtifactStore, tmp_path: Path, bad_id: str
) -> None:
    """Path-traversal-shaped paper ids raise and create nothing anywhere."""

    before = sorted(p.name for p in tmp_path.parent.iterdir())

    with pytest.raises(ArtifactStoreError):
        store.put(paper_id=bad_id, content=PDF_BYTES, artifact_type="pdf")

    assert sorted(p.name for p in tmp_path.parent.iterdir()) == before
    assert list(tmp_path.iterdir()) == []


def test_bad_explicit_sha256_rejected(
    store: LocalArtifactStore, tmp_path: Path
) -> None:
    """A caller-supplied digest that is not 64 hex chars raises."""

    with pytest.raises(ArtifactStoreError):
        store.put(
            paper_id=PAPER_ID,
            content=PDF_BYTES,
            artifact_type="pdf",
            sha256="xyz",
        )

    assert list(tmp_path.rglob("*")) == []


def test_read_missing_artifact_error_hides_path(
    store: LocalArtifactStore, tmp_path: Path
) -> None:
    """A missing-artifact error mentions type + sha prefix, not the path."""

    digest = sha256_hex(PDF_BYTES)

    with pytest.raises(ArtifactStoreError) as excinfo:
        store.read(paper_id=PAPER_ID, sha256=digest, artifact_type="pdf")

    message = str(excinfo.value)
    assert str(tmp_path) not in message
    assert "pdf" in message
    assert digest[:12] in message


def test_derived_text_lands_next_to_source_pdf(
    store: LocalArtifactStore, tmp_path: Path
) -> None:
    """Text keyed by the source pdf digest sits beside it as ``.txt``."""

    pdf_ref = store.put(paper_id=PAPER_ID, content=PDF_BYTES, artifact_type="pdf")
    text_ref = store.put(
        paper_id=PAPER_ID,
        content=TEXT_BYTES,
        artifact_type="text",
        sha256=pdf_ref.sha256,
        source_url="https://example.org/paper.pdf",
    )
    expected = paper_dir(tmp_path, PAPER_ID) / f"{pdf_ref.sha256}.txt"

    assert expected.is_file()
    assert text_ref.object_key == expected.relative_to(tmp_path).as_posix()
    assert text_ref.source_url == "https://example.org/paper.pdf"
    assert text_ref.mime_type == "text/plain; charset=utf-8"
    assert text_ref.byte_size == len(TEXT_BYTES)
    assert (
        store.read(paper_id=PAPER_ID, sha256=pdf_ref.sha256, artifact_type="text")
        == TEXT_BYTES
    )


def test_no_temp_files_remain_after_puts(
    store: LocalArtifactStore, tmp_path: Path
) -> None:
    """Successful puts leave no temp-file debris behind."""

    pdf_ref = store.put(paper_id=PAPER_ID, content=PDF_BYTES, artifact_type="pdf")
    store.put(
        paper_id=PAPER_ID,
        content=TEXT_BYTES,
        artifact_type="text",
        sha256=pdf_ref.sha256,
    )

    names = [p.name for p in paper_dir(tmp_path, PAPER_ID).iterdir()]

    assert all("tmp" not in name for name in names)

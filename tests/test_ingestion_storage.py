from __future__ import annotations

import hashlib
from collections.abc import Iterator

import pytest

from research_radar.artifacts.base import ArtifactRef
from research_radar.models import Paper
from research_radar.storage import ResearchRepository
from research_radar.storage.database import Database, create_database
from research_radar.storage.ingestion_repository import IngestionRepository
from research_radar.storage.repositories import StorageError


@pytest.fixture
def database() -> Iterator[Database]:
    db = create_database("sqlite:///:memory:")
    db.initialize_schema()
    try:
        yield db
    finally:
        db.dispose()


@pytest.fixture
def paper_id(database: Database) -> str:
    paper = Paper(
        id="W123",
        title="Reliable Visual Anomaly Detection",
        source="openalex",
        doi="10.1000/example",
        abstract="A short abstract.",
        authors=["Ada Lovelace"],
        publication_year=2026,
        venue="Research Journal",
        citation_count=4,
        external_ids={"openalex": "W123"},
    )
    return ResearchRepository(database).upsert_merged_paper(paper)


@pytest.fixture
def repository(database: Database) -> IngestionRepository:
    return IngestionRepository(database)


def _sha(seed: str) -> str:
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


def _ref(
    paper_id: str,
    *,
    seed: str = "seed-a",
    artifact_type: str = "pdf",
) -> ArtifactRef:
    digest = _sha(seed)
    suffix = {"pdf": ".pdf", "text": ".txt", "sections": ".sections.json"}[artifact_type]
    return ArtifactRef(
        paper_id=paper_id,
        sha256=digest,
        artifact_type=artifact_type,  # type: ignore[arg-type]
        object_key=f"{paper_id}/{digest}{suffix}",
        mime_type="application/pdf" if artifact_type == "pdf" else "text/plain",
        byte_size=1024,
        source_url=f"https://example.org/{paper_id}{suffix}",
    )


def test_record_artifact_is_idempotent_on_identity(
    repository: IngestionRepository,
    paper_id: str,
) -> None:
    ref = _ref(paper_id)
    first = repository.record_artifact(ref)
    updated_ref = ref.model_copy(update={"object_key": f"{paper_id}/moved.pdf", "byte_size": 2048})
    second = repository.record_artifact(updated_ref)

    assert second.id == first.id
    assert repository.count_artifacts() == 1
    stored = repository.get_artifact(paper_id, ref.sha256, "pdf")
    assert stored is not None
    assert stored.object_key == f"{paper_id}/moved.pdf"
    assert stored.byte_size == 2048


def test_record_artifact_new_sha_creates_second_document_version(
    repository: IngestionRepository,
    paper_id: str,
) -> None:
    first = repository.record_artifact(_ref(paper_id, seed="v1"))
    second = repository.record_artifact(_ref(paper_id, seed="v2"))

    assert first.sha256 != second.sha256
    assert first.id != second.id
    assert repository.count_artifacts() == 2


def test_get_artifact_returns_none_for_unknown_sha(
    repository: IngestionRepository,
    paper_id: str,
) -> None:
    unknown_sha = _sha("never-stored")
    assert repository.get_artifact(paper_id, unknown_sha, "pdf") is None

    stored = repository.record_artifact(_ref(paper_id))

    fetched = repository.get_artifact(paper_id, stored.sha256, "pdf")
    assert fetched is not None
    assert fetched.id == stored.id
    assert fetched.object_key == stored.object_key


def test_list_artifacts_returns_all_types_ordered(
    repository: IngestionRepository,
    paper_id: str,
) -> None:
    for artifact_type in ("pdf", "text", "sections"):
        repository.record_artifact(
            _ref(paper_id, seed=f"multi-{artifact_type}", artifact_type=artifact_type)
        )

    records = repository.list_artifacts(paper_id)

    assert {record.artifact_type for record in records} == {"pdf", "text", "sections"}
    keys = [(record.created_at, record.artifact_type) for record in records]
    assert keys == sorted(keys)


def test_ingestion_run_lifecycle_running_then_completed(
    repository: IngestionRepository,
) -> None:
    run = repository.start_ingestion_run(
        query="  Transformers   for Science ",
        requested_limit=5,
        providers=["arxiv"],
    )
    assert run.status == "running"
    assert run.completed_at is None
    assert run.normalized_query == "transformers for science"
    assert run.discovered_count == 0
    assert run.canonical_count == 0

    completed = repository.complete_ingestion_run(
        run.id,
        discovered_count=12,
        canonical_count=7,
    )

    assert completed is not None
    assert completed.status == "completed"
    assert completed.discovered_count == 12
    assert completed.canonical_count == 7
    assert completed.completed_at is not None
    assert repository.count_ingestion_runs() == 1


def test_fail_ingestion_run_sets_status_and_safe_error(
    repository: IngestionRepository,
) -> None:
    run = repository.start_ingestion_run(query="graph nets", requested_limit=3, providers=["arxiv"])

    failed = repository.fail_ingestion_run(run.id, safe_error="provider arxiv timed out")

    assert failed is not None
    assert failed.status == "failed"
    assert failed.completed_at is not None
    assert failed.safe_error == "provider arxiv timed out"


def test_completion_and_failure_return_none_for_unknown_run(
    repository: IngestionRepository,
) -> None:
    assert (
        repository.complete_ingestion_run("missing-run", discovered_count=1, canonical_count=1)
        is None
    )
    assert repository.fail_ingestion_run("missing-run", safe_error="boom") is None


def test_provider_retrieval_round_trip_caps_external_ids(
    repository: IngestionRepository,
    paper_id: str,
) -> None:
    run = repository.start_ingestion_run(
        query="diffusion models", requested_limit=10, providers=["arxiv"]
    )
    retrieval = repository.record_provider_retrieval(
        run_id=run.id,
        provider="arxiv",
        query="diffusion models",
        external_ids=[f"2401.{index:04d}" for index in range(250)],
        status="ok",
        result_count=250,
    )
    assert len(retrieval.external_ids) == 200
    assert retrieval.external_ids[0] == "2401.0000"
    assert retrieval.external_ids[-1] == "2401.0199"

    retrievals = repository.list_provider_retrievals(run.id)

    assert [record.id for record in retrievals] == [retrieval.id]
    assert retrievals[0].status == "ok"
    assert retrievals[0].result_count == 250
    assert retrievals[0].safe_error is None


def test_list_recent_runs_orders_newest_first_and_clamps_limit(
    repository: IngestionRepository,
) -> None:
    runs = [
        repository.start_ingestion_run(
            query=f"query {number}", requested_limit=number + 1, providers=["arxiv"]
        )
        for number in range(3)
    ]

    recent = repository.list_recent_runs(limit=2)
    assert [run.id for run in recent] == [runs[2].id, runs[1].id]

    everything = repository.list_recent_runs(limit=1000)
    assert [run.id for run in everything] == [runs[2].id, runs[1].id, runs[0].id]

    clamped_to_one = repository.list_recent_runs(limit=0)
    assert [run.id for run in clamped_to_one] == [runs[2].id]


def test_safe_error_longer_than_500_chars_is_truncated(
    repository: IngestionRepository,
) -> None:
    run = repository.start_ingestion_run(
        query="long error", requested_limit=1, providers=["openalex"]
    )

    failed = repository.fail_ingestion_run(run.id, safe_error="x" * 600)

    assert failed is not None
    assert failed.safe_error is not None
    assert len(failed.safe_error) == 500


def test_record_artifact_unknown_paper_raises_storage_error(
    repository: IngestionRepository,
) -> None:
    with pytest.raises(StorageError):
        repository.record_artifact(_ref("00000000-0000-0000-0000-000000000000"))

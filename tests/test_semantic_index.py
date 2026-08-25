"""Unit tests for the semantic index backends; no network and no pinecone."""

from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from research_radar.semantic.base import (
    PAPER_SCHEMA_VERSION,
    EntityType,
    SemanticIndex,
    SemanticIndexError,
    SemanticRecord,
    entity_vector_id,
)
from research_radar.semantic.index import (
    DisabledSemanticIndex,
    FakeSemanticIndex,
    PineconeSemanticIndex,
)

ALLOWED_METADATA_KEYS = {
    "entity_id",
    "entity_type",
    "paper_id",
    "embedding_schema_version",
    "embedding_model",
}
API_KEY = "secret-key-do-not-leak"


def _record(
    paper_id: str,
    vector: list[float],
    *,
    entity_type: EntityType = "paper",
) -> SemanticRecord:
    """Build a minimal SemanticRecord for the given entity."""

    return SemanticRecord(
        entity_id=entity_vector_id(entity_type, paper_id),
        entity_type=entity_type,
        paper_id=paper_id,
        vector=vector,
        embedding_schema_version=PAPER_SCHEMA_VERSION,
        embedding_model="test-embedding-model",
    )


class _RecordingIndex:
    """Stub exposing upsert/query/delete while recording every call."""

    def __init__(
        self,
        *,
        response: Any = None,
        query_error: Exception | None = None,
        upsert_error: Exception | None = None,
    ) -> None:
        """Configure canned responses or errors for each operation."""

        self.upsert_calls: list[dict[str, Any]] = []
        self.query_calls: list[dict[str, Any]] = []
        self.delete_calls: list[dict[str, Any]] = []
        self._response = response
        self._query_error = query_error
        self._upsert_error = upsert_error

    def upsert(self, *, vectors: list[dict[str, Any]], namespace: str) -> dict[str, int]:
        """Record the batch, then raise if configured to fail."""

        self.upsert_calls.append({"vectors": vectors, "namespace": namespace})
        if self._upsert_error is not None:
            raise self._upsert_error
        return {"upserted_count": len(vectors)}

    def query(self, **kwargs: Any) -> Any:
        """Record the query, then raise or return the canned response."""

        self.query_calls.append(kwargs)
        if self._query_error is not None:
            raise self._query_error
        return self._response

    def delete(self, *, ids: list[str], namespace: str) -> dict[str, Any]:
        """Record the delete request."""

        self.delete_calls.append({"ids": ids, "namespace": namespace})
        return {}


class _StubPineconeClient:
    """Stand-in for the Pinecone client returning a fixed index handle."""

    def __init__(self, index: _RecordingIndex) -> None:
        """Hold the single index handle this client hands out."""

        self._index = index

    def Index(self, name: str) -> _RecordingIndex:
        """Return the preconfigured index regardless of the requested name."""

        return self._index


def _match(match_id: str, metadata: dict[str, Any], score: float) -> Any:
    """Build one object-style Pinecone match."""

    return SimpleNamespace(id=match_id, score=score, metadata=metadata)


def _pinecone(index_stub: _RecordingIndex) -> PineconeSemanticIndex:
    """Create a PineconeSemanticIndex with an injected stub client."""

    return PineconeSemanticIndex(
        api_key=API_KEY,
        index_name="research-radar-test",
        namespace="research-radar",
        client=_StubPineconeClient(index_stub),
    )


def test_disabled_index_is_a_total_noop() -> None:
    """Every DisabledSemanticIndex operation succeeds and does nothing."""

    index = DisabledSemanticIndex()
    assert index.available is False
    assert index.upsert([_record("p1", [1.0, 0.0])]) == 0
    assert index.search([1.0, 0.0]) == []
    assert index.delete(["paper:p1"]) == 0
    status = index.status()
    assert status.backend == "disabled"
    assert status.available is False


def test_fake_search_returns_nearest_record_first() -> None:
    """The most similar stored vector is ranked first."""

    index = FakeSemanticIndex()
    near = _record("near", [1.0, 0.0])
    far = _record("far", [0.0, 1.0])
    assert index.upsert([near, far]) == 2
    hits = index.search([1.0, 0.0])
    assert hits[0].entity_id == "paper:near"
    assert hits[0].paper_id == "near"
    assert hits[0].score == pytest.approx(1.0)
    assert hits[-1].score < hits[0].score


def test_fake_upsert_is_idempotent_per_entity() -> None:
    """Upserting the same entity twice leaves one record and returns 1."""

    index = FakeSemanticIndex()
    record = _record("p1", [1.0, 0.0])
    assert index.upsert([record]) == 1
    assert index.upsert([record]) == 1
    hits = index.search([1.0, 0.0], top_k=10)
    assert len(hits) == 1
    assert hits[0].entity_id == "paper:p1"


def test_fake_search_filters_by_entity_type() -> None:
    """An entity_type filter excludes records of the other type."""

    index = FakeSemanticIndex()
    paper = _record("p1", [1.0, 0.0], entity_type="paper")
    card = _record("c1", [1.0, 0.0], entity_type="papercard")
    index.upsert([paper, card])
    hits = index.search([1.0, 0.0], entity_type="papercard")
    assert [hit.entity_id for hit in hits] == ["papercard:c1"]


def test_fake_search_respects_top_k() -> None:
    """top_k bounds how many candidates come back."""

    index = FakeSemanticIndex()
    index.upsert(
        [
            _record("a", [1.0, 0.0]),
            _record("b", [0.0, 1.0]),
            _record("c", [-1.0, 0.0]),
        ]
    )
    hits = index.search([1.0, 0.0], top_k=2)
    assert len(hits) == 2


def test_fake_ties_break_by_entity_id_over_two_runs() -> None:
    """Identical scores order deterministically by entity_id ascending."""

    index = FakeSemanticIndex()
    index.upsert([_record("zzz", [1.0, 0.0]), _record("aaa", [1.0, 0.0])])
    first_run = [hit.entity_id for hit in index.search([1.0, 0.0])]
    second_run = [hit.entity_id for hit in index.search([1.0, 0.0])]
    expected = ["paper:aaa", "paper:zzz"]
    assert first_run == expected
    assert second_run == expected


def test_fake_delete_removes_and_reports_unknown_ids() -> None:
    """Delete drops present entities; unknown IDs report zero removals."""

    index = FakeSemanticIndex()
    index.upsert([_record("p1", [1.0, 0.0])])
    assert index.delete(["paper:p1"]) == 1
    assert index.search([1.0, 0.0]) == []
    assert index.delete(["paper:p1"]) == 0
    assert index.delete(["paper:never-existed"]) == 0


def test_fake_unavailable_search_returns_empty() -> None:
    """An unavailable fake degrades to an empty result list."""

    index = FakeSemanticIndex(available=False)
    assert index.available is False
    assert index.search([1.0, 0.0]) == []
    index.set_available(True)
    assert index.search([1.0, 0.0]) == []


def test_fake_wrong_length_vector_raises() -> None:
    """A dimension mismatch raises SemanticIndexError instead of guessing."""

    index = FakeSemanticIndex()
    index.upsert([_record("p1", [1.0, 0.0, 0.0])])
    with pytest.raises(SemanticIndexError):
        index.search([1.0, 0.0])


def test_pinecone_upsert_sends_compact_metadata() -> None:
    """Upsert payloads carry deterministic IDs and only allowed metadata keys."""

    stub = _RecordingIndex()
    index = _pinecone(stub)
    records = [_record("p1", [1.0, 0.0]), _record("c1", [0.5, 0.5], entity_type="papercard")]
    sent = index.upsert(records)
    assert sent == 2
    call = stub.upsert_calls[0]
    assert call["namespace"] == "research-radar"
    ids = [vector["id"] for vector in call["vectors"]]
    assert ids == ["paper:p1", "papercard:c1"]
    first = call["vectors"][0]
    assert first["values"] == [1.0, 0.0]
    assert set(first["metadata"].keys()) == ALLOWED_METADATA_KEYS
    assert set(call["vectors"][1]["metadata"].keys()) == ALLOWED_METADATA_KEYS
    assert "title" not in first["metadata"]
    assert "abstract" not in first["metadata"]
    assert "claim" not in first["metadata"]


def test_pinecone_upsert_batches_at_100() -> None:
    """250 records produce exactly three batches of 100/100/50."""

    stub = _RecordingIndex()
    index = _pinecone(stub)
    records = [_record(f"p{i:03d}", [float(i), 1.0]) for i in range(250)]
    assert index.upsert(records) == 250
    sizes = [len(call["vectors"]) for call in stub.upsert_calls]
    assert sizes == [100, 100, 50]
    assert all(call["namespace"] == "research-radar" for call in stub.upsert_calls)
    sent_ids = [
        vector["id"] for call in stub.upsert_calls for vector in call["vectors"]
    ]
    assert len(sent_ids) == 250
    assert len(set(sent_ids)) == 250


def test_pinecone_search_maps_matches_to_hits() -> None:
    """Matches become SemanticHits and the query forwards the right arguments."""

    stub = _RecordingIndex(
        response=SimpleNamespace(
            matches=[
                _match(
                    "paper:p1",
                    {
                        "entity_id": "paper:p1",
                        "entity_type": "paper",
                        "paper_id": "p1",
                    },
                    0.92,
                ),
                _match(
                    "papercard:c1",
                    {
                        "entity_id": "papercard:c1",
                        "entity_type": "papercard",
                        "paper_id": "c1",
                    },
                    0.41,
                ),
            ]
        )
    )
    index = _pinecone(stub)
    hits = index.search([0.1, 0.2], top_k=5)
    assert [(hit.entity_id, hit.entity_type, hit.paper_id, hit.score) for hit in hits] == [
        ("paper:p1", "paper", "p1", 0.92),
        ("papercard:c1", "papercard", "c1", 0.41),
    ]
    kwargs = stub.query_calls[0]
    assert kwargs["vector"] == [0.1, 0.2]
    assert kwargs["top_k"] == 5
    assert kwargs["namespace"] == "research-radar"
    assert kwargs["include_metadata"] is True
    assert kwargs["filter"] is None
    index.search([0.1, 0.2], top_k=3, entity_type="papercard")
    assert stub.query_calls[1]["filter"] == {"entity_type": "papercard"}


def test_pinecone_search_skips_malformed_match() -> None:
    """Matches missing required metadata are skipped without raising."""

    stub = _RecordingIndex(
        response=SimpleNamespace(
            matches=[
                _match("bad1", {"entity_type": "paper"}, 0.9),
                _match("bad2", {"paper_id": "p2", "entity_type": "alien"}, 0.8),
                _match(
                    "good1",
                    {
                        "entity_id": "paper:p1",
                        "entity_type": "paper",
                        "paper_id": "p1",
                    },
                    0.7,
                ),
            ]
        )
    )
    index = _pinecone(stub)
    hits = index.search([1.0, 0.0])
    assert [hit.entity_id for hit in hits] == ["paper:p1"]


def test_pinecone_query_failure_degrades_and_marks_unavailable() -> None:
    """A query outage returns no hits and flips availability off."""

    stub = _RecordingIndex(query_error=RuntimeError("boom"))
    index = _pinecone(stub)
    hits = index.search([1.0, 0.0])
    assert hits == []
    assert index.available is False
    status = index.status()
    assert status.available is False
    assert API_KEY not in (status.detail or "")
    assert "http" not in (status.detail or "").lower()


def test_pinecone_upsert_failure_returns_zero() -> None:
    """An upsert outage reports zero writes without raising."""

    stub = _RecordingIndex(upsert_error=RuntimeError("boom"))
    index = _pinecone(stub)
    assert index.upsert([_record("p1", [1.0, 0.0])]) == 0
    assert index.available is False


def test_pinecone_makes_exactly_one_attempt() -> None:
    """Failures never trigger retries: one client call per operation."""

    stub = _RecordingIndex(query_error=RuntimeError("boom"), upsert_error=RuntimeError("boom"))
    index = _pinecone(stub)
    index.search([1.0, 0.0])
    index.upsert([_record("p1", [1.0, 0.0])])
    assert len(stub.query_calls) == 1
    assert len(stub.upsert_calls) == 1


def test_pinecone_rejects_blank_credentials() -> None:
    """Empty api_key or index_name is rejected at construction time."""

    with pytest.raises(ValueError):
        PineconeSemanticIndex(api_key="", index_name="some-index")
    with pytest.raises(ValueError):
        PineconeSemanticIndex(api_key=API_KEY, index_name="")


def test_backends_satisfy_semantic_index_protocol() -> None:
    """All three backends pass the runtime-checkable protocol check."""

    assert isinstance(DisabledSemanticIndex(), SemanticIndex)
    assert isinstance(FakeSemanticIndex(), SemanticIndex)
    assert isinstance(_pinecone(_RecordingIndex()), SemanticIndex)


def test_importing_module_does_not_require_pinecone() -> None:
    """The lazy import keeps pinecone out of sys.modules at import time."""

    sys.modules.pop("research_radar.semantic.index", None)
    module = importlib.import_module("research_radar.semantic.index")
    assert module is not None
    assert "pinecone" not in sys.modules

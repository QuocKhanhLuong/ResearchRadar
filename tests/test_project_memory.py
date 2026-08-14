"""Unit tests for Project Memory V1 persistence and relations."""

from __future__ import annotations

import pytest

from research_radar.models.gap import CandidateGap, GapProvenance
from research_radar.models.paper import Paper
from research_radar.storage.database import Database
from research_radar.storage.repositories import ResearchRepository


@pytest.fixture
def repo(tmp_path_factory: object) -> ResearchRepository:
    db_file = tmp_path_factory.mktemp("db") / "test_project.db"  # type: ignore[attr-defined]
    db = Database.create(f"sqlite:///{db_file}")
    db.initialize_schema()
    repository = ResearchRepository(db)
    yield repository
    db.dispose()


def test_project_create_read_update_list(repo: ResearchRepository) -> None:
    proj = repo.create_project(
        name="MRI Reconstruction",
        goal="Preserve small lesions during reconstruction",
        keywords=["MRI", "reconstruction", "lesion"],
        constraints=["Max memory 16GB"],
        hypotheses=["Spectral regularization reduces noise"],
    )

    assert proj.id is not None
    assert proj.name == "MRI Reconstruction"
    assert proj.goal == "Preserve small lesions during reconstruction"
    assert "MRI" in proj.keywords
    assert "Spectral regularization reduces noise" in proj.hypotheses

    # Read by name & by ID
    fetched_by_name = repo.get_project("MRI Reconstruction")
    fetched_by_id = repo.get_project(proj.id)

    assert fetched_by_name is not None
    assert fetched_by_id is not None
    assert fetched_by_name.id == proj.id
    assert fetched_by_id.name == proj.name

    # List projects
    all_projects = repo.list_projects()
    assert len(all_projects) == 1
    assert all_projects[0].id == proj.id

    # Update project
    updated = repo.update_project(
        proj.id,
        description="Detailed project for lesion preservation",
        rejected_ideas=["Pure MSE loss"],
    )
    assert updated.description == "Detailed project for lesion preservation"
    assert "Pure MSE loss" in updated.rejected_ideas


def test_project_paper_and_gap_relationships(repo: ResearchRepository) -> None:
    p1 = Paper(id="p1", title="Fast MRI Paper", source="arxiv")
    p2 = Paper(id="p2", title="Lesion Segmentation Paper", source="arxiv")

    id1 = repo.upsert_merged_paper(p1)
    id2 = repo.upsert_merged_paper(p2)

    proj1 = repo.create_project(name="Project Alpha", goal="Goal Alpha")
    proj2 = repo.create_project(name="Project Beta", goal="Goal Beta")

    # Same paper (p1) attached to multiple projects
    link1 = repo.add_paper_to_project(proj1.id, id1, relation="seed")
    link2 = repo.add_paper_to_project(proj2.id, id1, relation="background")
    repo.add_paper_to_project(proj1.id, id2, relation="supporting")

    assert link1.relation == "seed"
    assert link2.relation == "background"

    p1_links = repo.list_project_papers(proj1.id)
    p2_links = repo.list_project_papers(proj2.id)

    assert len(p1_links) == 2
    assert len(p2_links) == 1
    assert p2_links[0].paper_id == id1

    # Attach CandidateGap to project
    cand = CandidateGap(
        id="cand-101",
        title="Sample Gap",
        description="Sample Desc",
        gap_type="explicit",
        research_question="Sample RQ?",
        supporting_papers=[id1],
        evidence_count=1,
        search_scope="Scope",
        provenance=GapProvenance(retrievals=[], corpus_paper_ids=[id1], corpus_description="Desc"),
        created_at=proj1.created_at,
    )
    repo.save_candidate(cand)

    gap_link = repo.add_gap_to_project(proj1.id, cand.id, status="active")
    assert gap_link.candidate_id == cand.id
    assert gap_link.status == "active"

    project_gaps = repo.list_project_gaps(proj1.id)
    assert len(project_gaps) == 1
    assert project_gaps[0].candidate_id == cand.id

"""Automated tests verifying seed_demo_research_memory and smoke_test_research_radar."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research_radar.storage.database import Database  # noqa: E402
from research_radar.storage.repositories import ResearchRepository  # noqa: E402
from scripts.seed_demo_research_memory import seed_demo_memory  # noqa: E402
from scripts.smoke_test_research_radar import run_smoke_tests  # noqa: E402


def test_seed_demo_memory_is_idempotent_and_smoke_tests_pass(tmp_path: Path) -> None:
    db_file = tmp_path / "test_demo_memory.db"
    db_url = f"sqlite:///{db_file}"

    # First run
    seed_demo_memory(db_url)

    # Second run to test idempotence
    seed_demo_memory(db_url)

    # Run smoke test against the seeded database
    success = run_smoke_tests(db_url)
    assert success is True

    # Verify all 5 gap types exist and coverage candidate integrity
    db = Database.create(db_url)
    repo = ResearchRepository(db)

    all_gaps = repo.list_candidates(limit=50)
    gap_types = {g.gap_type for g in all_gaps}
    expected_types = {"explicit", "coverage", "evaluation", "contradiction", "method_transfer"}
    assert expected_types.issubset(gap_types)

    # Explicit coverage gap checks
    cov_gap = repo.get_candidate("gap-coverage-mri")
    assert cov_gap is not None
    assert cov_gap.gap_type == "coverage"
    assert cov_gap.review_status == "preserved"
    assert len(cov_gap.provenance.corpus_paper_ids) >= 2
    assert len(cov_gap.provenance.supporting_evidence) >= 1
    assert cov_gap.provenance.supporting_evidence[0].claim_or_field == "evaluation_conditions"

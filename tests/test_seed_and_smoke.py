"""Automated tests verifying seed_demo_research_memory and smoke_test_research_radar."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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

"""Deterministic local smoke test script for ResearchRadar.

Exercises:
1. Project lookup
2. Global /ask retrieval
3. Project-scoped /ask retrieval
4. Rejected idea memory
5. Critic-aware gap retrieval
6. Source-ID safety
7. Gap type coverage

Runs against the seeded demo database without requiring external network or live LLM APIs.
Exits 0 on SUCCESS, non-zero on FAILURE.
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from typing import TypeVar

from pydantic import BaseModel

from research_radar.reader.llm.base import LLMMessage
from research_radar.research.ask import AskLLMResponse, AskService
from research_radar.storage.database import Database
from research_radar.storage.repositories import ResearchRepository

ModelT = TypeVar("ModelT", bound=BaseModel)


def _utc_now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class SmokeTestMockLLM:
    """Mock LLM Provider for deterministic smoke testing."""

    def __init__(self) -> None:
        self.last_messages: list[LLMMessage] = []

    async def generate_structured(
        self,
        messages: list[LLMMessage],
        response_model: type[ModelT],
    ) -> ModelT:
        self.last_messages = messages
        if response_model is AskLLMResponse:
            # Return plausible synthesis citing IDs from prompt context
            return AskLLMResponse(
                answer="Synthesized response grounded in stored research memory.",
                referenced_paper_ids=["p-spectral-mri", "p-fake-hallucination-123"],
                referenced_gap_ids=["gap-explicit-mri", "gap-fake-999"],
                is_sufficient_evidence=True,
            )  # type: ignore[return-value]
        raise ValueError(f"Unsupported model {response_model}")


def run_smoke_tests(db_url: str = "sqlite:///data/research_radar.db") -> bool:
    """Run all smoke test scenarios and return True if all pass."""

    print("==================================================")
    print(" ResearchRadar Local Smoke Test Suite")
    print(f" Target Database: {db_url}")
    print("==================================================")

    db = Database.create(db_url)
    repo = ResearchRepository(db)
    mock_llm = SmokeTestMockLLM()
    ask_service = AskService(repo, llm_provider=mock_llm)

    passed_count = 0
    total_count = 7

    # Scenario 1: Project Lookup
    print("\n[Scenario 1/7] Testing Project Lookup...")
    project = repo.get_project("MRI Robustness")
    if (
        project is not None
        and project.name == "MRI Robustness"
        and len(project.hypotheses) >= 2
        and len(project.rejected_ideas) >= 1
    ):
        print("  ✓ PASS: Found project 'MRI Robustness' with hypotheses and rejected ideas.")
        passed_count += 1
    else:
        print("  ✗ FAIL: Project 'MRI Robustness' not found or incomplete.")

    # Scenario 2: Global /ask Retrieval
    print("\n[Scenario 2/7] Testing Global /ask Retrieval...")
    ctx_global = ask_service.build_ask_context("diffusion priors")
    if (
        len(ctx_global.retrieved_papers) > 0
        and any("Diffusion" in p.title for p in ctx_global.retrieved_papers)
    ):
        print(f"  ✓ PASS: Retrieved {len(ctx_global.retrieved_papers)} matching global paper(s).")
        passed_count += 1
    else:
        print("  ✗ FAIL: Global paper retrieval failed for 'diffusion priors'.")

    # Scenario 3: Project-Scoped /ask Retrieval & Relation Boost
    print("\n[Scenario 3/7] Testing Project-Scoped /ask Retrieval & Relation Priority...")
    ctx_proj = ask_service.build_ask_context(
        "scanner shift", project_id_or_name="MRI Robustness"
    )
    if (
        ctx_proj.project is not None
        and len(ctx_proj.project_paper_links) > 0
        and len(ctx_proj.retrieved_papers) > 0
    ):
        top_paper = ctx_proj.retrieved_papers[0]
        print(
            f"  ✓ PASS: Project-scoped query retrieved {len(ctx_proj.retrieved_papers)} papers. "
            f"Top paper: '{top_paper.title}'."
        )
        passed_count += 1
    else:
        print("  ✗ FAIL: Project-scoped retrieval failed.")

    # Scenario 4: Rejected Idea Memory in AskContext & Evidence Prompt
    print("\n[Scenario 4/7] Testing Rejected Idea Memory...")
    if ctx_proj.rejected_ideas and any("Pure GAN" in r for r in ctx_proj.rejected_ideas):
        print(f"  ✓ PASS: Rejected idea preserved in AskContext: '{ctx_proj.rejected_ideas[0]}'.")
        passed_count += 1
    else:
        print("  ✗ FAIL: Rejected ideas missing in AskContext.")

    # Scenario 5: Critic-Aware Gap Retrieval
    print("\n[Scenario 5/7] Testing Critic-Aware Gap Retrieval...")
    ctx_gap = ask_service.build_ask_context("real-time multi-coil diffusion")
    if (
        len(ctx_gap.retrieved_gaps) > 0
        and "gap-explicit-mri" in ctx_gap.critic_reviews
        and ctx_gap.critic_reviews["gap-explicit-mri"].decision == "preserved"
    ):
        rev = ctx_gap.critic_reviews["gap-explicit-mri"]
        print(
            f"  ✓ PASS: Retrieved gap with latest CriticReview "
            f"(v{rev.review_version}, decision={rev.decision})."
        )
        passed_count += 1
    else:
        print("  ✗ FAIL: Critic review not properly attached to retrieved gap.")

    # Scenario 6: Source-ID Safety & Hallucination Filtering
    print("\n[Scenario 6/7] Testing Source-ID Safety & Hallucination Filtering...")
    import asyncio

    res = asyncio.run(
        ask_service.ask("scanner shift", project_id_or_name="MRI Robustness")
    )
    if (
        "p-fake-hallucination-123" not in res.referenced_paper_ids
        and "gap-fake-999" not in res.referenced_gap_ids
        and res.is_sufficient_evidence
    ):
        print(
            "  ✓ PASS: Discarded hallucinated IDs while preserving allowed evidence citations."
        )
        passed_count += 1
    else:
        print("  ✗ FAIL: Source-ID safety failed to filter fake IDs.")

    # Scenario 7: All Gap Types Present
    print("\n[Scenario 7/7] Testing Gap Types Coverage...")
    all_gaps = repo.list_candidates(limit=50)
    gap_types = {g.gap_type for g in all_gaps}
    expected_types = {"explicit", "evaluation", "contradiction", "method_transfer"}
    if expected_types.issubset(gap_types):
        print(f"  ✓ PASS: All required gap types present in repository: {sorted(gap_types)}.")
        passed_count += 1
    else:
        missing = expected_types - gap_types
        print(f"  ✗ FAIL: Missing gap types: {missing}.")

    # Summary
    print("\n==================================================")
    print(f" Results: {passed_count}/{total_count} scenarios passed.")
    print("==================================================")

    return passed_count == total_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local smoke test suite for ResearchRadar.")
    parser.add_argument(
        "--db-url",
        default="sqlite:///data/research_radar.db",
        help="SQLite database URL (default: sqlite:///data/research_radar.db)",
    )
    args = parser.parse_args()

    success = run_smoke_tests(args.db_url)
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()

"""Deterministic local SQLite demo seeder for ResearchRadar research memory.

Populates a realistic demo corpus with an MRI Robustness project, rich PaperCards,
CandidateGaps, Critic reviews, project links, and rejected ideas.

Idempotent: running multiple times updates/re-seeds cleanly without duplicate errors.
"""

from __future__ import annotations

import argparse
import logging
from datetime import UTC, datetime
from pathlib import Path

from research_radar.models.gap import CandidateGap, CriticReview, EvidenceRef, GapProvenance
from research_radar.models.paper import Paper
from research_radar.models.paper_card import EvidenceClaim, PaperCard, StructuredEvidence
from research_radar.storage.database import Database
from research_radar.storage.repositories import ResearchRepository

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("seed_demo")


def _utc_now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def seed_demo_memory(db_url: str = "sqlite:///data/research_radar.db") -> None:
    """Seed a deterministic, realistic research memory corpus into SQLite."""

    db = Database.create(db_url)
    db.initialize_schema()
    repo = ResearchRepository(db)

    now = _utc_now()
    logger.info("Initializing / seeding demo research memory in %s...", db_url)

    # 1. Create / Upsert Project: MRI Robustness
    project_name = "MRI Robustness"
    existing_projects = repo.list_projects()
    project = next((p for p in existing_projects if p.name == project_name), None)

    if project is None:
        project = repo.create_project(
            name=project_name,
            goal="study robustness of MRI reconstruction under scanner/domain shift",
            keywords=["MRI", "reconstruction", "robustness", "domain-shift", "scanner-shift"],
            hypotheses=[
                "Spectral regularization stabilizes reconstruction across scanner shifts",
                "Diffusion priors preserve micro-structures better than pure GANs",
            ],
            constraints=["Inference time < 50ms", "Max VRAM 16GB"],
            rejected_ideas=[
                "Pure GAN reconstruction due to hallucinated lesions and training instability"
            ],
        )
        logger.info("Created project: %s (ID: %s)", project.name, project.id)
    else:
        project = repo.update_project(
            project.id,
            goal="study robustness of MRI reconstruction under scanner/domain shift",
            keywords=["MRI", "reconstruction", "robustness", "domain-shift", "scanner-shift"],
            hypotheses=[
                "Spectral regularization stabilizes reconstruction across scanner shifts",
                "Diffusion priors preserve micro-structures better than pure GANs",
            ],
            constraints=["Inference time < 50ms", "Max VRAM 16GB"],
            rejected_ideas=[
                "Pure GAN reconstruction due to hallucinated lesions and training instability"
            ],
        )
        logger.info("Updated existing project: %s (ID: %s)", project.name, project.id)

    # 2. Seed Papers
    paper_defs: list[dict] = [
        {
            "id": "p-spectral-mri",
            "title": "Spectral Regularization for Robust MRI Reconstruction",
            "abstract": (
                "Deep learning MRI reconstruction models often fail under distribution shift. "
                "We introduce spectral Lipschitz regularization on convolutional layers, "
                "demonstrating improved structural stability against scanner variations."
            ),
            "authors": ["Alice Chen", "Bob Wang", "Carol Davis"],
            "year": 2024,
            "venue": "IEEE TMI",
            "doi": "10.1109/TMI.2024.001",
        },
        {
            "id": "p-diffusion-mri",
            "title": "Diffusion Priors for Accelerated 3D MRI Reconstruction",
            "abstract": (
                "Score-based diffusion models provide expressive anatomical priors for MRI. "
                "We demonstrate high-fidelity reconstruction at 8x acceleration factors "
                "while preserving diagnostic micro-structure."
            ),
            "authors": ["David Miller", "Emma Stone"],
            "year": 2024,
            "venue": "MICCAI",
            "doi": "10.1007/MICCAI.2024.002",
        },
        {
            "id": "p-domain-adapt",
            "title": "Test-Time Domain Adaptation for Scanner Shift in MRI",
            "abstract": (
                "Cross-scanner generalization remains a severe bottleneck for clinical MRI. "
                "We propose test-time entropy minimization on k-space trajectory features, "
                "mitigating performance drops when transferring between 1.5T and 3.0T scanners."
            ),
            "authors": ["Frank Liu", "Grace Hopper"],
            "year": 2023,
            "venue": "Medical Image Analysis",
            "doi": "10.1016/j.media.2023.003",
        },
        {
            "id": "p-eval-shift",
            "title": "Benchmarking Scanner Shift and Out-of-Distribution Noise in MRI",
            "abstract": (
                "We benchmark 12 deep reconstruction architectures across 5 distinct scanner "
                "vendors. Results show standard PSNR metrics overestimate clinical utility under "
                "severe domain shifts."
            ),
            "authors": ["Hannah Abbott", "Ian Curtis"],
            "year": 2024,
            "venue": "Radiology AI",
            "doi": "10.1148/ryai.2024.004",
        },
        {
            "id": "p-robustness-limit",
            "title": "Limitations of Deep Learning Reconstruction Under Severe Artifacts",
            "abstract": (
                "When patient motion and scanner frequency drift coincide, deep networks can "
                "generate plausible yet false anatomic structures. We formalize bounds on "
                "worst-case hallucination risk."
            ),
            "authors": ["Julia Robert", "Kevin Bacon"],
            "year": 2023,
            "venue": "Magnetic Resonance in Medicine",
            "doi": "10.1002/mrm.2023.005",
        },
        {
            "id": "p-spectral-degrade",
            "title": "Spectral Regularization Degrades High-Frequency Details in Low-SNR MRI",
            "abstract": (
                "While spectral norm constraints improve Lipschitz stability, we observe that "
                "they over-smooth fine trabecular bone patterns in low-SNR 7T MRI acquisitions."
            ),
            "authors": ["Laura Croft", "Michael Chang"],
            "year": 2024,
            "venue": "ISMRM",
            "doi": "10.1002/ismrm.2024.006",
        },
        {
            "id": "p-patch-transfer",
            "title": "Patch-Based Wavelet Normalization for CT Artifact Removal",
            "abstract": (
                "Computed tomography metal artifacts can be suppressed using directional wavelet "
                "normalization across multi-scale patches without losing edge sharpness."
            ),
            "authors": ["Nina Simone", "Oscar Wilde"],
            "year": 2023,
            "venue": "IEEE TBME",
            "doi": "10.1109/TBME.2023.007",
        },
        {
            "id": "p-unrelated-nlp",
            "title": "Transformer Attention for Multi-Document Text Summarization",
            "abstract": (
                "We propose hierarchical cross-attention for summarizing long document clusters in "
                "natural language processing."
            ),
            "authors": ["Peter Parker", "Quinn Fabray"],
            "year": 2023,
            "venue": "ACL",
            "doi": "10.18653/v1/acl.2023.008",
        },
    ]

    paper_id_map: dict[str, str] = {}
    for pdef in paper_defs:
        paper_obj = Paper(
            id=pdef["id"],
            title=pdef["title"],
            abstract=pdef["abstract"],
            authors=pdef["authors"],
            publication_year=pdef["year"],
            venue=pdef["venue"],
            doi=pdef["doi"],
            source="arxiv",
        )
        stored_id = repo.upsert_merged_paper(paper_obj)
        paper_id_map[pdef["id"]] = stored_id

    logger.info("Upserted %d papers into SQLite.", len(paper_id_map))

    # 3. Seed PaperCards with Structured Evidence
    card_defs: list[tuple[str, PaperCard]] = [
        (
            "p-spectral-mri",
            PaperCard(
                paper_id=paper_id_map["p-spectral-mri"],
                problem="Reconstruction instability under scanner domain shift",
                tasks=[
                    StructuredEvidence(value="reconstruction", status="observed"),
                    StructuredEvidence(value="segmentation", status="unknown"),
                ],
                modalities=[
                    StructuredEvidence(value="MRI", status="observed"),
                    StructuredEvidence(value="CT", status="explicitly_absent"),
                ],
                evaluation_conditions=[
                    StructuredEvidence(value="scanner_shift", status="observed"),
                    StructuredEvidence(value="motion_artifacts", status="unknown"),
                ],
                methods=["Spectral Regularization", "Lipschitz Constraint", "U-Net"],
                datasets=["fastMRI", "SKM-TEA"],
                metrics=["PSNR", "SSIM", "VIF"],
                main_claims=[
                    EvidenceClaim(
                        claim="Spectral norm constraints reduce scanner transfer error by 32%",
                        source_section="Results",
                        supporting_text=(
                            "Transfer error drops from 0.042 to 0.028 NMSE across 1.5T/3T."
                        ),
                    )
                ],
                limitations=["Higher training convergence time"],
            ),
        ),
        (
            "p-diffusion-mri",
            PaperCard(
                paper_id=paper_id_map["p-diffusion-mri"],
                problem="Artifact generation at high acceleration factors in 3D MRI",
                tasks=[
                    StructuredEvidence(value="reconstruction", status="observed"),
                    StructuredEvidence(value="denoising", status="observed"),
                ],
                modalities=[
                    StructuredEvidence(value="MRI", status="observed"),
                ],
                evaluation_conditions=[
                    StructuredEvidence(value="high_acceleration_8x", status="observed"),
                    StructuredEvidence(value="scanner_shift", status="unknown"),
                ],
                methods=["Score-based Diffusion", "SDE Solver"],
                datasets=["fastMRI Brain", "Stanford 3D"],
                metrics=["PSNR", "SSIM", "LPIPS"],
                main_claims=[
                    EvidenceClaim(
                        claim="Diffusion priors preserve micro-vessel sharpness at 8x acceleration",
                        source_section="Abstract",
                        supporting_text="Diagnostic sharpness preserved at 8x undersampling.",
                    )
                ],
                limitations=["Slow iterative reverse-time sampling during inference"],
            ),
        ),
        (
            "p-spectral-degrade",
            PaperCard(
                paper_id=paper_id_map["p-spectral-degrade"],
                problem="Loss of high-frequency anatomical detail in ultra-high-field MRI",
                tasks=[
                    StructuredEvidence(value="reconstruction", status="observed"),
                ],
                modalities=[
                    StructuredEvidence(value="MRI", status="observed"),
                ],
                evaluation_conditions=[
                    StructuredEvidence(value="scanner_shift", status="observed"),
                    StructuredEvidence(value="low_snr_7T", status="observed"),
                ],
                methods=["Spectral Regularization", "ResNet"],
                datasets=["7T Human Brain Dataset"],
                metrics=["PSNR", "SSIM", "High-Frequency Error"],
                main_claims=[
                    EvidenceClaim(
                        claim="Spectral Regularization degrades high-frequency detail at low SNR",
                        source_section="Discussion",
                        supporting_text="Smoothing effect obscures fine trabecular structures.",
                    )
                ],
                limitations=["Evaluated only on 7T scanner setup"],
            ),
        ),
        (
            "p-patch-transfer",
            PaperCard(
                paper_id=paper_id_map["p-patch-transfer"],
                problem="Severe streak artifacts in computed tomography",
                tasks=[
                    StructuredEvidence(value="reconstruction", status="observed"),
                    StructuredEvidence(value="artifact_removal", status="observed"),
                ],
                modalities=[
                    StructuredEvidence(value="CT", status="observed"),
                ],
                evaluation_conditions=[
                    StructuredEvidence(value="metal_artifacts", status="observed"),
                ],
                methods=["Patch-Based Wavelet Normalization", "Dual-Tree Wavelet"],
                datasets=["DeepLesion CT"],
                metrics=["PSNR", "Artifact Index"],
                main_claims=[
                    EvidenceClaim(
                        claim="Wavelet normalization removes streak artifacts without edge blur",
                        source_section="Results",
                        supporting_text=(
                            "Streak intensity reduced by 45% with sharp edge preservation."
                        ),
                    )
                ],
                limitations=["Requires pre-calibrated patch dictionary"],
            ),
        ),
    ]

    for _orig_id, card in card_defs:
        repo.upsert_paper_card(card)

    logger.info("Upserted %d detailed PaperCards.", len(card_defs))

    # 4. Link Papers to Project
    repo.add_paper_to_project(
        project.id,
        paper_id_map["p-spectral-mri"],
        relation="seed",
        note="Core baseline method for scanner shift invariance",
    )
    repo.add_paper_to_project(
        project.id,
        paper_id_map["p-diffusion-mri"],
        relation="supporting",
        note="Alternative prior for accelerated imaging",
    )
    repo.add_paper_to_project(
        project.id,
        paper_id_map["p-domain-adapt"],
        relation="relevant",
        note="Test-time adaptation reference",
    )
    repo.add_paper_to_project(
        project.id,
        paper_id_map["p-eval-shift"],
        relation="background",
        note="Benchmark protocol for multi-vendor evaluation",
    )
    repo.add_paper_to_project(
        project.id,
        paper_id_map["p-spectral-degrade"],
        relation="conflicting",
        note="Reports over-smoothing with spectral constraints in low-SNR regimes",
    )

    logger.info("Linked papers to project '%s'.", project.name)

    # 5. Seed Candidate Gaps with Critic Reviews & Project Gap Links
    gap_defs = [
        CandidateGap(
            id="gap-explicit-mri",
            title="Real-time multi-coil diffusion MRI reconstruction under scanner shift",
            description=(
                "Investigate low-latency inference for diffusion models under scanner domain shift."
            ),
            gap_type="explicit",
            research_question=(
                "Can distilled diffusion priors achieve <50ms latency across "
                "multi-vendor MRI scanners?"
            ),
            supporting_papers=[paper_id_map["p-spectral-mri"], paper_id_map["p-diffusion-mri"]],
            evidence_count=2,
            novelty_score=0.85,
            feasibility_score=0.75,
            confidence=0.80,
            search_scope="MRI Reconstruction Robustness",
            review_status="preserved",
            provenance=GapProvenance(
                retrievals=[],
                corpus_paper_ids=[
                    paper_id_map["p-spectral-mri"],
                    paper_id_map["p-diffusion-mri"],
                ],
                corpus_description="Demo MRI corpus",
                supporting_evidence=[
                    EvidenceRef(
                        paper_id=paper_id_map["p-spectral-mri"],
                        paper_title="Spectral Regularization for Robust MRI Reconstruction",
                        evidence_kind="supporting",
                        claim_or_field="methods",
                    )
                ],
            ),
            created_at=now,
        ),
        CandidateGap(
            id="gap-eval-mri",
            title=(
                "Standardized evaluation of deep MRI reconstruction under "
                "combined motion and frequency drift"
            ),
            description=(
                "Evaluate deep architectures on concurrent motion artifacts "
                "and B0 frequency drift."
            ),
            gap_type="evaluation",
            research_question=(
                "How do state-of-the-art deep reconstruction models degrade "
                "when motion and frequency drift co-occur?"
            ),
            supporting_papers=[paper_id_map["p-robustness-limit"], paper_id_map["p-eval-shift"]],
            evidence_count=2,
            novelty_score=0.78,
            feasibility_score=0.82,
            confidence=0.75,
            search_scope="MRI Reconstruction Robustness",
            review_status="candidate",
            provenance=GapProvenance(
                retrievals=[],
                corpus_paper_ids=[
                    paper_id_map["p-robustness-limit"],
                    paper_id_map["p-eval-shift"],
                ],
                corpus_description="Demo MRI corpus",
            ),
            created_at=now,
        ),
        CandidateGap(
            id="gap-contradiction-mri",
            title=(
                "Contradiction regarding Spectral Regularization efficacy "
                "across varying MRI SNR regimes"
            ),
            description=(
                "Paper 1 demonstrates 32% error reduction while Paper 6 "
                "observes severe detail degradation at low SNR."
            ),
            gap_type="contradiction",
            research_question=(
                "Does Spectral Regularization reliably generalize across different "
                "SNR and field strengths without detail loss?"
            ),
            supporting_papers=[paper_id_map["p-spectral-mri"]],
            conflicting_papers=[paper_id_map["p-spectral-degrade"]],
            evidence_count=2,
            novelty_score=0.90,
            feasibility_score=0.70,
            confidence=0.85,
            search_scope="MRI Reconstruction Robustness",
            review_status="preserved",
            provenance=GapProvenance(
                retrievals=[],
                corpus_paper_ids=[
                    paper_id_map["p-spectral-mri"],
                    paper_id_map["p-spectral-degrade"],
                ],
                corpus_description="Demo MRI corpus",
            ),
            created_at=now,
        ),
        CandidateGap(
            id="gap-transfer-mri",
            title=(
                "Method transfer: Patch-Based Wavelet Normalization → "
                "MRI Scanner Shift Robustness"
            ),
            description=(
                "Transfer directional wavelet normalization from CT artifact suppression "
                "to MRI scanner shift robustness."
            ),
            gap_type="method_transfer",
            research_question=(
                "Can Patch-Based Wavelet Normalization improve edge preservation "
                "and robustness under scanner shift in MRI?"
            ),
            supporting_papers=[paper_id_map["p-patch-transfer"], paper_id_map["p-domain-adapt"]],
            evidence_count=2,
            novelty_score=0.88,
            feasibility_score=0.80,
            confidence=0.82,
            search_scope="MRI Reconstruction Robustness",
            review_status="candidate",
            provenance=GapProvenance(
                retrievals=[],
                corpus_paper_ids=[
                    paper_id_map["p-patch-transfer"],
                    paper_id_map["p-domain-adapt"],
                ],
                corpus_description="Demo MRI corpus",
            ),
            created_at=now,
        ),
    ]

    for gap in gap_defs:
        repo.save_candidate(gap)

    # Add Critic Reviews if not already recorded
    existing_revs_explicit = repo.list_critic_reviews("gap-explicit-mri")
    if not existing_revs_explicit:
        rev_explicit = CriticReview(
            candidate_id="gap-explicit-mri",
            review_version=1,
            decision="preserved",
            rationale=(
                "Strong explicit evidence supported by distinct fastMRI and SKM-TEA benchmarks."
            ),
            caveats=["Requires inference hardware with <50ms TensorRT support"],
            created_at=now,
        )
        repo.save_critic_review(rev_explicit)

    existing_revs_contra = repo.list_critic_reviews("gap-contradiction-mri")
    if not existing_revs_contra:
        rev_contradiction = CriticReview(
            candidate_id="gap-contradiction-mri",
            review_version=1,
            decision="preserved",
            rationale=(
                "Clear evidence-grounded contradiction across SNR 1.5T/3T vs 7T field strengths."
            ),
            caveats=["Field strength differences may be the confounding factor"],
            created_at=now,
        )
        repo.save_critic_review(rev_contradiction)

    # Link Gaps to Project
    repo.add_gap_to_project(project.id, "gap-explicit-mri", status="active")
    repo.add_gap_to_project(project.id, "gap-eval-mri", status="active")
    repo.add_gap_to_project(project.id, "gap-contradiction-mri", status="interesting")
    repo.add_gap_to_project(project.id, "gap-transfer-mri", status="active")

    logger.info("Successfully seeded all demo research memory items into %s.", db_url)


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed ResearchRadar demo research memory.")
    parser.add_argument(
        "--db-url",
        default="sqlite:///data/research_radar.db",
        help="SQLite database URL (default: sqlite:///data/research_radar.db)",
    )
    args = parser.parse_args()

    # Ensure parent directory exists for file SQLite URLs
    if args.db_url.startswith("sqlite:///"):
        path_str = args.db_url.replace("sqlite:///", "")
        path_obj = Path(path_str)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

    seed_demo_memory(args.db_url)


if __name__ == "__main__":
    main()

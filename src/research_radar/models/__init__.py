"""Provider-neutral research domain models."""

from research_radar.models.document import PaperDocument
from research_radar.models.gap import (
    CandidateGap,
    CriticReview,
    EvidenceRef,
    GapProvenance,
    RetrievalRecord,
)
from research_radar.models.paper import Paper
from research_radar.models.paper_card import EvidenceClaim, PaperCard, StructuredEvidence
from research_radar.models.project import Project, ProjectGapLink, ProjectPaperLink

__all__ = [
    "CandidateGap",
    "CriticReview",
    "EvidenceClaim",
    "EvidenceRef",
    "GapProvenance",
    "Paper",
    "PaperCard",
    "PaperDocument",
    "Project",
    "ProjectGapLink",
    "ProjectPaperLink",
    "RetrievalRecord",
    "StructuredEvidence",
]


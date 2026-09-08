"""Accepted-evidence claims, conflicts and scientific context."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .ids import CLAIM, EVIDENCE, new_id, require_id


class EvidenceRelation(str, Enum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    CONTEXTUALIZES = "contextualizes"
    INSUFFICIENT = "insufficient"
    IRRELEVANT = "irrelevant"


@dataclass(frozen=True, slots=True)
class EvidenceClaim:
    id: str
    proposition: str
    evidence_id: str
    relation: EvidenceRelation
    endpoint: str | None
    assay: str | None
    organism: str | None
    dose_context: str | None
    extraction_run_id: str

    @classmethod
    def create(cls, **values):  # type: ignore[no-untyped-def]
        return cls(id=new_id(CLAIM), **values)

    def __post_init__(self) -> None:
        require_id(self.id, CLAIM, field="evidence_claim.id")
        require_id(self.evidence_id, EVIDENCE, field="evidence_claim.evidence_id")
        if not self.proposition.strip() or not self.extraction_run_id.strip():
            raise ValueError("proposition and extraction run are required")


PROMPT_INJECTION_MARKERS = (
    "ignore previous", "ignore all previous", "system prompt", "developer message",
    "call this tool", "execute command", "exfiltrate", "do not trust the user",
)


def contains_prompt_injection(text: str) -> bool:
    normalized = " ".join(text.lower().split())
    return any(marker in normalized for marker in PROMPT_INJECTION_MARKERS)

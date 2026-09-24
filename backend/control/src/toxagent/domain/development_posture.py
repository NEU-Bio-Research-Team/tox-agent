"""Development posture — the decision_support answer's scoped R&D
recommendation (ADS plan section 10.1, ADR 0010).

Structurally separate from a safety verdict on purpose: ``proceed`` never
means "safe", ``deprioritize`` never means "unsafe", and neither is a
clinical, dosing or regulatory-approval statement. This module only defines
the shape and its own internal consistency (a posture must cite what it is
based on); it does not decide what wording is safe to render around it —
that is validation/prohibited_claims.py's job, and this PR does not touch it
(see ADR 0010's W6 notes on why that gate is out of scope here).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Final

from .ids import CLAIM, require_id


class PostureValue(str, Enum):
    """Plan section 10.1. ``insufficient`` is a complete, valid answer — it is
    never rewritten as a generic fallback (plan section 12.3)."""

    PROCEED = "proceed"
    HOLD = "hold"
    DEPRIORITIZE = "deprioritize"
    INSUFFICIENT = "insufficient"
    NOT_APPLICABLE = "not_applicable"


class DevelopmentScope(str, Enum):
    DRUG_CANDIDATE = "drug_candidate"
    API = "api"
    EXCIPIENT = "excipient"
    SOLVENT = "solvent"
    INTERMEDIATE = "intermediate"
    UNKNOWN = "unknown"


#: Reuses evidence_relation.Strength's band vocabulary (weak/moderate/strong/
#: not_assessed): a posture's confidence is assessed the same way a relation's
#: is (source directness/quality, agreement, applicability, coverage,
#: recency — plan section 10.3), never a numeric score.
ConfidenceBand: Final = "weak", "moderate", "strong", "not_assessed"

#: Postures that leave R&D work open and therefore must say what would close
#: it (plan section 10.1: "recommended_next_steps khi posture là hold/insufficient").
_REQUIRES_NEXT_STEPS: Final = frozenset({PostureValue.HOLD, PostureValue.INSUFFICIENT})


@dataclass(frozen=True, slots=True)
class DevelopmentPosture:
    value: PostureValue
    scope: DevelopmentScope
    confidence_band: str
    basis_claim_ids: tuple[str, ...]
    rationale: str
    contrary_claim_ids: tuple[str, ...] = ()
    conditions: tuple[str, ...] = ()
    recommended_next_steps: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.confidence_band not in ConfidenceBand:
            raise ValueError(
                f"confidence_band must be one of {ConfidenceBand}, got {self.confidence_band!r}"
            )
        for claim_id in (*self.basis_claim_ids, *self.contrary_claim_ids):
            require_id(claim_id, CLAIM, field="development_posture.claim_id")
        if self.value in (PostureValue.PROCEED, PostureValue.HOLD, PostureValue.DEPRIORITIZE):
            if not self.basis_claim_ids:
                raise ValueError(
                    f"a {self.value.value} posture must cite at least one basis claim — "
                    "a recommendation with no basis is not an accepted state (plan section 10.1)"
                )
        if not self.rationale.strip():
            raise ValueError("a development posture must state its rationale")
        if self.value in _REQUIRES_NEXT_STEPS and not self.recommended_next_steps:
            raise ValueError(
                f"a {self.value.value} posture must name recommended_next_steps — "
                "what would change the answer, not just that something is missing"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "value": self.value.value,
            "scope": self.scope.value,
            "confidence_band": self.confidence_band,
            "basis_claim_ids": list(self.basis_claim_ids),
            "contrary_claim_ids": list(self.contrary_claim_ids),
            "conditions": list(self.conditions),
            "rationale": self.rationale,
            "recommended_next_steps": list(self.recommended_next_steps),
        }

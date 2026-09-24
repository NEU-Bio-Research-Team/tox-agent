"""GroundedAnswer — the only shape an answer may take.

Plan section 5.7. Note what is absent: there is no field for an overall
toxicity, safety or severity score, so a model cannot emit one and a renderer
cannot display one (ADR 0002). Every claim names its own basis, and the
recommendation kind is structurally separate from the fact kinds so that
"consider running an hERG assay" can never be stored as a measurement.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Final

from .ids import ANSWER, CLAIM, EVIDENCE, OBSERVATION, RUN, SESSION, new_id, require_id
from .provenance import content_sha256

SCHEMA_VERSION: Final = "grounded-answer-v1"


class ClaimKind(str, Enum):
    NUMERIC = "numeric"
    CLASSIFICATION = "classification"
    SCIENTIFIC = "scientific"
    COMPARISON = "comparison"
    LIMITATION = "limitation"
    RECOMMENDATION = "recommendation"


#: Kinds whose value must equal a canonical field, exactly (plan section 9.1-9.2).
FIELD_BACKED_KINDS: Final[frozenset[ClaimKind]] = frozenset(
    {ClaimKind.NUMERIC, ClaimKind.CLASSIFICATION}
)


class LimitationCode(str, Enum):
    """Plan section 9.4. The renderer may merge the wording of two of these into
    one sentence; it may not drop the code."""

    UNCALIBRATED_PROBABILITY = "uncalibrated_probability"
    APPLICABILITY_IS_RULE_BASED = "applicability_is_rule_based"
    ATTRIBUTION_NOT_CAUSALITY = "attribution_not_causality"
    ENDPOINT_UNAVAILABLE = "endpoint_unavailable"
    EVIDENCE_SCOPE_LIMITED = "evidence_scope_limited"
    SCREENING_NOT_SAFETY_ASSESSMENT = "screening_not_safety_assessment"


# identity | round:0..6 | percent:0..6 | difference | ratio
TRANSFORM_GRAMMAR: Final = re.compile(r"^(identity|round:[0-6]|percent:[0-6]|difference|ratio)$")
DERIVED_TRANSFORMS: Final[frozenset[str]] = frozenset({"difference", "ratio"})


@dataclass(frozen=True, slots=True)
class Claim:
    claim_id: str
    kind: ClaimKind
    text: str
    observation_id: str | None = None
    field_path: str | None = None
    source_value: Any = None
    rendered_value: str | None = None
    transform: str = "identity"
    citation_ids: tuple[str, ...] = ()
    input_claim_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        require_id(self.claim_id, CLAIM, field="claim.claim_id")
        if self.observation_id is not None:
            require_id(self.observation_id, OBSERVATION, field="claim.observation_id")
        for citation in self.citation_ids:
            require_id(citation, EVIDENCE, field="claim.citation_ids[]")
        for source in self.input_claim_ids:
            require_id(source, CLAIM, field="claim.input_claim_ids[]")
        if not TRANSFORM_GRAMMAR.match(self.transform):
            raise ValueError(
                f"claim.transform {self.transform!r} is not in the allowlist "
                "(identity, round:0-6, percent:0-6, difference, ratio)"
            )
        if self.kind in FIELD_BACKED_KINDS and not (self.observation_id and self.field_path):
            raise ValueError(
                f"a {self.kind.value} claim must name an observation_id and a field_path"
            )
        if not self.text.strip():
            raise ValueError("claim.text must not be empty")

    @property
    def is_derived(self) -> bool:
        return self.transform in DERIVED_TRANSFORMS

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "claim_id": self.claim_id,
            "kind": self.kind.value,
            "text": self.text,
            "transform": self.transform,
            "citation_ids": list(self.citation_ids),
        }
        if self.observation_id:
            out["observation_id"] = self.observation_id
        if self.field_path:
            out["field_path"] = self.field_path
        if self.source_value is not None:
            out["source_value"] = self.source_value
        if self.rendered_value is not None:
            out["rendered_value"] = self.rendered_value
        if self.input_claim_ids:
            out["input_claim_ids"] = list(self.input_claim_ids)
        return out


@dataclass(frozen=True, slots=True)
class Limitation:
    code: LimitationCode
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code.value, "text": self.text}


@dataclass(frozen=True, slots=True)
class RecommendedNextStep:
    text: str
    basis_claim_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "basis_claim_ids": list(self.basis_claim_ids)}


@dataclass(frozen=True, slots=True)
class GroundedAnswer:
    """A committed answer. Reaching this type means validation already passed;
    the candidate a model submits is a separate wire model in ``tools``."""

    id: str
    session_id: str
    run_id: str
    answer_markdown: str
    claims: tuple[Claim, ...]
    limitations: tuple[Limitation, ...]
    recommended_next_steps: tuple[RecommendedNextStep, ...]
    candidate_generation: int
    content_sha256: str
    created_at: datetime
    schema_version: str = SCHEMA_VERSION
    is_fallback: bool = False

    def __post_init__(self) -> None:
        require_id(self.id, ANSWER, field="answer.id")
        require_id(self.session_id, SESSION, field="answer.session_id")
        require_id(self.run_id, RUN, field="answer.run_id")
        ids = [c.claim_id for c in self.claims]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate claim_id within one answer")

    @classmethod
    def create(
        cls,
        *,
        session_id: str,
        run_id: str,
        answer_markdown: str,
        claims: tuple[Claim, ...],
        limitations: tuple[Limitation, ...] = (),
        recommended_next_steps: tuple[RecommendedNextStep, ...] = (),
        candidate_generation: int = 1,
        now: datetime,
        is_fallback: bool = False,
    ) -> "GroundedAnswer":
        body = {
            "answer_markdown": answer_markdown,
            "claims": [c.to_dict() for c in claims],
            "limitations": [l.to_dict() for l in limitations],
            "recommended_next_steps": [s.to_dict() for s in recommended_next_steps],
        }
        return cls(
            id=new_id(ANSWER),
            session_id=session_id,
            run_id=run_id,
            answer_markdown=answer_markdown,
            claims=claims,
            limitations=limitations,
            recommended_next_steps=recommended_next_steps,
            candidate_generation=candidate_generation,
            content_sha256=content_sha256(body),
            created_at=now,
            is_fallback=is_fallback,
        )

    @property
    def limitation_codes(self) -> frozenset[str]:
        return frozenset(l.code.value for l in self.limitations)

    @property
    def cited_observation_ids(self) -> frozenset[str]:
        return frozenset(c.observation_id for c in self.claims if c.observation_id)

    @property
    def cited_evidence_ids(self) -> frozenset[str]:
        return frozenset(e for c in self.claims for e in c.citation_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "answer_id": self.id,
            "run_id": self.run_id,
            "answer_markdown": self.answer_markdown,
            "claims": [c.to_dict() for c in self.claims],
            "limitations": [l.to_dict() for l in self.limitations],
            "recommended_next_steps": [s.to_dict() for s in self.recommended_next_steps],
            "candidate_generation": self.candidate_generation,
            "is_fallback": self.is_fallback,
            "content_sha256": self.content_sha256,
            "created_at": self.created_at.isoformat(),
        }

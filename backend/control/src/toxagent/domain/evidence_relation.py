"""Evidence relations for decision support (ADS plan section 9.3, ADR 0010).

A relation is the assessed bearing of one source on one proposition — never a
single pseudo-precise score, and never a source-class hierarchy applied
regardless of what is being asked. Distinct from the report-build capability's
own, narrower ``EvidenceRelation``/``EvidenceSynthesis`` (``domain/report.py``)
and from ``research/relevance.py``'s ``Relevance`` (which answers "is this hit
citable at all", not "what does this source say about this proposition") —
reconciling those three into one type is future work (see ADR 0010 and
``docs/glossary/ads-glossary.md``); this module is decision_support's own,
introduced without touching either existing type's live wire contract.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from .ids import EVIDENCE_RELATION, PROPOSITION, RUN, SESSION, new_id, require_id


class SourceClass(str, Enum):
    """What a source is canonical for (plan section 5.2). Independent of
    whether it currently agrees with anything."""

    PREDICTOR_FACT = "predictor_fact"
    EXPLANATION_FACT = "explanation_fact"
    EXTERNAL_EXPERIMENTAL = "external_experimental"
    EXTERNAL_REGULATORY = "external_regulatory"
    REPORT_FACT = "report_fact"
    AGENT_SYNTHESIS = "agent_synthesis"


class RelationLabel(str, Enum):
    """Plan section 9.3. ``INSUFFICIENT`` and ``NOT_APPLICABLE`` are both real,
    reportable outcomes, not the absence of one: "this source doesn't bear on
    the question" is a different fact from "we looked and found nothing"."""

    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    CONTEXTUAL = "contextual"
    INSUFFICIENT = "insufficient"
    NOT_APPLICABLE = "not_applicable"


class Directness(str, Enum):
    DIRECT = "direct"
    INDIRECT = "indirect"


class Applicability(str, Enum):
    """Reuses the predictor's own applicability vocabulary (ADR: this must not
    become a second, disagreeing scale) plus the two states a non-predictor
    source can be in."""

    OK = "ok"
    LIMITED = "limited"
    OUT_OF_DOMAIN = "out_of_domain"
    NOT_APPLICABLE = "not_applicable"


class Strength(str, Enum):
    """A band with a reason, not a global score (plan section 9.3)."""

    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    NOT_ASSESSED = "not_assessed"


@dataclass(frozen=True, slots=True)
class EvidenceScope:
    """What the assessed relation is actually scoped to. Two sources that
    look contradictory often disagree because one of these differs, not
    because either is wrong (plan section 9.4)."""

    endpoint: str | None = None
    species: str | None = None
    dose: str | None = None
    use_context: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "endpoint": self.endpoint,
            "species": self.species,
            "dose": self.dose,
            "use_context": self.use_context,
        }


@dataclass(frozen=True, slots=True)
class SourceRef:
    """What was assessed. ``source_id`` is a local id of the referenced kind
    (an observation id for predictor_fact/explanation_fact, an evidence id for
    external_experimental/external_regulatory, a report id for report_fact) —
    never a value copied out of it."""

    source_class: SourceClass
    source_id: str

    def to_dict(self) -> dict[str, Any]:
        return {"source_class": self.source_class.value, "source_id": self.source_id}


@dataclass(frozen=True, slots=True)
class EvidenceRelationAssessment:
    """One source's assessed relation to one proposition, for one run.

    Server-validated (W5-06): the model proposes ``relation``/``directness``/
    ``applicability``/``strength``/``reason_codes``/``scope``, but the server
    is what mints ``id`` and must independently confirm ``source_ref`` resolves
    to a real, session-owned observation/evidence/report before this is
    persisted or shown as accepted — a proposed relation whose source cannot
    be resolved is not a partially-trusted assessment, it is rejected.
    """

    id: str
    session_id: str
    run_id: str
    proposition_id: str
    source_ref: SourceRef
    relation: RelationLabel
    directness: Directness
    applicability: Applicability
    strength: Strength
    reason_codes: tuple[str, ...]
    scope: EvidenceScope
    created_at: datetime
    #: P1-03: for an ``agent_synthesis`` source, the typed refs of the
    #: artifacts the synthesis was derived from (``observation:…``,
    #: ``evidence:…``, ``report:…``). A synthesis is a transformation, never a
    #: source in its own right; the server resolves every one of these.
    input_refs: tuple[str, ...] = ()
    #: evidence_ontology.Assessor: who judged this relation, and by what.
    assessor: str = "model"
    method_version: str = "grounded-answer-v2"

    def __post_init__(self) -> None:
        require_id(self.id, EVIDENCE_RELATION, field="evidence_relation.id")
        if self.source_ref.source_class is SourceClass.AGENT_SYNTHESIS and not self.input_refs:
            raise ValueError(
                "an agent_synthesis relation must name the artifacts it was derived from "
                "(input_refs) — a synthesis cannot be its own provenance"
            )
        require_id(self.session_id, SESSION, field="evidence_relation.session_id")
        require_id(self.run_id, RUN, field="evidence_relation.run_id")
        require_id(self.proposition_id, PROPOSITION, field="evidence_relation.proposition_id")
        if self.relation in (RelationLabel.INSUFFICIENT, RelationLabel.NOT_APPLICABLE):
            return
        if not self.reason_codes:
            raise ValueError(
                f"a {self.relation.value} relation must carry at least one reason code — "
                "'we assessed this but decline to say why' is not an accepted state"
            )

    @classmethod
    def create(
        cls,
        *,
        session_id: str,
        run_id: str,
        proposition_id: str,
        source_ref: SourceRef,
        relation: RelationLabel,
        directness: Directness,
        applicability: Applicability,
        strength: Strength,
        reason_codes: tuple[str, ...] = (),
        scope: EvidenceScope = EvidenceScope(),
        input_refs: tuple[str, ...] = (),
        now: datetime,
    ) -> "EvidenceRelationAssessment":
        return cls(
            id=new_id(EVIDENCE_RELATION),
            session_id=session_id,
            run_id=run_id,
            proposition_id=proposition_id,
            source_ref=source_ref,
            relation=relation,
            directness=directness,
            applicability=applicability,
            strength=strength,
            reason_codes=reason_codes,
            scope=scope,
            created_at=now,
            input_refs=input_refs,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "proposition_id": self.proposition_id,
            "source_ref": self.source_ref.to_dict(),
            "relation": self.relation.value,
            "directness": self.directness.value,
            "applicability": self.applicability.value,
            "strength": self.strength.value,
            "reason_codes": list(self.reason_codes),
            "scope": self.scope.to_dict(),
            "input_refs": list(self.input_refs),
            "assessor": self.assessor,
            "method_version": self.method_version,
            "created_at": self.created_at.isoformat(),
        }


def new_proposition_id() -> str:
    return new_id(PROPOSITION)

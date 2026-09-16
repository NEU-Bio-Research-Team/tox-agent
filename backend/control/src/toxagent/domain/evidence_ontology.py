"""The canonical evidence ontology, v1 (P1-02).

Three vocabularies describe "what a source has to do with a claim", each
introduced by a different workstream:

* ``research.relevance.Relevance`` — direct | contextual | uncertain |
  irrelevant: is a search hit about the compound and endpoint at all;
* ``domain.report.EvidenceRelation`` — supports | contradicts | contextualizes
  | insufficient: a report proposition's bearing;
* ``domain.evidence_relation.RelationLabel`` — supports | contradicts |
  contextual | insufficient | not_applicable: decision support's bearing.

They overlap without agreeing (``contextual`` vs ``contextualizes``;
``insufficient`` meaning "nothing bears on it" in one and "looked, could not
tell" in another), and relevance was sometimes read as a relation. This module
is the one ontology they map into. It does not replace their wire contracts —
each keeps its enum for existing rows and payloads (dual-read) — but every new
consumer reads through ``canonical_relation``/``canonical_relevance``, and the
eval graders compare paths on the canonical values.

Two axes, never merged:

* **relation** — ``supports | contradicts | contextual | unrelated |
  unresolved``: what the source says about *this claim*;
* **relevance** — ``direct | contextual | uncertain | irrelevant``: whether the
  source is about the subject at all. Confidence is a third, separate field.

An assessment records its **assessor** (``deterministic | model | sme``), the
method and version, and the evidence span it rests on.

A source may be cited for a claim only when the artifact exists, has been
read, and its canonical relation to that claim is neither ``unrelated`` nor
``unresolved`` (``citable_for_claim``).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

ONTOLOGY_VERSION = "evidence-ontology-v1"


class CanonicalRelation(str, Enum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    CONTEXTUAL = "contextual"
    #: The source does not bear on this claim.
    UNRELATED = "unrelated"
    #: The source was assessed and its bearing could not be determined.
    UNRESOLVED = "unresolved"


class CanonicalRelevance(str, Enum):
    DIRECT = "direct"
    CONTEXTUAL = "contextual"
    UNCERTAIN = "uncertain"
    IRRELEVANT = "irrelevant"


class Assessor(str, Enum):
    DETERMINISTIC = "deterministic"
    MODEL = "model"
    SME = "sme"


#: vocabulary name -> legacy value -> canonical relation.
_RELATION_MAPS: dict[str, dict[str, CanonicalRelation]] = {
    # domain.evidence_relation.RelationLabel (decision support)
    "decision_support": {
        "supports": CanonicalRelation.SUPPORTS,
        "contradicts": CanonicalRelation.CONTRADICTS,
        "contextual": CanonicalRelation.CONTEXTUAL,
        "insufficient": CanonicalRelation.UNRESOLVED,
        "not_applicable": CanonicalRelation.UNRELATED,
    },
    # domain.report.EvidenceRelation (report build)
    "report": {
        "supports": CanonicalRelation.SUPPORTS,
        "contradicts": CanonicalRelation.CONTRADICTS,
        "contextualizes": CanonicalRelation.CONTEXTUAL,
        "insufficient": CanonicalRelation.UNRESOLVED,
    },
}

#: research.relevance.Relevance is already the canonical relevance vocabulary;
#: the legacy "rule" assessor name maps to ``deterministic``.
_ASSESSOR_ALIASES = {"rule": Assessor.DETERMINISTIC, "deterministic": Assessor.DETERMINISTIC,
                     "model": Assessor.MODEL, "sme": Assessor.SME}


def canonical_relation(vocabulary: str, value: str) -> CanonicalRelation:
    try:
        return _RELATION_MAPS[vocabulary][value]
    except KeyError:
        raise ValueError(f"no canonical relation for {vocabulary}:{value!r}") from None


def canonical_relevance(value: str) -> CanonicalRelevance:
    return CanonicalRelevance(value)


def canonical_assessor(value: str) -> Assessor:
    try:
        return _ASSESSOR_ALIASES[value]
    except KeyError:
        raise ValueError(f"unknown assessor {value!r}") from None


def citable_for_claim(relation: CanonicalRelation, *, exists: bool, read: bool) -> bool:
    return exists and read and relation not in (
        CanonicalRelation.UNRELATED, CanonicalRelation.UNRESOLVED
    )


@dataclass(frozen=True, slots=True)
class CanonicalAssessment:
    """One source's canonical bearing on one claim/proposition."""

    source_ref: str
    target_ref: str
    relation: CanonicalRelation
    relevance: CanonicalRelevance | None
    assessor: Assessor
    method: str
    method_version: str
    confidence: str = "not_assessed"
    evidence_span: str | None = None
    ontology_version: str = ONTOLOGY_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_ref": self.source_ref,
            "target_ref": self.target_ref,
            "relation": self.relation.value,
            "relevance": self.relevance.value if self.relevance else None,
            "assessor": self.assessor.value,
            "method": self.method,
            "method_version": self.method_version,
            "confidence": self.confidence,
            "evidence_span": self.evidence_span,
            "ontology_version": self.ontology_version,
        }


def from_decision_support(assessment) -> CanonicalAssessment:
    """Dual-read of a persisted ``EvidenceRelationAssessment``."""
    return CanonicalAssessment(
        source_ref=f"{assessment.source_ref.source_class.value}:{assessment.source_ref.source_id}",
        target_ref=f"proposition:{assessment.proposition_id}",
        relation=canonical_relation("decision_support", assessment.relation.value),
        relevance=None,
        assessor=canonical_assessor(getattr(assessment, "assessor", "model")),
        method="submit_grounded_answer.evidence_relations",
        method_version=getattr(assessment, "method_version", None) or "grounded-answer-v2",
        confidence=assessment.strength.value,
    )


def from_report_synthesis(synthesis) -> list[CanonicalAssessment]:
    """Dual-read of a report ``EvidenceSynthesis`` (one row per cited record)."""
    relation = canonical_relation("report", synthesis.relation.value)
    return [
        CanonicalAssessment(
            source_ref=f"evidence:{evidence_id}",
            target_ref=f"report_proposition:{synthesis.synthesis_id}",
            relation=relation, relevance=None, assessor=Assessor.MODEL,
            method="report_synthesis", method_version="toxagent-report-v3",
        )
        for evidence_id in synthesis.evidence_ids
    ]


def from_relevance(assessment, *, evidence_id: str, target_ref: str) -> CanonicalAssessment:
    """A relevance assessment says nothing about bearing: relation is
    ``unresolved`` unless the hit is irrelevant, which is ``unrelated``."""
    relevance = canonical_relevance(assessment.relevance.value)
    return CanonicalAssessment(
        source_ref=f"evidence:{evidence_id}",
        target_ref=target_ref,
        relation=(
            CanonicalRelation.UNRELATED if relevance is CanonicalRelevance.IRRELEVANT
            else CanonicalRelation.UNRESOLVED
        ),
        relevance=relevance,
        assessor=canonical_assessor(assessment.assessor),
        method="research.relevance",
        method_version=assessment.policy_version,
    )

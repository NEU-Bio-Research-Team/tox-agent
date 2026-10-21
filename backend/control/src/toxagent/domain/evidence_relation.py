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

from collections.abc import Sequence
from dataclasses import dataclass
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
    because either is wrong (plan section 9.4).

    ``endpoint``/``species``/``dose``/``use_context`` were not enough to tell
    a real disagreement from two different measurements: two records can both
    say "hERG" and measure different things. The axes below name what makes a
    measurement the same measurement -- which molecule, which protein, and by
    what assay read out in what unit against what comparator.

    A field left ``None`` is *not stated*, not "the same as yours". Comparison
    therefore only looks at axes both sides declare, and a scope that states
    too little to be compared is ``not_assessed`` rather than comparable.
    """

    endpoint: str | None = None
    species: str | None = None
    dose: str | None = None
    use_context: str | None = None
    #: Which molecule. A canonical identifier where a tool gave one; otherwise
    #: the name the sources themselves use. Compared as written, so two
    #: spellings of one molecule read as two molecules -- a false difference,
    #: which is the cheaper error.
    ligand_id: str | None = None
    #: Which protein, and which form of it.
    target_id: str | None = None
    target_variant: str | None = None
    target_sequence_sha256: str | None = None
    #: How it was measured.
    assay_type: str | None = None
    readout: str | None = None
    unit: str | None = None
    #: The relation the value is stated against, e.g. "<" for a censored IC50.
    comparator: str | None = None
    #: Conditions that change what the number means, e.g. the cell system or
    #: temperature a channel assay was run at.
    experimental_context: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "endpoint": self.endpoint,
            "species": self.species,
            "dose": self.dose,
            "use_context": self.use_context,
            "ligand_id": self.ligand_id,
            "target_id": self.target_id,
            "target_variant": self.target_variant,
            "target_sequence_sha256": self.target_sequence_sha256,
            "assay_type": self.assay_type,
            "readout": self.readout,
            "unit": self.unit,
            "comparator": self.comparator,
            "experimental_context": self.experimental_context,
        }

    def declared(self) -> dict[str, str]:
        return {k: v for k, v in self.to_dict().items() if v is not None}

    def differences(self, other: "EvidenceScope") -> tuple[str, ...]:
        """Axes both sides declare and disagree on."""
        mine, theirs = self.declared(), other.declared()
        return tuple(
            axis
            for axis in MEASUREMENT_AXES
            if axis in mine and axis in theirs and mine[axis] != theirs[axis]
        )

    def shared_axes(self, other: "EvidenceScope") -> tuple[str, ...]:
        mine, theirs = self.declared(), other.declared()
        return tuple(
            axis for axis in MEASUREMENT_AXES if axis in mine and axis in theirs
        )

    def compare(
        self, other: "EvidenceScope", *, measurement: bool = False
    ) -> "ScopeComparison":
        """Whether these two can be set against each other, and if not, why.

        Agreeing on nothing is not agreement, and neither is agreeing on one
        thing. Two scopes that share only ``endpoint="herg"`` produce no
        differences, which must not read as "the same measurement" -- that is
        exactly how "both say hERG" becomes a contradiction between a binding
        affinity and a clinical outcome. So each group in
        :data:`FINDING_GROUPS` (and, for two measurements,
        :data:`MEASUREMENT_GROUPS`) needs an axis both sides declare.

        ``measurement`` is for the caller that knows a side is a measured
        value even though its scope does not say how it was measured; a scope
        that does say (:data:`METHOD_AXES`) makes it one by itself.
        """
        differing = self.differences(other)
        shared = self.shared_axes(other)
        mine, theirs = self.declared(), other.declared()
        measured = measurement or any(
            axis in mine or axis in theirs for axis in METHOD_AXES
        )
        if differing:
            return ScopeComparison(
                Comparability.DIFFERENT, shared_axes=shared,
                differing_axes=differing, measurement=measured,
            )
        groups = MEASUREMENT_GROUPS if measured else FINDING_GROUPS
        missing = tuple(
            name for name, axes in groups if not any(axis in shared for axis in axes)
        )
        state = Comparability.NOT_ASSESSED if missing else Comparability.COMPARABLE
        return ScopeComparison(
            state, shared_axes=shared, missing_groups=missing, measurement=measured,
        )

    def comparable_with(
        self, other: "EvidenceScope", *, measurement: bool = False
    ) -> bool:
        return self.compare(other, measurement=measurement).state is Comparability.COMPARABLE


class Comparability(str, Enum):
    """Three answers, not two: "not shown to be the same" is neither "the
    same" nor "different", and only the first licenses a contradiction."""

    COMPARABLE = "comparable"
    DIFFERENT = "different"
    NOT_ASSESSED = "not_assessed"


@dataclass(frozen=True, slots=True)
class ScopeComparison:
    state: Comparability
    shared_axes: tuple[str, ...] = ()
    #: Set when ``state`` is ``different``: axes both declare and disagree on.
    differing_axes: tuple[str, ...] = ()
    #: Set when ``state`` is ``not_assessed``: groups no shared axis covers.
    missing_groups: tuple[str, ...] = ()
    measurement: bool = False


#: The axes that decide whether two sources measured the same thing. A
#: difference on any of them means the two numbers are not commensurable, so a
#: disagreement between them is not yet a contradiction.
#: ``dose`` is included: downgrading a contradiction to a contextual relation
#: with a named axis is the cheaper error. A false contradiction is on the
#: critical-error list; an over-cautious ``contextual`` is an argument the
#: reviewer can reopen.
MEASUREMENT_AXES: tuple[str, ...] = (
    "endpoint",
    "species",
    "dose",
    "use_context",
    "ligand_id",
    "target_id",
    "target_variant",
    "target_sequence_sha256",
    "assay_type",
    "readout",
    "unit",
    "comparator",
    "experimental_context",
)

#: What a comparison has to pin down before a disagreement counts as one.
#: Each group needs at least one axis both sides declare: ``outcome`` is what
#: was observed, ``subject`` is in which molecule or system. These are what a
#: statement in a paper can be expected to say; assay fields are not, so a
#: finding is not held to them.
FINDING_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("outcome", ("endpoint", "target_id")),
    ("subject", (
        "ligand_id", "species", "experimental_context", "use_context", "target_variant",
    )),
)

#: Two measured values also need to share how they were measured: a binding
#: IC50 and a patch-clamp IC50 are different numbers about the same channel.
MEASUREMENT_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = FINDING_GROUPS + (
    ("method", ("assay_type", "readout")),
)

#: A scope that states any of these describes a measurement, so it is held to
#: the measurement groups. The reverse escape -- a measured record described
#: without these -- is closed by the caller's ``measurement`` flag.
METHOD_AXES: tuple[str, ...] = ("assay_type", "readout", "unit", "comparator")


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


_GROUP_HINTS = {
    "outcome": "what was observed (endpoint or target_id)",
    "subject": (
        "in which molecule or system (ligand_id, species, experimental_context, "
        "use_context or target_variant)"
    ),
    "method": "how it was measured (assay_type or readout)",
}


@dataclass(frozen=True, slots=True)
class ScopeConflict:
    """A ``contradicts`` whose scope does not show a disagreement.

    Either the two sources differ on a declared axis (``scope_differs``: the
    relation can be restated as ``contextual`` on that axis instead of being
    argued about in prose), or they have not been shown to be about the same
    thing (``scope_undeclared`` when they share no declared axis at all,
    ``scope_incomplete`` when what they share leaves a group open).
    """

    proposition_id: str
    contradicting_relation_id: str
    other_relation_id: str
    differing_axes: tuple[str, ...]
    #: ``scope_differs``, ``scope_undeclared`` or ``scope_incomplete``.
    reason: str = "scope_differs"
    #: For ``scope_incomplete``/``scope_undeclared``: the groups no shared axis
    #: covers, from :data:`FINDING_GROUPS`/:data:`MEASUREMENT_GROUPS`.
    missing_groups: tuple[str, ...] = ()
    shared_axes: tuple[str, ...] = ()

    @property
    def comparability(self) -> Comparability:
        if self.reason == "scope_differs":
            return Comparability.DIFFERENT
        return Comparability.NOT_ASSESSED

    def describe(self) -> str:
        if self.reason == "scope_differs":
            axes = ", ".join(self.differing_axes)
            return (
                f"relation {self.contradicting_relation_id} contradicts a source whose "
                f"scope differs on {axes}; two sources that did not measure the same "
                "thing are not in disagreement"
            )
        needed = "; ".join(_GROUP_HINTS[group] for group in self.missing_groups)
        if self.reason == "scope_undeclared":
            return (
                f"relation {self.contradicting_relation_id} contradicts a source it "
                "shares no declared measurement axis with, so whether they observed "
                f"the same thing was not assessed; state, for both, {needed}"
            )
        shared = ", ".join(self.shared_axes)
        return (
            f"relation {self.contradicting_relation_id} contradicts a source it shares "
            f"only {shared} with, which does not show they observed the same thing; "
            f"state, for both, {needed}"
        )


def scope_conflicts(
    assessments: Sequence[EvidenceRelationAssessment],
    *,
    measured_source_ids: frozenset[str] = frozenset(),
) -> tuple[ScopeConflict, ...]:
    """Find contradictions that the scopes do not show to be one.

    Within one proposition, a ``contradicts`` set against a ``supports`` is a
    claim that the two sources cannot both hold. That claim needs the two to
    be commensurable: where they differ on a measurement axis, the honest
    label is ``contextual``; where their scopes say too little to tell, it has
    not been assessed. Only the opposite side is compared -- two contradicting
    sources do not disagree with each other, and a ``contextual``,
    ``insufficient`` or ``not_applicable`` source takes no side to disagree
    with.

    ``measured_source_ids`` names sources the caller knows are measured values
    (a record with a value), which are compared as measurements whatever
    their declared scope says.

    Four outcomes are kept apart this way: a real contradiction (comparable
    scope), different measurement conditions (``scope_differs``), indirect
    evidence (``directness``) and not enough data
    (``RelationLabel.INSUFFICIENT``).
    """
    by_proposition: dict[str, list[EvidenceRelationAssessment]] = {}
    for assessment in assessments:
        by_proposition.setdefault(assessment.proposition_id, []).append(assessment)

    conflicts: list[ScopeConflict] = []
    for proposition_id, group in by_proposition.items():
        for assessment in group:
            if assessment.relation is not RelationLabel.CONTRADICTS:
                continue
            for other in group:
                if other.relation is not RelationLabel.SUPPORTS:
                    continue
                comparison = assessment.scope.compare(
                    other.scope,
                    measurement=bool(
                        {assessment.source_ref.source_id, other.source_ref.source_id}
                        & measured_source_ids
                    ),
                )
                if comparison.state is Comparability.COMPARABLE:
                    continue
                if comparison.state is Comparability.DIFFERENT:
                    reason = "scope_differs"
                elif not comparison.shared_axes:
                    # Agreeing on nothing is not agreement.
                    reason = "scope_undeclared"
                else:
                    reason = "scope_incomplete"
                conflicts.append(
                    ScopeConflict(
                        proposition_id=proposition_id,
                        contradicting_relation_id=assessment.id,
                        other_relation_id=other.id,
                        differing_axes=comparison.differing_axes,
                        reason=reason,
                        missing_groups=comparison.missing_groups,
                        shared_axes=comparison.shared_axes,
                    )
                )
    return tuple(conflicts)


def new_proposition_id() -> str:
    return new_id(PROPOSITION)

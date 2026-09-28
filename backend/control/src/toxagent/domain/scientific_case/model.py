"""The scientific case: its limits, vocabulary, entries and ``ScientificCaseV1``."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

SCHEMA_VERSION = "scientific-case-v1"


DOSSIER_SCHEMA_VERSION = "decision-dossier-v1"


MAX_HYPOTHESES = 8


MAX_EVIDENCE = 200


MAX_UNCERTAINTIES = 32


MAX_ACTIONS = 400


MAX_NEXT_TESTS = 12


MAX_CONTEXT = 32


MAX_TEXT = 1000


class InvalidCaseUpdate(ValueError):
    """A refused update. The message is written for the model that sent it."""


class Actor(str, Enum):
    MODEL = "model"
    SERVER = "server"
    USER = "user"


class CaseStatus(str, Enum):
    OPEN = "open"
    CLOSED = "closed"


class Stance(str, Enum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    #: Bears on the question (scope, assay, exposure) without taking a side.
    CONTEXTUAL = "contextual"
    #: Looked at; does not settle anything.
    INSUFFICIENT = "insufficient"


class SourceClass(str, Enum):
    PREDICTOR_FACT = "predictor_fact"
    EXPLANATION_FACT = "explanation_fact"
    EXTERNAL_EXPERIMENTAL = "external_experimental"
    EXTERNAL_REGULATORY = "external_regulatory"
    REPORT_FACT = "report_fact"
    USER_SUPPLIED = "user_supplied"


#: Which ref kind each source class must cite.
REF_KIND: Mapping[str, str] = {
    SourceClass.PREDICTOR_FACT.value: "observation",
    SourceClass.EXPLANATION_FACT.value: "observation",
    SourceClass.EXTERNAL_EXPERIMENTAL.value: "evidence",
    SourceClass.EXTERNAL_REGULATORY.value: "evidence",
    SourceClass.REPORT_FACT.value: "report",
    SourceClass.USER_SUPPLIED.value: "context",
}


#: Sources independent of the model under investigation. Predictor and
#: explanation facts are fallible signals about their own output, not
#: confirmation of each other (harness/context.py).
INDEPENDENT_SOURCES = frozenset({
    SourceClass.EXTERNAL_EXPERIMENTAL.value, SourceClass.EXTERNAL_REGULATORY.value,
    SourceClass.USER_SUPPLIED.value,
})


class Directness(str, Enum):
    #: The same compound and the same endpoint as the question.
    DIRECT = "direct"
    #: A structural analogue or a related compound.
    ANALOGUE = "analogue"
    #: A class-level or target-level statement.
    CLASS_LEVEL = "class_level"
    #: A mechanistic argument, not an observation of the outcome.
    MECHANISTIC_INFERENCE = "mechanistic_inference"
    #: Not direct, kind unstated (an answer relation's ``indirect``).
    INDIRECT = "indirect"
    NOT_ASSESSED = "not_assessed"


class HypothesisKind(str, Enum):
    MODEL_SIGNAL = "model_signal"
    MECHANISM = "mechanism"
    ASSAY_OR_EXPOSURE = "assay_or_exposure"
    DATA_INSUFFICIENT = "data_insufficient"
    ALTERNATIVE_EXPLANATION = "alternative_explanation"
    OTHER = "other"


class HypothesisStatus(str, Enum):
    OPEN = "open"
    SUPPORTED = "supported"
    #: Evidence against, not enough to refute.
    WEAKENED = "weakened"
    REFUTED = "refuted"
    #: Cannot be settled with the data the case can reach.
    UNRESOLVABLE = "unresolvable"


class UncertaintyKind(str, Enum):
    MISSING_ENDPOINT = "missing_endpoint"
    MODEL_CALIBRATION = "model_calibration"
    APPLICABILITY_DOMAIN = "applicability_domain"
    EXPLAINER_FAITHFULNESS = "explainer_faithfulness"
    OCR_AMBIGUITY = "ocr_ambiguity"
    CONFLICTING_SOURCES = "conflicting_sources"
    ASSAY_MISMATCH = "assay_mismatch"
    MISSING_EXPOSURE = "missing_exposure"
    MISSING_DATA = "missing_data"
    PROVIDER_COVERAGE = "provider_coverage"
    OTHER = "other"


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    #: The conclusion cannot be drawn until this is resolved.
    BLOCKING = "blocking"


class ActionDecision(str, Enum):
    CONTINUE = "continue"
    REDIRECT = "redirect"
    ASK_USER = "ask_user"
    ANSWER = "answer"
    STOP = "stop"


def _values(enum: type[Enum]) -> list[str]:
    return [member.value for member in enum]


@dataclass(frozen=True, slots=True)
class Hypothesis:
    id: str
    statement: str
    kind: str
    #: What observation would refute it. Required: a hypothesis nothing could
    #: refute is not one the case can investigate.
    refutation_condition: str
    status: str = HypothesisStatus.OPEN.value
    status_reason: str = ""
    actor: str = Actor.MODEL.value
    run_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "statement": self.statement, "kind": self.kind,
            "refutation_condition": self.refutation_condition, "status": self.status,
            "status_reason": self.status_reason, "actor": self.actor, "run_id": self.run_id,
        }


@dataclass(frozen=True, slots=True)
class EvidenceEntry:
    id: str
    claim: str
    source_class: str
    #: ``observation:obs_…`` / ``evidence:evd_…`` / ``report:rpt_…`` / ``context:c1``.
    source_ref: str
    stance: str
    directness: str
    hypothesis_ids: tuple[str, ...] = ()
    #: Where in the source: a field path, a quoted span, a section.
    locator: str | None = None
    #: endpoint / species / assay / dose / use_context, as far as known.
    scope: Mapping[str, str] = field(default_factory=dict)
    actor: str = Actor.MODEL.value
    run_id: str | None = None

    @property
    def independent(self) -> bool:
        return self.source_class in INDEPENDENT_SOURCES

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "claim": self.claim, "source_class": self.source_class,
            "source_ref": self.source_ref, "stance": self.stance, "directness": self.directness,
            "hypothesis_ids": list(self.hypothesis_ids), "locator": self.locator,
            "scope": dict(self.scope), "actor": self.actor, "run_id": self.run_id,
        }


@dataclass(frozen=True, slots=True)
class Uncertainty:
    id: str
    kind: str
    description: str
    severity: str
    hypothesis_ids: tuple[str, ...] = ()
    status: str = "open"
    resolution: str = ""
    resolving_refs: tuple[str, ...] = ()
    actor: str = Actor.MODEL.value
    run_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "kind": self.kind, "description": self.description,
            "severity": self.severity, "hypothesis_ids": list(self.hypothesis_ids),
            "status": self.status, "resolution": self.resolution,
            "resolving_refs": list(self.resolving_refs), "actor": self.actor,
            "run_id": self.run_id,
        }


@dataclass(frozen=True, slots=True)
class Action:
    id: str
    action: str
    purpose: str
    decision: str
    outcome: str = ""
    hypothesis_ids: tuple[str, ...] = ()
    actor: str = Actor.MODEL.value
    run_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "action": self.action, "purpose": self.purpose,
            "decision": self.decision, "outcome": self.outcome,
            "hypothesis_ids": list(self.hypothesis_ids), "actor": self.actor,
            "run_id": self.run_id,
        }


@dataclass(frozen=True, slots=True)
class NextTest:
    id: str
    test: str
    rationale: str
    #: Hypotheses whose status the result would change. At least one: a test
    #: that discriminates nothing is not worth proposing (RETHINK §3.2).
    discriminates: tuple[str, ...]
    #: "If the readout is X, then …" — what each outcome would mean.
    expected_readouts: tuple[str, ...] = ()
    actor: str = Actor.MODEL.value
    run_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "test": self.test, "rationale": self.rationale,
            "discriminates": list(self.discriminates),
            "expected_readouts": list(self.expected_readouts), "actor": self.actor,
            "run_id": self.run_id,
        }


@dataclass(frozen=True, slots=True)
class ContextItem:
    """Something the researcher told the case: an assay result, an exposure."""

    id: str
    key: str
    value: str
    note: str = ""
    actor: str = Actor.USER.value
    run_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "key": self.key, "value": self.value, "note": self.note,
                "actor": self.actor, "run_id": self.run_id}


@dataclass(frozen=True, slots=True)
class ConclusionLine:
    text: str
    evidence_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "evidence_ids": list(self.evidence_ids)}


@dataclass(frozen=True, slots=True)
class Conclusion:
    """What the case can say so far, conditionally. Replaced, never merged."""

    can_say: tuple[ConclusionLine, ...] = ()
    cannot_say: tuple[str, ...] = ()
    what_would_change: tuple[str, ...] = ()
    #: Who must approve any step outside the system (RETHINK §3.1 item 7).
    approver: str = ""
    run_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "can_say": [line.to_dict() for line in self.can_say],
            "cannot_say": list(self.cannot_say),
            "what_would_change": list(self.what_would_change),
            "approver": self.approver, "run_id": self.run_id,
        }


@dataclass(frozen=True, slots=True)
class DataScope:
    """What the case may reach (RETHINK §3.1 item 2). The researcher sets it.

    ``external_search`` is the one scope the product can enforce today: whether
    the compound may be sent to an external literature provider. A confidential
    structure is the reason to turn it off; the search tool then refuses, so
    this is a permission, not advice to the model (W9-07).
    """

    external_search: bool = True
    reason: str = ""
    run_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"external_search": self.external_search, "reason": self.reason,
                "run_id": self.run_id}


@dataclass(frozen=True, slots=True)
class RunRecord:
    run_id: str
    goal: str
    stop_reason: str | None = None
    answer_id: str | None = None
    usage: Mapping[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"run_id": self.run_id, "goal": self.goal, "stop_reason": self.stop_reason,
                "answer_id": self.answer_id, "usage": dict(self.usage)}


@dataclass(frozen=True, slots=True)
class CaseUpdate:
    """One entry of the append-only log. ``revision`` is the case's after it."""

    op: str
    payload: Mapping[str, Any]
    actor: str
    at: str
    run_id: str | None = None
    revision: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {"op": self.op, "payload": dict(self.payload), "actor": self.actor,
                "at": self.at, "run_id": self.run_id, "revision": self.revision}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CaseUpdate":
        return cls(op=data["op"], payload=dict(data.get("payload") or {}), actor=data["actor"],
                   at=data["at"], run_id=data.get("run_id"), revision=int(data.get("revision", 0)))


@dataclass(frozen=True, slots=True)
class ScientificCaseV1:
    id: str
    session_id: str
    #: Scopes a case to one subject; a subject switch opens another case, so
    #: one compound's evidence never lands in another's ledger.
    subject_key: str
    question: str = ""
    decision_context: str = ""
    subject_refs: tuple[str, ...] = ()
    #: Who the investigation is for: the session owner's subject id, recorded
    #: by the server when the case opens (RETHINK §3.1 item 2).
    requester: str = ""
    data_scope: DataScope = field(default_factory=DataScope)
    status: str = CaseStatus.OPEN.value
    context: tuple[ContextItem, ...] = ()
    hypotheses: tuple[Hypothesis, ...] = ()
    evidence: tuple[EvidenceEntry, ...] = ()
    uncertainties: tuple[Uncertainty, ...] = ()
    actions: tuple[Action, ...] = ()
    next_tests: tuple[NextTest, ...] = ()
    conclusion: Conclusion = field(default_factory=Conclusion)
    runs: tuple[RunRecord, ...] = ()
    revision: int = 0
    created_at: str = ""
    updated_at: str = ""
    schema_version: str = SCHEMA_VERSION

    # ------------------------------------------------------------ reading

    def hypothesis(self, hypothesis_id: str) -> Hypothesis | None:
        return next((h for h in self.hypotheses if h.id == hypothesis_id), None)

    def evidence_for(self, hypothesis_id: str, stance: str) -> tuple[EvidenceEntry, ...]:
        return tuple(e for e in self.evidence
                     if hypothesis_id in e.hypothesis_ids and e.stance == stance)

    @property
    def open_uncertainties(self) -> tuple[Uncertainty, ...]:
        return tuple(u for u in self.uncertainties if u.status == "open")

    @property
    def coverage(self) -> dict[str, int]:
        """Ref coverage and quality coverage, side by side and never combined."""
        with_ref = with_independent_direct = with_counter = 0
        counter_searched = {
            h for a in self.actions if a.action.startswith("counterevidence") for h in a.hypothesis_ids
        }
        for h in self.hypotheses:
            linked = [e for e in self.evidence if h.id in e.hypothesis_ids]
            if linked:
                with_ref += 1
            if any(e.independent and e.directness == Directness.DIRECT.value for e in linked):
                with_independent_direct += 1
            # "Considered" is either an entry against it or a recorded search
            # for one; a search that found nothing is still a search.
            if h.id in counter_searched or any(
                e.stance == Stance.CONTRADICTS.value for e in linked
            ):
                with_counter += 1
        return {
            "hypotheses": len(self.hypotheses),
            "with_any_source": with_ref,
            "with_independent_direct_evidence": with_independent_direct,
            "with_counterevidence_considered": with_counter,
            "open_uncertainties": len(self.open_uncertainties),
            "blocking_uncertainties": sum(
                1 for u in self.open_uncertainties if u.severity == Severity.BLOCKING.value
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "case_id": self.id, "session_id": self.session_id, "subject_key": self.subject_key,
            "question": self.question, "decision_context": self.decision_context,
            "subject_refs": list(self.subject_refs), "status": self.status,
            "requester": self.requester, "data_scope": self.data_scope.to_dict(),
            "context": [c.to_dict() for c in self.context],
            "hypotheses": [h.to_dict() for h in self.hypotheses],
            "evidence": [e.to_dict() for e in self.evidence],
            "uncertainties": [u.to_dict() for u in self.uncertainties],
            "actions": [a.to_dict() for a in self.actions],
            "next_tests": [t.to_dict() for t in self.next_tests],
            "conclusion": self.conclusion.to_dict(),
            "runs": [r.to_dict() for r in self.runs],
            "coverage": self.coverage,
            "revision": self.revision, "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ScientificCaseV1":
        conclusion = data.get("conclusion") or {}
        return cls(
            id=data["case_id"], session_id=data["session_id"], subject_key=data["subject_key"],
            question=data.get("question", ""), decision_context=data.get("decision_context", ""),
            subject_refs=tuple(data.get("subject_refs") or ()),
            requester=data.get("requester", ""),
            data_scope=DataScope(**(data.get("data_scope") or {})),
            status=data.get("status", CaseStatus.OPEN.value),
            context=tuple(ContextItem(**c) for c in data.get("context") or ()),
            hypotheses=tuple(Hypothesis(**h) for h in data.get("hypotheses") or ()),
            evidence=tuple(
                EvidenceEntry(**{**e, "hypothesis_ids": tuple(e.get("hypothesis_ids") or ()),
                                 "scope": dict(e.get("scope") or {})})
                for e in data.get("evidence") or ()
            ),
            uncertainties=tuple(
                Uncertainty(**{**u, "hypothesis_ids": tuple(u.get("hypothesis_ids") or ()),
                               "resolving_refs": tuple(u.get("resolving_refs") or ())})
                for u in data.get("uncertainties") or ()
            ),
            actions=tuple(
                Action(**{**a, "hypothesis_ids": tuple(a.get("hypothesis_ids") or ())})
                for a in data.get("actions") or ()
            ),
            next_tests=tuple(
                NextTest(**{**t, "discriminates": tuple(t.get("discriminates") or ()),
                            "expected_readouts": tuple(t.get("expected_readouts") or ())})
                for t in data.get("next_tests") or ()
            ),
            conclusion=Conclusion(
                can_say=tuple(
                    ConclusionLine(text=line["text"], evidence_ids=tuple(line.get("evidence_ids") or ()))
                    for line in conclusion.get("can_say") or ()
                ),
                cannot_say=tuple(conclusion.get("cannot_say") or ()),
                what_would_change=tuple(conclusion.get("what_would_change") or ()),
                approver=conclusion.get("approver", ""), run_id=conclusion.get("run_id"),
            ),
            runs=tuple(
                RunRecord(run_id=r["run_id"], goal=r.get("goal", ""),
                          stop_reason=r.get("stop_reason"), answer_id=r.get("answer_id"),
                          usage=dict(r.get("usage") or {}))
                for r in data.get("runs") or ()
            ),
            revision=int(data.get("revision", 0)), created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
        )


def subject_key(analysis_id: str | None) -> str:
    return f"analysis:{analysis_id}" if analysis_id else "none"

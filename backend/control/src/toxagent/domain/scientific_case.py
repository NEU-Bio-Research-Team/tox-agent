"""ScientificCaseV1: the durable record of one investigation, across turns.

RETHINK §3.1 and §4.4 step 1. ``DecisionSupportStateV1`` records what *one run*
set out to establish; it ends with the run. A researcher's question does not:
they come back with an assay result, the compound's exposure, a second paper.
This is the state that survives — the decision question, the competing
hypotheses and what would refute each, a ledger of evidence for and against, a
ledger of what is still unknown, why the agent did what it did, and the
conclusion it could support so far.

Rules that make it a research record rather than a transcript:

* **Append-only.** A case is the fold of its updates (``replay``). Nothing is
  overwritten; a hypothesis that was supported and later refuted keeps both
  updates in the log, with the run that made each.
* **Evidence is an artifact, never prose.** A ledger entry names a source the
  session really has (an observation, an evidence record, a report, or context
  the user supplied). There is no ``agent_synthesis`` source class here: an
  inference is a hypothesis, not evidence for one (RETHINK §4.3, last
  paragraph).
* **A status needs its evidence.** A hypothesis is ``supported`` or
  ``refuted`` only with a ledger entry of that stance behind it, and a
  conclusion line the case can say must name the entries it rests on.
* **Ref coverage is not quality coverage.** ``coverage`` reports how many
  hypotheses have *any* source separately from how many have direct,
  independent evidence and how many had counter-evidence considered. It is
  never folded into one number.

Ids inside a case are short and server-issued (``h1``, ``e3``, ``u2``); the
model never names one that does not exist. Every transition is a pure function,
so a case can be rebuilt and tested without a database.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

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


# --------------------------------------------------------------- the records

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


# ------------------------------------------------------------ validation

def _text(payload: Mapping[str, Any], key: str, *, required: bool = True,
          limit: int = MAX_TEXT) -> str:
    value = payload.get(key)
    if value is None or (isinstance(value, str) and not value.strip()):
        if required:
            raise InvalidCaseUpdate(f"{key} is required")
        return ""
    if not isinstance(value, str):
        raise InvalidCaseUpdate(f"{key} must be text")
    value = value.strip()
    if len(value) > limit:
        raise InvalidCaseUpdate(f"{key} exceeds {limit} characters")
    return value


def _choice(payload: Mapping[str, Any], key: str, enum: type[Enum], default: str | None = None) -> str:
    value = payload.get(key, default)
    allowed = _values(enum)
    if value not in allowed:
        raise InvalidCaseUpdate(f"{key} must be one of {allowed}; got {value!r}")
    return value


def _texts(payload: Mapping[str, Any], key: str, *, limit: int = 8) -> tuple[str, ...]:
    raw = payload.get(key) or ()
    if isinstance(raw, str) or not isinstance(raw, (list, tuple)):
        raise InvalidCaseUpdate(f"{key} must be a list of text")
    if len(raw) > limit:
        raise InvalidCaseUpdate(f"{key} holds at most {limit} items")
    out: list[str] = []
    for index, item in enumerate(raw):
        if not isinstance(item, str) or not item.strip():
            raise InvalidCaseUpdate(f"{key}[{index}] must be non-empty text")
        if len(item) > MAX_TEXT:
            raise InvalidCaseUpdate(f"{key}[{index}] exceeds {MAX_TEXT} characters")
        out.append(item.strip())
    return tuple(out)


def _known_ids(case: ScientificCaseV1, raw: Any, *, kind: str, key: str) -> tuple[str, ...]:
    ids = tuple(dict.fromkeys(raw or ()))
    known = {
        "hypothesis": {h.id for h in case.hypotheses},
        "evidence": {e.id for e in case.evidence},
        "uncertainty": {u.id for u in case.uncertainties},
    }[kind]
    unknown = [i for i in ids if i not in known]
    if unknown:
        raise InvalidCaseUpdate(
            f"{key} names {kind} ids this case does not have: {unknown}; known: {sorted(known)}"
        )
    return ids


def _next_local_id(prefix: str, existing: Iterable[str]) -> str:
    highest = 0
    for value in existing:
        if value.startswith(prefix) and value[len(prefix):].isdigit():
            highest = max(highest, int(value[len(prefix):]))
    return f"{prefix}{highest + 1}"


def parse_ref(ref: str) -> tuple[str, str]:
    kind, sep, identifier = str(ref).partition(":")
    if not sep or not identifier:
        raise InvalidCaseUpdate(f"source_ref must look like 'kind:id'; got {ref!r}")
    return kind, identifier


def _scope(payload: Mapping[str, Any]) -> dict[str, str]:
    raw = payload.get("scope") or {}
    if not isinstance(raw, Mapping):
        raise InvalidCaseUpdate("scope must be an object")
    allowed = {"endpoint", "species", "assay", "dose", "use_context", "compound"}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise InvalidCaseUpdate(f"scope keys must be among {sorted(allowed)}; got {unknown}")
    return {k: str(v)[:200] for k, v in raw.items() if v not in (None, "")}


# ------------------------------------------------------------ transitions

def _open(case: ScientificCaseV1 | None, u: CaseUpdate) -> ScientificCaseV1:
    if case is not None:
        raise InvalidCaseUpdate("the case is already open")
    p = u.payload
    return ScientificCaseV1(
        id=str(p["case_id"]), session_id=str(p["session_id"]), subject_key=str(p["subject_key"]),
        question=_text(p, "question", required=False, limit=2000),
        decision_context=_text(p, "decision_context", required=False),
        subject_refs=tuple(dict.fromkeys(p.get("subject_refs") or ())),
        created_at=u.at,
    )


def _set_question(case: ScientificCaseV1, u: CaseUpdate) -> ScientificCaseV1:
    return replace(
        case, question=_text(u.payload, "question", limit=2000),
        decision_context=_text(u.payload, "decision_context", required=False)
        or case.decision_context,
    )


def _add_context(case: ScientificCaseV1, u: CaseUpdate) -> ScientificCaseV1:
    if len(case.context) >= MAX_CONTEXT:
        raise InvalidCaseUpdate(f"a case holds at most {MAX_CONTEXT} context items")
    item = ContextItem(
        id=_next_local_id("c", (c.id for c in case.context)),
        key=_text(u.payload, "key", limit=120), value=_text(u.payload, "value"),
        note=_text(u.payload, "note", required=False), actor=u.actor, run_id=u.run_id,
    )
    return replace(case, context=case.context + (item,))


def _add_hypothesis(case: ScientificCaseV1, u: CaseUpdate) -> ScientificCaseV1:
    if len(case.hypotheses) >= MAX_HYPOTHESES:
        raise InvalidCaseUpdate(
            f"a case holds at most {MAX_HYPOTHESES} hypotheses; revise or refute one instead"
        )
    statement = _text(u.payload, "statement", limit=500)
    if any(h.statement.casefold() == statement.casefold() for h in case.hypotheses):
        raise InvalidCaseUpdate("this hypothesis is already in the case")
    hypothesis = Hypothesis(
        id=_next_local_id("h", (h.id for h in case.hypotheses)), statement=statement,
        kind=_choice(u.payload, "kind", HypothesisKind),
        refutation_condition=_text(u.payload, "refutation_condition", limit=500),
        actor=u.actor, run_id=u.run_id,
    )
    return replace(case, hypotheses=case.hypotheses + (hypothesis,))


def _revise_hypothesis(case: ScientificCaseV1, u: CaseUpdate) -> ScientificCaseV1:
    target = _known_ids(case, [u.payload.get("hypothesis_id")], kind="hypothesis",
                        key="hypothesis_id")[0]
    status = _choice(u.payload, "status", HypothesisStatus)
    reason = _text(u.payload, "reason", limit=500)
    needed = {
        HypothesisStatus.SUPPORTED.value: Stance.SUPPORTS.value,
        HypothesisStatus.REFUTED.value: Stance.CONTRADICTS.value,
        HypothesisStatus.WEAKENED.value: Stance.CONTRADICTS.value,
    }.get(status)
    if needed and not case.evidence_for(target, needed):
        raise InvalidCaseUpdate(
            f"{target} cannot become {status} without a ledger entry that {needed} it; "
            "record_evidence first, or say why it is unresolvable"
        )
    hypotheses = tuple(
        replace(h, status=status, status_reason=reason) if h.id == target else h
        for h in case.hypotheses
    )
    return replace(case, hypotheses=hypotheses)


def _record_evidence(case: ScientificCaseV1, u: CaseUpdate) -> ScientificCaseV1:
    if len(case.evidence) >= MAX_EVIDENCE:
        raise InvalidCaseUpdate(f"a case holds at most {MAX_EVIDENCE} evidence entries")
    p = u.payload
    source_class = _choice(p, "source_class", SourceClass)
    source_ref = _text(p, "source_ref", limit=120)
    kind, identifier = parse_ref(source_ref)
    if kind != REF_KIND[source_class]:
        raise InvalidCaseUpdate(
            f"a {source_class} entry cites a '{REF_KIND[source_class]}:' ref, not '{kind}:'"
        )
    if kind == "context" and identifier not in {c.id for c in case.context}:
        raise InvalidCaseUpdate(f"this case has no context item {identifier!r}")
    hypothesis_ids = _known_ids(case, p.get("hypothesis_ids"), kind="hypothesis",
                                key="hypothesis_ids")
    stance = _choice(p, "stance", Stance)
    if stance in (Stance.SUPPORTS.value, Stance.CONTRADICTS.value) and not hypothesis_ids:
        raise InvalidCaseUpdate(f"an entry that {stance} something names the hypothesis_ids it bears on")
    entry = EvidenceEntry(
        id=_next_local_id("e", (e.id for e in case.evidence)), claim=_text(p, "claim"),
        source_class=source_class, source_ref=f"{kind}:{identifier}", stance=stance,
        directness=_choice(p, "directness", Directness, Directness.NOT_ASSESSED.value),
        hypothesis_ids=hypothesis_ids, locator=_text(p, "locator", required=False, limit=300) or None,
        scope=_scope(p), actor=u.actor, run_id=u.run_id,
    )
    duplicate = next((
        e for e in case.evidence
        if (e.source_ref, e.claim.casefold(), e.stance, e.hypothesis_ids)
        == (entry.source_ref, entry.claim.casefold(), entry.stance, entry.hypothesis_ids)
    ), None)
    if duplicate is not None:
        return case  # idempotent: the same entry recorded twice is one entry
    return replace(case, evidence=case.evidence + (entry,))


def _record_uncertainty(case: ScientificCaseV1, u: CaseUpdate) -> ScientificCaseV1:
    if len(case.uncertainties) >= MAX_UNCERTAINTIES:
        raise InvalidCaseUpdate(f"a case holds at most {MAX_UNCERTAINTIES} uncertainties")
    p = u.payload
    item = Uncertainty(
        id=_next_local_id("u", (x.id for x in case.uncertainties)),
        kind=_choice(p, "kind", UncertaintyKind), description=_text(p, "description"),
        severity=_choice(p, "severity", Severity, Severity.MEDIUM.value),
        hypothesis_ids=_known_ids(case, p.get("hypothesis_ids"), kind="hypothesis",
                                  key="hypothesis_ids"),
        actor=u.actor, run_id=u.run_id,
    )
    if any((x.kind, x.description.casefold()) == (item.kind, item.description.casefold())
           and x.status == "open" for x in case.uncertainties):
        return case
    return replace(case, uncertainties=case.uncertainties + (item,))


def _resolve_uncertainty(case: ScientificCaseV1, u: CaseUpdate) -> ScientificCaseV1:
    target = _known_ids(case, [u.payload.get("uncertainty_id")], kind="uncertainty",
                        key="uncertainty_id")[0]
    resolution = _text(u.payload, "resolution")
    refs = _known_ids(case, u.payload.get("evidence_ids"), kind="evidence", key="evidence_ids")
    items = tuple(
        replace(x, status="resolved", resolution=resolution, resolving_refs=refs)
        if x.id == target else x
        for x in case.uncertainties
    )
    return replace(case, uncertainties=items)


def _record_action(case: ScientificCaseV1, u: CaseUpdate) -> ScientificCaseV1:
    if len(case.actions) >= MAX_ACTIONS:
        return case  # the log is bounded; the run's tool calls remain the full record
    p = u.payload
    action = Action(
        id=_next_local_id("a", (a.id for a in case.actions)),
        action=_text(p, "action", limit=120), purpose=_text(p, "purpose", limit=500),
        decision=_choice(p, "decision", ActionDecision, ActionDecision.CONTINUE.value),
        outcome=_text(p, "outcome", required=False, limit=500),
        hypothesis_ids=_known_ids(case, p.get("hypothesis_ids"), kind="hypothesis",
                                  key="hypothesis_ids"),
        actor=u.actor, run_id=u.run_id,
    )
    return replace(case, actions=case.actions + (action,))


def _propose_next_test(case: ScientificCaseV1, u: CaseUpdate) -> ScientificCaseV1:
    if len(case.next_tests) >= MAX_NEXT_TESTS:
        raise InvalidCaseUpdate(f"a case holds at most {MAX_NEXT_TESTS} proposed tests")
    p = u.payload
    discriminates = _known_ids(case, p.get("discriminates"), kind="hypothesis", key="discriminates")
    if not discriminates:
        raise InvalidCaseUpdate(
            "a proposed test names the hypotheses its result would discriminate between"
        )
    test = NextTest(
        id=_next_local_id("t", (t.id for t in case.next_tests)), test=_text(p, "test", limit=500),
        rationale=_text(p, "rationale"), discriminates=discriminates,
        expected_readouts=_texts(p, "expected_readouts", limit=6), actor=u.actor, run_id=u.run_id,
    )
    return replace(case, next_tests=case.next_tests + (test,))


def _set_conclusion(case: ScientificCaseV1, u: CaseUpdate) -> ScientificCaseV1:
    p = u.payload
    raw_lines = p.get("can_say") or ()
    if not isinstance(raw_lines, (list, tuple)) or len(raw_lines) > 8:
        raise InvalidCaseUpdate("can_say is a list of at most 8 lines")
    lines: list[ConclusionLine] = []
    for index, raw in enumerate(raw_lines):
        if not isinstance(raw, Mapping):
            raise InvalidCaseUpdate(f"can_say[{index}] must be an object with text and evidence_ids")
        evidence_ids = _known_ids(case, raw.get("evidence_ids"), kind="evidence",
                                  key=f"can_say[{index}].evidence_ids")
        if not evidence_ids:
            raise InvalidCaseUpdate(
                f"can_say[{index}] names no evidence; a line the case cannot trace to its "
                "ledger belongs in cannot_say or is a hypothesis"
            )
        lines.append(ConclusionLine(text=_text(raw, "text"), evidence_ids=evidence_ids))
    conclusion = Conclusion(
        can_say=tuple(lines), cannot_say=_texts(p, "cannot_say"),
        what_would_change=_texts(p, "what_would_change"),
        approver=_text(p, "approver", required=False, limit=200), run_id=u.run_id,
    )
    return replace(case, conclusion=conclusion)


def _run_id(u: CaseUpdate) -> str:
    run_id = u.run_id or u.payload.get("run_id")
    if not run_id:
        raise InvalidCaseUpdate(f"{u.op} needs the run it is about")
    return str(run_id)


def _attach_run(case: ScientificCaseV1, u: CaseUpdate) -> ScientificCaseV1:
    run_id = _run_id(u)
    if any(r.run_id == run_id for r in case.runs):
        return case
    record = RunRecord(run_id=run_id, goal=_text(u.payload, "goal", required=False, limit=2000))
    return replace(case, runs=case.runs + (record,))


def _finish_run(case: ScientificCaseV1, u: CaseUpdate) -> ScientificCaseV1:
    run_id = _run_id(u)
    runs = tuple(
        replace(r, stop_reason=u.payload.get("stop_reason"), answer_id=u.payload.get("answer_id"),
                usage=dict(u.payload.get("usage") or {}))
        if r.run_id == run_id else r
        for r in case.runs
    )
    return replace(case, runs=runs)


def _close(case: ScientificCaseV1, u: CaseUpdate) -> ScientificCaseV1:
    return replace(case, status=CaseStatus.CLOSED.value)


_OPS = {
    "set_question": (_set_question, {Actor.MODEL, Actor.USER}),
    "add_context": (_add_context, {Actor.USER}),
    "add_hypothesis": (_add_hypothesis, {Actor.MODEL, Actor.USER}),
    "revise_hypothesis": (_revise_hypothesis, {Actor.MODEL, Actor.USER}),
    "record_evidence": (_record_evidence, {Actor.MODEL, Actor.USER, Actor.SERVER}),
    "record_uncertainty": (_record_uncertainty, {Actor.MODEL, Actor.USER, Actor.SERVER}),
    "resolve_uncertainty": (_resolve_uncertainty, {Actor.MODEL, Actor.USER}),
    "record_action": (_record_action, {Actor.MODEL, Actor.SERVER}),
    "propose_next_test": (_propose_next_test, {Actor.MODEL, Actor.USER}),
    "set_conclusion": (_set_conclusion, {Actor.MODEL}),
    "attach_run": (_attach_run, {Actor.SERVER}),
    "finish_run": (_finish_run, {Actor.SERVER}),
    "close": (_close, {Actor.USER, Actor.SERVER}),
}

#: The operations a model may send through update_scientific_case.
MODEL_OPS = tuple(sorted(op for op, (_, actors) in _OPS.items() if Actor.MODEL in actors))


def apply(case: ScientificCaseV1 | None, update: CaseUpdate) -> ScientificCaseV1:
    """Apply one update. Returns ``case`` itself when the update is a no-op
    (an idempotent repeat), so a caller can skip writing it."""
    if update.op == "open":
        opened = _open(case, update)
        return replace(opened, revision=1, updated_at=update.at)
    if case is None:
        raise InvalidCaseUpdate("the case is not open")
    try:
        transition, actors = _OPS[update.op]
    except KeyError:
        raise InvalidCaseUpdate(f"unknown operation {update.op!r}; one of {sorted(_OPS)}") from None
    if update.actor not in {a.value for a in actors}:
        raise InvalidCaseUpdate(f"{update.op} cannot be performed by the {update.actor}")
    if case.status == CaseStatus.CLOSED.value and update.op not in ("finish_run",):
        raise InvalidCaseUpdate("the case is closed")
    updated = transition(case, update)
    if updated is case:
        return case
    return replace(updated, revision=case.revision + 1, updated_at=update.at)


def replay(updates: Sequence[CaseUpdate]) -> ScientificCaseV1:
    """Rebuild a case from its log. The stored snapshot must equal this."""
    case: ScientificCaseV1 | None = None
    for update in updates:
        case = apply(case, update)
    if case is None:
        raise InvalidCaseUpdate("an empty log is not a case")
    return case


def subject_key(analysis_id: str | None) -> str:
    return f"analysis:{analysis_id}" if analysis_id else "none"


# ----------------------------------------------- answers flow into the ledger

#: Relation labels of an accepted answer (evidence ontology) -> ledger stance.
_STANCE_BY_RELATION = {
    "supports": Stance.SUPPORTS.value,
    "contradicts": Stance.CONTRADICTS.value,
    "contextual": Stance.CONTEXTUAL.value,
    "insufficient": Stance.INSUFFICIENT.value,
    "not_applicable": Stance.CONTEXTUAL.value,
}

#: grounded-answer-v2 ``directness`` -> ledger directness.
_DIRECTNESS_BY_RELATION = {
    "direct": Directness.DIRECT.value,
    "indirect": Directness.INDIRECT.value,
}


def updates_from_answer(
    case: ScientificCaseV1, relations: Sequence[Mapping[str, Any]], *, run_id: str, at: str,
) -> list[CaseUpdate]:
    """The accepted answer's evidence relations as server-recorded ledger entries.

    The answer validator already resolved every ``source_id``. Synthesis
    relations are skipped: in the case, an inference is not evidence. A
    relation whose proposition matches a hypothesis statement is linked to it;
    otherwise it is recorded unlinked, which the dossier shows as such.
    """
    by_statement = {h.statement.casefold(): h.id for h in case.hypotheses}
    updates: list[CaseUpdate] = []
    for relation in relations:
        source_class = relation.get("source_class")
        if source_class not in REF_KIND or source_class == SourceClass.USER_SUPPLIED.value:
            continue
        source_id = relation.get("source_id")
        if not source_id:
            continue
        stance = _STANCE_BY_RELATION.get(str(relation.get("relation")), Stance.CONTEXTUAL.value)
        hypothesis = by_statement.get(str(relation.get("proposition", "")).strip().casefold())
        if stance in (Stance.SUPPORTS.value, Stance.CONTRADICTS.value) and hypothesis is None:
            # Unlinked, it can only be context: a stance needs a hypothesis to bear on.
            stance = Stance.CONTEXTUAL.value
        scope = {k: str(relation[k]) for k in ("endpoint", "species", "dose", "use_context")
                 if relation.get(k)}
        updates.append(CaseUpdate(
            op="record_evidence", actor=Actor.SERVER.value, run_id=run_id, at=at,
            payload={
                "claim": str(relation.get("proposition", "")).strip()[:MAX_TEXT] or "(no proposition)",
                "source_class": source_class,
                "source_ref": f"{REF_KIND[source_class]}:{source_id}",
                "stance": stance,
                "directness": _DIRECTNESS_BY_RELATION.get(
                    str(relation.get("directness")), Directness.NOT_ASSESSED.value
                ),
                "hypothesis_ids": [hypothesis] if hypothesis else [],
                "scope": scope,
            },
        ))
    return updates


# ------------------------------------------------------------- the dossier

def compile_dossier(
    case: ScientificCaseV1, *, run_id: str, stop_reason: str | None, answer_id: str | None,
    explainer_statements: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """``DecisionDossierV1``: the typed product of one run over its case.

    RETHINK §4.4 step 3. Chat, report and UI are views of this. The three
    explanation layers are kept apart (§4.3): what the model's attribution
    shows (with the explainer's measured verdict), what independent evidence
    says for and against each hypothesis, and why the agent acted and stopped.
    """
    statements = explainer_statements or {}
    ledger = {e.id: e for e in case.evidence}

    def entries(items: Iterable[EvidenceEntry]) -> list[dict[str, Any]]:
        return [e.to_dict() for e in items]

    hypotheses = []
    for h in case.hypotheses:
        hypotheses.append({
            **h.to_dict(),
            "evidence_for": entries(case.evidence_for(h.id, Stance.SUPPORTS.value)),
            "evidence_against": entries(case.evidence_for(h.id, Stance.CONTRADICTS.value)),
            "context": entries(
                e for e in case.evidence
                if h.id in e.hypothesis_ids
                and e.stance in (Stance.CONTEXTUAL.value, Stance.INSUFFICIENT.value)
            ),
        })
    linked = {i for h in case.hypotheses for e in case.evidence if h.id in e.hypothesis_ids
              for i in (e.id,)}
    return {
        "schema_version": DOSSIER_SCHEMA_VERSION,
        "case_id": case.id,
        "case_revision": case.revision,
        "run_id": run_id,
        "answer_id": answer_id,
        "question": case.question,
        "decision_context": case.decision_context,
        "subject_refs": list(case.subject_refs),
        "context": [c.to_dict() for c in case.context],
        "predictor_facts": entries(
            e for e in case.evidence if e.source_class == SourceClass.PREDICTOR_FACT.value
        ),
        "explanation_layers": {
            "model_attribution": [
                {**e.to_dict(), "explainer_validation": statements.get(e.source_ref)}
                for e in case.evidence if e.source_class == SourceClass.EXPLANATION_FACT.value
            ],
            "scientific_evidence": entries(e for e in case.evidence if e.independent),
            "agent_decisions": [a.to_dict() for a in case.actions],
        },
        "hypotheses": hypotheses,
        "unlinked_evidence": entries(e for e in case.evidence if e.id not in linked),
        "open_uncertainties": [u.to_dict() for u in case.open_uncertainties],
        "resolved_uncertainties": [u.to_dict() for u in case.uncertainties if u.status != "open"],
        "next_tests": [t.to_dict() for t in case.next_tests],
        "conclusion": {
            **case.conclusion.to_dict(),
            "can_say": [
                {**line.to_dict(),
                 "sources": [ledger[i].source_ref for i in line.evidence_ids if i in ledger]}
                for line in case.conclusion.can_say
            ],
        },
        "coverage": case.coverage,
        "stop_reason": stop_reason,
    }


def checkpoint_summary(case: ScientificCaseV1, *, limit: int = 6) -> str:
    """What a new turn of this case needs to know, with ids to read the rest."""
    lines = [f"Open scientific case {case.id} (revision {case.revision})."]
    if case.question:
        lines.append(f"Decision question: {case.question[:400]}")
    for item in case.context[:limit]:
        lines.append(f"- context {item.id} ({item.actor}): {item.key} = {item.value[:160]}")
    for h in case.hypotheses[:limit]:
        n_for = len(case.evidence_for(h.id, Stance.SUPPORTS.value))
        n_against = len(case.evidence_for(h.id, Stance.CONTRADICTS.value))
        lines.append(f"- {h.id} [{h.status}] {h.statement[:200]} "
                     f"(for: {n_for}, against: {n_against})")
    open_items = case.open_uncertainties[:limit]
    if open_items:
        lines.append("Open uncertainties: " + "; ".join(
            f"{u.id} {u.kind} ({u.severity})" for u in open_items))
    if case.next_tests:
        lines.append("Proposed tests: " + "; ".join(t.test[:120] for t in case.next_tests[:3]))
    cov = case.coverage
    lines.append(
        f"Coverage: {cov['with_any_source']}/{cov['hypotheses']} hypotheses have a source; "
        f"{cov['with_independent_direct_evidence']} have direct independent evidence; "
        f"{cov['with_counterevidence_considered']} had counter-evidence considered."
    )
    lines.append("Read the full case with get_scientific_case; change it with update_scientific_case.")
    return "\n".join(lines)

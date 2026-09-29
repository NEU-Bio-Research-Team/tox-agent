"""The closed set of case operations, and ``apply`` / ``replay`` over them."""
from __future__ import annotations

from dataclasses import replace
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

from .model import (
    MAX_ACTIONS,
    MAX_CONTEXT,
    MAX_EVIDENCE,
    MAX_HYPOTHESES,
    MAX_NEXT_TESTS,
    MAX_TEXT,
    MAX_UNCERTAINTIES,
    REF_KIND,
    Action,
    ActionDecision,
    Actor,
    CaseStatus,
    CaseUpdate,
    Conclusion,
    ConclusionLine,
    ContextItem,
    DataScope,
    Directness,
    EvidenceEntry,
    Hypothesis,
    HypothesisKind,
    HypothesisStatus,
    InvalidCaseUpdate,
    NextTest,
    RunRecord,
    ScientificCaseV1,
    Severity,
    SourceClass,
    Stance,
    Uncertainty,
    UncertaintyKind,
    _values,
)


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


def _open(case: ScientificCaseV1 | None, u: CaseUpdate) -> ScientificCaseV1:
    if case is not None:
        raise InvalidCaseUpdate("the case is already open")
    p = u.payload
    return ScientificCaseV1(
        id=str(p["case_id"]), session_id=str(p["session_id"]), subject_key=str(p["subject_key"]),
        question=_text(p, "question", required=False, limit=2000),
        decision_context=_text(p, "decision_context", required=False),
        subject_refs=tuple(dict.fromkeys(p.get("subject_refs") or ())),
        requester=_text(p, "requester", required=False, limit=200),
        data_scope=DataScope(**p["data_scope"]) if p.get("data_scope") else DataScope(),
        created_at=u.at,
    )


def _set_question(case: ScientificCaseV1, u: CaseUpdate) -> ScientificCaseV1:
    return replace(
        case, question=_text(u.payload, "question", limit=2000),
        decision_context=_text(u.payload, "decision_context", required=False)
        or case.decision_context,
    )


def _set_scope(case: ScientificCaseV1, u: CaseUpdate) -> ScientificCaseV1:
    allowed = u.payload.get("external_search")
    if not isinstance(allowed, bool):
        raise InvalidCaseUpdate("external_search must be true or false")
    reason = _text(u.payload, "reason", required=not allowed, limit=500)
    scope = DataScope(external_search=allowed, reason=reason, run_id=u.run_id)
    if scope == case.data_scope:
        return case
    return replace(case, data_scope=scope)


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
    current = next((r for r in case.runs if r.run_id == run_id), None)
    if current is not None and (
        current.stop_reason, current.answer_id, dict(current.usage)
    ) == (u.payload.get("stop_reason"), u.payload.get("answer_id"),
          dict(u.payload.get("usage") or {})):
        return case  # the same ending recorded twice is one ending
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
    #: Only the researcher widens or narrows what the case may reach.
    "set_scope": (_set_scope, {Actor.USER}),
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

"""Where ScientificCaseV1 transitions meet the database (ADR 0012).

The domain module owns the transitions; this module owns reading a case,
applying updates with a revision check, and writing the snapshot together
with the event rows that produced it.

Two write disciplines, on purpose:

* **Requested writes** (a model's ``update_scientific_case``, a user adding
  context) go through ``apply_updates`` and fail loudly: a refused update is
  the caller's to correct, so ``InvalidCaseUpdate`` propagates with a message
  written for whoever sent it.
* **Bookkeeping writes** (attaching a run, recording an accepted answer's
  relations, finishing a run, storing the dossier) go through ``advance`` and
  never break the run they observe, exactly like ``decision_state_service``:
  a lost race is retried, anything else is logged.
"""
from __future__ import annotations

import logging
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Mapping, Sequence

from ..domain import scientific_case as sc
from ..domain.errors import Conflict
from ..domain.ids import SCIENTIFIC_CASE, new_id

log = logging.getLogger("toxagent.scientific_case")

UpdatesFor = Callable[[sc.ScientificCaseV1], Sequence[sc.CaseUpdate]]


def _now() -> datetime:
    return datetime.now(timezone.utc)


def update(op: str, *, actor: str, run_id: str | None = None, at: datetime | None = None,
           **payload: Any) -> sc.CaseUpdate:
    return sc.CaseUpdate(op=op, payload=payload, actor=actor, run_id=run_id,
                         at=(at or _now()).isoformat())


def fold(case: sc.ScientificCaseV1 | None,
         updates: Iterable[sc.CaseUpdate]) -> tuple[sc.ScientificCaseV1 | None, list[sc.CaseUpdate]]:
    """Apply ``updates`` in order; return the case and the updates that changed it,
    each stamped with the revision it produced. No-ops are dropped, so an
    idempotent repeat writes nothing."""
    applied: list[sc.CaseUpdate] = []
    for item in updates:
        after = sc.apply(case, item)
        if after is case:
            continue
        applied.append(replace(item, revision=after.revision))
        case = after
    return case, applied


async def apply_updates(
    uow, *, case_id: str, session_id: str, updates: Sequence[sc.CaseUpdate],
) -> sc.ScientificCaseV1:
    """Apply requested updates inside the caller's unit of work.

    Raises ``InvalidCaseUpdate`` for a refused update (nothing is written: the
    updates are all-or-nothing) and ``Conflict`` on a concurrent write.
    """
    case = await uow.scientific_cases.get(case_id, session_id=session_id)
    if case is None:
        raise sc.InvalidCaseUpdate("this session has no such scientific case")
    updated, applied = fold(case, updates)
    await uow.scientific_cases.append(updated, applied, expected_revision=case.revision, now=_now())
    return updated


async def open_or_continue(
    uow, *, session_id: str, analysis_id: str | None, run_id: str, goal: str,
    subject_refs: Iterable[str], extra_updates: Sequence[sc.CaseUpdate] = (),
) -> sc.ScientificCaseV1:
    """The session's open case for this subject, with this run attached.

    A subject switch (another analysis) opens another case: one compound's
    evidence never lands in another's ledger. The first run's goal becomes the
    case's question; later runs are recorded against it without replacing it —
    changing the question is an explicit ``set_question``.
    """
    key = sc.subject_key(analysis_id)
    case = await uow.scientific_cases.open_for_subject(session_id, key)
    now = _now()
    updates: list[sc.CaseUpdate] = []
    if case is None:
        updates.append(update(
            "open", actor=sc.Actor.SERVER.value, run_id=run_id, at=now,
            case_id=new_id(SCIENTIFIC_CASE), session_id=session_id, subject_key=key,
            question=(goal or "").strip()[:2000], subject_refs=list(dict.fromkeys(subject_refs)),
        ))
    updates.append(update("attach_run", actor=sc.Actor.SERVER.value, run_id=run_id, at=now,
                          goal=(goal or "").strip()[:2000]))
    updates.extend(extra_updates)
    updated, applied = fold(case, updates)
    await uow.scientific_cases.append(
        updated, applied, expected_revision=case.revision if case else None, now=now,
    )
    return updated


async def case_for_run(uow, *, session_id: str, run_id: str) -> sc.ScientificCaseV1 | None:
    case_id = await uow.scientific_cases.case_id_for_run(run_id, session_id=session_id)
    if case_id is None:
        return None
    return await uow.scientific_cases.get(case_id, session_id=session_id)


async def advance_in(uow, *, session_id: str, run_id: str, updates_for: UpdatesFor) -> sc.ScientificCaseV1 | None:
    """Server bookkeeping inside the caller's unit of work.

    ``updates_for`` sees the current case and returns the updates to apply;
    refused ones are logged and skipped individually rather than failing the
    batch — a bookkeeping write records what it can.
    """
    case = await case_for_run(uow, session_id=session_id, run_id=run_id)
    if case is None:
        return None
    applied: list[sc.CaseUpdate] = []
    current = case
    for item in updates_for(case):
        try:
            after = sc.apply(current, item)
        except sc.InvalidCaseUpdate as exc:
            log.warning("case bookkeeping update refused: %s", exc,
                        extra={"run_id": run_id, "op": item.op})
            continue
        if after is current:
            continue
        applied.append(replace(item, revision=after.revision))
        current = after
    await uow.scientific_cases.append(current, applied, expected_revision=case.revision, now=_now())
    return current


async def advance(database, *, session_id: str, run_id: str, updates_for: UpdatesFor,
                  attempts: int = 4) -> sc.ScientificCaseV1 | None:
    """``advance_in`` in its own unit of work, retrying a lost race; never raises."""
    for attempt in range(attempts):
        try:
            async with database.unit_of_work() as uow:
                case = await advance_in(uow, session_id=session_id, run_id=run_id,
                                        updates_for=updates_for)
                await uow.commit()
            return case
        except Conflict:
            if attempt == attempts - 1:
                log.warning("case update lost %d races; skipped", attempts,
                            extra={"run_id": run_id})
        except Exception:  # noqa: BLE001 - bookkeeping must not fail the run
            log.exception("case update failed", extra={"run_id": run_id})
            return None
    return None


def answer_relation_updates(relations: Iterable[Any], *, run_id: str) -> UpdatesFor:
    """The accepted answer's relations as ledger updates, for ``advance_in``."""
    payload = [
        {
            key: getattr(r, key, None) if not isinstance(r, Mapping) else r.get(key)
            for key in ("proposition", "source_class", "source_id", "relation", "directness",
                        "endpoint", "species", "dose", "use_context")
        }
        for r in relations
    ]
    at = _now().isoformat()
    return lambda case: sc.updates_from_answer(case, payload, run_id=run_id, at=at)


def analysis_uncertainties(snapshot, *, run_id: str) -> list[sc.CaseUpdate]:
    """What the server already knows is uncertain about the case's subject.

    Deterministic, so recording them on every turn is idempotent (the domain
    drops an open uncertainty of the same kind and description): an
    applicability status other than ``ok`` and every requested endpoint this
    deployment does not serve. Nothing here is a judgement a model made.
    """
    if snapshot is None:
        return []
    at = _now().isoformat()
    updates: list[sc.CaseUpdate] = []
    applicability = (snapshot.predictor_response.get("applicability") or {})
    status = applicability.get("status")
    if status and status != "ok":
        reasons = ", ".join(str(r) for r in applicability.get("reasons") or ()) or "no reason given"
        updates.append(sc.CaseUpdate(
            op="record_uncertainty", actor=sc.Actor.SERVER.value, run_id=run_id, at=at,
            payload={
                "kind": sc.UncertaintyKind.APPLICABILITY_DOMAIN.value,
                "severity": sc.Severity.HIGH.value if status == "out_of_domain" else sc.Severity.MEDIUM.value,
                "description": (
                    f"The predictor's rule-based applicability check says {status} ({reasons}); "
                    "its scores for this structure are less reliable."
                ),
            },
        ))
    for endpoint in snapshot.unavailable_endpoints:
        updates.append(sc.CaseUpdate(
            op="record_uncertainty", actor=sc.Actor.SERVER.value, run_id=run_id, at=at,
            payload={
                "kind": sc.UncertaintyKind.MISSING_ENDPOINT.value,
                "severity": sc.Severity.MEDIUM.value,
                "description": f"The {endpoint} endpoint was requested but is not served by this deployment.",
            },
        ))
    return updates


_UNSET = object()


async def finish_run(database, *, session_id: str, run_id: str,
                     stop_reason: Any = _UNSET) -> dict[str, Any] | None:
    """Record how a run ended in its case, then compile and store its dossier.

    ``stop_reason`` defaults to the run's final DecisionSupportStateV1; the
    gateway passes it explicitly when it finishes the case just before the run
    completes. One dossier per run: a second call records nothing new. Never
    raises — the run has already ended.
    """
    from ..domain import explainer_validation

    try:
        async with database.unit_of_work() as uow:
            state = await uow.decision_states.get(run_id)
            answer = await uow.answers.get_for_run(run_id)
        if stop_reason is _UNSET:
            stop_reason = state.stop_reason if state is not None else None
        usage = dict(state.usage) if state is not None else {}
        answer_id = answer.id if answer is not None else None
        finished = update("finish_run", actor=sc.Actor.SERVER.value, run_id=run_id,
                          stop_reason=stop_reason, answer_id=answer_id, usage=usage)
        case = await advance(database, session_id=session_id, run_id=run_id,
                             updates_for=lambda _case: [finished])
        if case is None:
            return None
        async with database.unit_of_work() as uow:
            statements: dict[str, dict[str, Any]] = {}
            for entry in case.evidence:
                if entry.source_class != sc.SourceClass.EXPLANATION_FACT.value:
                    continue
                if entry.source_ref in statements:
                    continue
                observation = await uow.observations.get(
                    sc.parse_ref(entry.source_ref)[1], session_id=session_id
                )
                if observation is None:
                    continue
                statements[entry.source_ref] = explainer_validation.for_observation(observation)
            dossier = sc.compile_dossier(
                case, run_id=run_id, stop_reason=stop_reason, answer_id=answer_id,
                explainer_statements=statements,
            )
            if await uow.scientific_cases.get_dossier(run_id, session_id=session_id) is None:
                await uow.scientific_cases.put_dossier(dossier, now=_now())
                await uow.commit()
        return dossier
    except Exception:  # noqa: BLE001 - the run has ended; bookkeeping must not raise
        log.exception("could not finish the scientific case", extra={"run_id": run_id})
        return None

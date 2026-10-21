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

from ...domain import scientific_case as sc
from ...domain.errors import Conflict
from ...domain.ids import SCIENTIFIC_CASE, new_id

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


async def apply_model_updates(
    uow, *, case_id: str, session_id: str, updates: Sequence[tuple[int, sc.CaseUpdate]],
) -> tuple[sc.ScientificCaseV1, list[tuple[int, str, str]]]:
    """The model's batch, applied operation by operation.

    Each refused operation is returned as ``(index, op, reason)`` and the rest
    are written. All-or-nothing cost a whole model round per mistake: a model
    rewrites its entire batch — every operation, every field — to fix one id,
    about a minute each, and three such rounds ended a live turn at its
    deadline with nothing recorded (2026-09-26). A refused operation is still
    refused loudly; it just no longer takes the valid ones with it. Raises
    ``InvalidCaseUpdate`` when nothing could be applied.
    """
    case = await uow.scientific_cases.get(case_id, session_id=session_id)
    if case is None:
        raise sc.InvalidCaseUpdate("this session has no such scientific case")
    current, applied, refused = case, [], []
    for index, item in updates:
        try:
            after = sc.apply(current, item)
        except sc.InvalidCaseUpdate as exc:
            refused.append((index, item.op, str(exc)))
            continue
        if after is not current:
            applied.append(replace(item, revision=after.revision))
            current = after
    if not applied:
        if refused:
            raise sc.InvalidCaseUpdate("; ".join(
                f"operations[{index}] ({op}): {reason}" for index, op, reason in refused))
        return case, refused
    await uow.scientific_cases.append(current, applied, expected_revision=case.revision, now=_now())
    return current, refused


def case_key(*, analysis_id: str | None = None, binding=None) -> str:
    """The case key of a subject: the bound subject's when there is one (#44),
    else the legacy key of the analysis (or ``none``)."""
    if binding is not None and not binding.pending_snapshot:
        return binding.subject_key
    return sc.subject_key(analysis_id)


async def open_or_continue(
    uow, *, session_id: str, analysis_id: str | None, run_id: str, goal: str,
    subject_refs: Iterable[str], extra_updates: Sequence[sc.CaseUpdate] = (),
    requester: str = "", binding=None,
) -> sc.ScientificCaseV1:
    """The session's open case for this subject, with this run attached.

    A subject switch (another analysis, another target) opens another case:
    one compound's evidence never lands in another's ledger. The first run's
    goal becomes the case's question; later runs are recorded against it
    without replacing it -- changing the question is an explicit
    ``set_question``.

    With a bound subject (#44) the case is keyed by the subject's identity and
    stores it; continuing a case whose stored identity disagrees with the
    binding on an axis both state (a different UniProt id for the same target
    id, say) is a ``SubjectMismatch``, never a silent merge.
    """
    from ...domain.errors import SubjectMismatch

    key = case_key(analysis_id=analysis_id, binding=binding)
    case = await uow.scientific_cases.open_for_subject(session_id, key)
    if case is not None and binding is not None and case.subject:
        differing = binding.mismatches(case.subject.get("identity") or {})
        if differing:
            raise SubjectMismatch(
                "the run's subject and the open case's subject differ on "
                + ", ".join(differing),
                case_id=case.id, axes=list(differing),
            )
    now = _now()
    updates: list[sc.CaseUpdate] = []
    if case is None:
        # A researcher's restriction outlives the case it was set on: closing
        # a case must not quietly re-open the compound to external search.
        previous = await latest_case_for_key(uow, session_id=session_id, key=key)
        inherited = (
            previous.data_scope.to_dict()
            if previous is not None and not previous.data_scope.external_search else None
        )
        updates.append(update(
            "open", actor=sc.Actor.SERVER.value, run_id=run_id, at=now,
            case_id=new_id(SCIENTIFIC_CASE), session_id=session_id, subject_key=key,
            question=(goal or "").strip()[:2000], subject_refs=list(dict.fromkeys(subject_refs)),
            requester=requester, **({"data_scope": inherited} if inherited else {}),
            **({"subject": binding.case_subject()} if binding is not None else {}),
        ))
    attach: dict[str, Any] = {"goal": (goal or "").strip()[:2000]}
    if binding is not None:
        attach["binding"] = {
            "binding_sha256": binding.binding_sha256, "resolved_by": binding.resolved_by,
            "artifact_refs": [ref.to_dict() for ref in binding.artifact_refs],
        }
    updates.append(update("attach_run", actor=sc.Actor.SERVER.value, run_id=run_id, at=now,
                          **attach))
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
    keys = ("proposition", "source_class", "source_id", "relation", "directness",
            *sc.RELATION_SCOPE_AXES)
    payload = [
        {
            key: getattr(r, key, None) if not isinstance(r, Mapping) else r.get(key)
            for key in keys
        }
        for r in relations
    ]
    at = _now().isoformat()
    return lambda case: sc.updates_from_answer(case, payload, run_id=run_id, at=at)


def answer_ledger_updates(relations: Iterable[Any], cited: Sequence[Mapping[str, Any]], *,
                          run_id: str) -> UpdatesFor:
    """An accepted answer's contribution to the ledger, for ``advance_in``.

    Its relations first (they carry a stance), then whatever else it cited
    (W9-04), so a source the answer both related and cited is one entry.
    """
    from_relations = answer_relation_updates(relations, run_id=run_id)
    at = _now().isoformat()

    def updates_for(case: sc.ScientificCaseV1) -> list[sc.CaseUpdate]:
        # #44: an activity artifact of another subject (the same ligand on
        # another target, read by its own screening id) is not this case's
        # evidence, whatever the answer did with it.
        foreign = {
            f"observation:{item['source_id']}" for item in cited
            if item.get("subject_key") and case.subject
            and item["subject_key"] != case.subject_key
        }
        if foreign:
            log.info("left %d artifact(s) of another subject out of case %s",
                     len(foreign), case.id, extra={"run_id": run_id})
        related = [u for u in from_relations(case)
                   if str(u.payload.get("source_ref")) not in foreign]
        related_refs = [str(u.payload.get("source_ref")) for u in related]
        return related + sc.updates_from_citations(
            case, [c for c in cited if f"observation:{c['source_id']}" not in foreign],
            run_id=run_id, at=at, skip_refs=related_refs,
        )

    return updates_for


def _activity_entry(observation) -> dict[str, Any]:
    """What the ledger records about a replayed or served activity artifact:
    its source class (a model output, a rule's output, a workflow fact -- none
    of them independent evidence), the scope its own subject block states, and
    where it came from. A pIC50 is scoped by target and unit, never by a
    toxicity endpoint, so it cannot be set against a probability."""
    from ...domain.observation import ObservationKind
    from ...domain.subject import InvalidSubject, activity_subject_key

    payload = observation.canonical_payload or {}
    subject = payload.get("subject") or {}
    target = subject.get("target") or {}
    source_class = {
        ObservationKind.ACTIVITY_ESTIMATE: sc.SourceClass.PREDICTOR_FACT.value,
        ObservationKind.PRIORITISATION_DECISION: sc.SourceClass.POLICY_FACT.value,
        ObservationKind.SCREENING_STATE: sc.SourceClass.WORKFLOW_FACT.value,
    }[observation.kind]
    scope = {
        "ligand_id": subject.get("canonical_smiles"),
        "target_id": target.get("target_id"),
        "target_variant": target.get("variant"),
        "target_sequence_sha256": target.get("sequence_sha256"),
        "experimental_context": subject.get("assay_context"),
    }
    if observation.kind is ObservationKind.ACTIVITY_ESTIMATE:
        scope["unit"] = "pIC50"
    if observation.kind is ObservationKind.PRIORITISATION_DECISION:
        policy = payload.get("policy") or {}
        scope["target_id"] = scope["target_id"] or policy.get("target_id")
        scope["experimental_context"] = scope["experimental_context"] or policy.get("assay_context")
    replayed = (observation.provenance or {}).get("replayed_from") or {}
    provenance = {
        "observation_kind": observation.kind.value,
        "screening_id": payload.get("screening_id"),
        "record_sha256": replayed.get("record_sha256"),
        "content_sha256": observation.content_sha256,
    }
    entry: dict[str, Any] = {
        "source_class": source_class,
        "scope": {k: v for k, v in scope.items() if v},
        "provenance": {k: v for k, v in provenance.items() if v},
    }
    try:
        entry["subject_key"] = activity_subject_key(subject) if subject else None
    except InvalidSubject:
        entry["subject_key"] = None
    return entry


async def cited_sources(uow, *, session_id: str, answer) -> list[dict[str, Any]]:
    """The sources an accepted answer's claims cite, classed for the ledger."""
    from ...domain.evidence import SourceType
    from ...domain.observation import ObservationKind

    activity_kinds = {
        ObservationKind.ACTIVITY_ESTIMATE, ObservationKind.PRIORITISATION_DECISION,
        ObservationKind.SCREENING_STATE,
    }
    cited: list[dict[str, Any]] = []
    for claim in answer.claims:
        if claim.observation_id:
            observation = await uow.observations.get(claim.observation_id, session_id=session_id)
            if observation is not None and observation.kind in activity_kinds:
                # #44: the three activity artifacts reach the ledger as what
                # they are, with the subject and scope they state.
                cited.append({"source_id": claim.observation_id, "claim": claim.text,
                              "locator": claim.field_path, **_activity_entry(observation)})
            else:
                source_class = {
                    ObservationKind.PREDICTION: sc.SourceClass.PREDICTOR_FACT.value,
                    ObservationKind.ATTRIBUTION: sc.SourceClass.EXPLANATION_FACT.value,
                }.get(observation.kind) if observation is not None else None
                if source_class:
                    cited.append({"source_class": source_class,
                                  "source_id": claim.observation_id,
                                  "claim": claim.text, "locator": claim.field_path})
        for evidence_id in claim.citation_ids:
            record = await uow.evidence.get(evidence_id, session_id=session_id)
            if record is None:
                continue
            source_class = (
                sc.SourceClass.EXTERNAL_REGULATORY.value
                if record.source_type is SourceType.REGULATORY
                else sc.SourceClass.EXTERNAL_EXPERIMENTAL.value
            )
            # #44: the passages the claim quoted, not only the record: an
            # entry without its locator cannot be checked against the text.
            spans = [locator for span_id, locator in claim.evidence_spans
                     if span_id == evidence_id]
            cited.append({"source_class": source_class, "source_id": evidence_id,
                          "claim": claim.text, "locator": spans[0] if spans else None,
                          **({"spans": spans} if spans else {})})
    return cited


async def latest_case_for_key(uow, *, session_id: str, key: str) -> sc.ScientificCaseV1 | None:
    """The session's most recent case under this subject key, open or closed."""
    for case in await uow.scientific_cases.list_for_session(session_id, limit=200):
        if case.subject_key == key:
            return case
    return None


async def latest_case_for_subject(uow, *, session_id: str,
                                  analysis_id: str | None,
                                  binding=None) -> sc.ScientificCaseV1 | None:
    """The session's most recent case about this subject, open or closed."""
    return await latest_case_for_key(
        uow, session_id=session_id, key=case_key(analysis_id=analysis_id, binding=binding),
    )


async def external_search_refusal(uow, *, session_id: str, analysis_id: str | None,
                                  binding=None) -> str | None:
    """Why the researcher forbade external search about this subject, or ``None``.

    Keyed on the subject, not the run: the evidence search tool serves chat
    turns and both report paths (the orchestrator's server-side search runs the
    same tool), and a confidential structure is confidential in all of them.
    """
    case = await latest_case_for_subject(uow, session_id=session_id, analysis_id=analysis_id,
                                         binding=binding)
    if case is None or case.data_scope.external_search:
        return None
    return case.data_scope.reason or "the researcher restricted this compound to internal data"


async def dossier_for_analysis(uow, *, session_id: str,
                               analysis_id: str | None) -> dict[str, Any] | None:
    """The latest DecisionDossierV1 of the subject's most recent case."""
    case = await latest_case_for_subject(uow, session_id=session_id, analysis_id=analysis_id)
    if case is None:
        return None
    return await uow.scientific_cases.latest_dossier(case.id, session_id=session_id)


def report_dossier_view(dossier: Mapping[str, Any]) -> dict[str, Any]:
    """The investigation record as a report reads it (RETHINK §4.4.3, §4.9).

    Compact and typed: each hypothesis with its status and the source refs for
    and against it, the conditional conclusion with the sources of every line,
    what is still unknown, and the proposed tests. The record is not itself a
    source: a report cites the observations and evidence records it names.
    """
    def refs(entries: Iterable[Mapping[str, Any]]) -> list[str]:
        return [str(e.get("source_ref")) for e in entries]

    conclusion = dossier.get("conclusion") or {}
    proposed = conclusion.get("status") == "proposed"
    if proposed:
        # #47: a report states what the case trusts, never a conclusion whose
        # run's answer was not published.
        conclusion = dossier.get("released_conclusion") or {}
    return {
        "case_id": dossier.get("case_id"), "run_id": dossier.get("run_id"),
        "case_revision": dossier.get("case_revision"),
        "question": dossier.get("question"),
        "data_scope": dossier.get("data_scope"),
        "hypotheses": [
            {"id": h.get("id"), "statement": h.get("statement"), "status": h.get("status"),
             "refutation_condition": h.get("refutation_condition"),
             "evidence_for": refs(h.get("evidence_for") or ()),
             "evidence_against": refs(h.get("evidence_against") or ())}
            for h in dossier.get("hypotheses") or ()
        ],
        "conclusion": {
            "can_say": [{"text": line.get("text"), "sources": list(line.get("sources") or ())}
                        for line in conclusion.get("can_say") or ()],
            "cannot_say": list(conclusion.get("cannot_say") or ()),
            "what_would_change": list(conclusion.get("what_would_change") or ()),
            "unpublished_proposal_left_out": proposed,
        },
        "open_uncertainties": [
            {"kind": u.get("kind"), "severity": u.get("severity"), "description": u.get("description")}
            for u in dossier.get("open_uncertainties") or ()
        ],
        "next_tests": [
            {"test": t.get("test"), "discriminates": list(t.get("discriminates") or ()),
             "expected_readouts": list(t.get("expected_readouts") or ())}
            for t in dossier.get("next_tests") or ()
        ],
        "coverage": dossier.get("coverage"),
        "how_to_use": (
            "The researcher's investigation of this compound so far. Report what it "
            "concluded and what is still open; cite the observations and evidence "
            "records its sources name, never the record itself."
        ),
    }


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
    from ...domain import explainer_validation

    try:
        async with database.unit_of_work() as uow:
            state = await uow.decision_states.get(run_id)
            answer = await uow.answers.get_for_run(run_id)
            result = await uow.run_results.get(run_id)
        if stop_reason is _UNSET:
            stop_reason = state.stop_reason if state is not None else None
        usage = dict(state.usage) if state is not None else {}
        # #47: under a declared policy the run's answer is its public one --
        # a withheld candidate is not what the case records as the answer.
        answer_id = (
            result.public_answer_id if result is not None
            else answer.id if answer is not None else None
        )
        finished = update("finish_run", actor=sc.Actor.SERVER.value, run_id=run_id,
                          stop_reason=stop_reason, answer_id=answer_id, usage=usage,
                          **({"disposition": result.disposition.value} if result is not None
                             else {}))
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

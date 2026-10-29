"""Read models of a case: the decision dossier and the checkpoint summary."""
from __future__ import annotations

from typing import Any, Iterable, Mapping

from .dependencies import needs_reassessment
from .model import (
    CONCLUSION_PROPOSED,
    CONCLUSION_RELEASED,
    DOSSIER_SCHEMA_VERSION,
    EvidenceEntry,
    ScientificCaseV1,
    SourceClass,
    Stance,
)


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
        "subject_key": case.subject_key,
        "subject": dict(case.subject),
        "subject_refs": list(case.subject_refs),
        "requester": case.requester,
        "data_scope": case.data_scope.to_dict(),
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
            "status": case.conclusion.status,
            "can_say": [
                {**line.to_dict(),
                 "sources": [ledger[i].source_ref for i in line.evidence_ids if i in ledger]}
                for line in case.conclusion.can_say
            ],
        },
        # #47: what the case last trusted, while the conclusion above is only
        # a proposal of a run whose answer was not published.
        "released_conclusion": (
            case.released_conclusion.to_dict() if case.released_conclusion is not None else None
        ),
        # #60: what the case derived, from what, and what a changed premise
        # has left waiting. §16 item 5 -- "thay đổi so với assessment trước" --
        # is read from here: a claim marked needs_reassessment is exactly the
        # part of the last assessment a new premise moved.
        "derived_claims": [c.to_dict() for c in case.derived_claims],
        "claims_needing_reassessment": [
            c.id for c in needs_reassessment(case.derived_claims)
        ],
        "coverage": case.coverage,
        "stop_reason": stop_reason,
    }


def checkpoint_summary(case: ScientificCaseV1, *, limit: int = 6, evidence_limit: int = 8) -> str:
    """What a new turn of this case needs to know, with the ids it writes against.

    Enough that a turn can write its one update without reading the case first
    (W9-01): every hypothesis and open uncertainty with its id, the most recent
    ledger entries with theirs, and the conclusion so far.
    """
    lines = [f"Open scientific case {case.id} (revision {case.revision})."]
    identity = (case.subject or {}).get("identity") or {}
    if identity.get("target"):
        target = identity["target"]
        variant = f" ({target['variant']})" if target.get("variant") else ""
        lines.append(
            f"Subject: {identity.get('canonical_smiles')} on {target.get('target_id')}{variant} "
            f"in {identity.get('assay_context')}. Evidence about another target or context "
            "is not this case's evidence."
        )
    if case.question:
        lines.append(f"Decision question: {case.question[:400]}")
    if not case.data_scope.external_search:
        lines.append(
            "Data scope: the researcher does not allow external literature search for this "
            f"case ({case.data_scope.reason[:200]}); work from the predictor, the session's "
            "existing records and what the researcher supplies."
        )
    for item in case.context[:limit]:
        lines.append(f"- context {item.id} ({item.actor}): {item.describe()}")
    unpublished = {r.run_id for r in case.runs if r.disposition not in (None, "publish")}
    for h in case.hypotheses[:limit]:
        n_for = len(case.evidence_for(h.id, Stance.SUPPORTS.value))
        n_against = len(case.evidence_for(h.id, Stance.CONTRADICTS.value))
        # #47: a status a withheld run set is a proposal, not the case's.
        status = (f"{h.status}, proposed by a run whose answer was not published"
                  if h.status_run_id in unpublished else h.status)
        lines.append(f"- {h.id} [{status}] {h.statement[:200]} "
                     f"(for: {n_for}, against: {n_against})")
    open_items = case.open_uncertainties[:limit]
    if open_items:
        lines.append("Open uncertainties: " + "; ".join(
            f"{u.id} {u.kind} ({u.severity})" for u in open_items))
    recent = case.evidence[-evidence_limit:]
    if recent:
        hidden = len(case.evidence) - len(recent)
        lines.append("Ledger" + (f" (latest {len(recent)} of {len(case.evidence)})" if hidden else "") + ":")
        for e in recent:
            bears = f" on {','.join(e.hypothesis_ids)}" if e.hypothesis_ids else ""
            lines.append(f"- {e.id} {e.stance}{bears} [{e.source_class} {e.source_ref}] {e.claim[:140]}")
    if case.next_tests:
        lines.append("Proposed tests: " + "; ".join(t.test[:120] for t in case.next_tests[:3]))
    # #60: a premise moved under these, so they are the work this turn starts
    # from. Said before the conclusion, because a conclusion that rests on one
    # of them is not what the case can say any more.
    stale = needs_reassessment(case.derived_claims)[:limit]
    if stale:
        lines.append(
            "Derived claims a changed premise left waiting (recompute or reassess before "
            "relying on them): " + "; ".join(f"{c.id} {c.statement[:120]} — {c.note[:160]}"
                                             for c in stale))
    def said(conclusion) -> str:
        return "; ".join(
            [f"can say: {line.text[:120]} ({','.join(line.evidence_ids)})"
             for line in conclusion.can_say[:3]]
            + [f"cannot say: {text[:120]}" for text in conclusion.cannot_say[:2]]
        )

    trusted = (
        case.conclusion if case.conclusion.status == CONCLUSION_RELEASED
        else case.released_conclusion
    )
    if trusted is not None and (trusted.can_say or trusted.cannot_say):
        lines.append("Conclusion so far: " + said(trusted))
    if case.conclusion.status == CONCLUSION_PROPOSED and (
        case.conclusion.can_say or case.conclusion.cannot_say
    ):
        lines.append("Proposed by an earlier run, not published -- not the case's conclusion: "
                     + said(case.conclusion))
    cov = case.coverage
    lines.append(
        f"Coverage: {cov['with_any_source']}/{cov['hypotheses']} hypotheses have a source; "
        f"{cov['with_independent_direct_evidence']} have direct independent evidence; "
        f"{cov['with_counterevidence_considered']} had counter-evidence considered."
    )
    lines.append("Change the case with update_scientific_case; get_scientific_case shows all of it.")
    return "\n".join(lines)

"""Read models of a case: the decision dossier and the checkpoint summary."""
from __future__ import annotations

from typing import Any, Iterable, Mapping

from .model import DOSSIER_SCHEMA_VERSION, EvidenceEntry, ScientificCaseV1, SourceClass, Stance


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
            "can_say": [
                {**line.to_dict(),
                 "sources": [ledger[i].source_ref for i in line.evidence_ids if i in ledger]}
                for line in case.conclusion.can_say
            ],
        },
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
    if case.question:
        lines.append(f"Decision question: {case.question[:400]}")
    if not case.data_scope.external_search:
        lines.append(
            "Data scope: the researcher does not allow external literature search for this "
            f"case ({case.data_scope.reason[:200]}); work from the predictor, the session's "
            "existing records and what the researcher supplies."
        )
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
    recent = case.evidence[-evidence_limit:]
    if recent:
        hidden = len(case.evidence) - len(recent)
        lines.append("Ledger" + (f" (latest {len(recent)} of {len(case.evidence)})" if hidden else "") + ":")
        for e in recent:
            bears = f" on {','.join(e.hypothesis_ids)}" if e.hypothesis_ids else ""
            lines.append(f"- {e.id} {e.stance}{bears} [{e.source_class} {e.source_ref}] {e.claim[:140]}")
    if case.next_tests:
        lines.append("Proposed tests: " + "; ".join(t.test[:120] for t in case.next_tests[:3]))
    if case.conclusion.can_say or case.conclusion.cannot_say:
        lines.append("Conclusion so far: " + "; ".join(
            [f"can say: {line.text[:120]} ({','.join(line.evidence_ids)})"
             for line in case.conclusion.can_say[:3]]
            + [f"cannot say: {text[:120]}" for text in case.conclusion.cannot_say[:2]]
        ))
    cov = case.coverage
    lines.append(
        f"Coverage: {cov['with_any_source']}/{cov['hypotheses']} hypotheses have a source; "
        f"{cov['with_independent_direct_evidence']} have direct independent evidence; "
        f"{cov['with_counterevidence_considered']} had counter-evidence considered."
    )
    lines.append("Change the case with update_scientific_case; get_scientific_case shows all of it.")
    return "\n".join(lines)

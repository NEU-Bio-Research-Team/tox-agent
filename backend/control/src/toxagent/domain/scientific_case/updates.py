"""Case updates derived from an accepted answer and from its citations."""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from .model import (
    MAX_TEXT,
    REF_KIND,
    Actor,
    CaseUpdate,
    Directness,
    ScientificCaseV1,
    SourceClass,
    Stance,
)

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


def updates_from_citations(
    case: ScientificCaseV1, cited: Sequence[Mapping[str, Any]], *, run_id: str, at: str,
    skip_refs: Iterable[str] = (),
) -> list[CaseUpdate]:
    """What an accepted answer cited, as server-recorded ledger entries.

    Works for every answer schema (W9-04): grounded-answer v1 carries no
    relations, so before this a case kept with v1 had a ledger only where the
    model had written entries itself. A citation says the answer relied on a
    source, not which way it bears on a hypothesis, so each entry is
    ``contextual`` and unlinked; the model's own ``record_evidence`` and a v2
    relation are where a stance comes from. A source the ledger already holds
    (or ``skip_refs`` is about to add) is not recorded again.

    ``cited`` items: ``source_class``, ``source_id``, ``claim`` and optionally
    ``locator`` (the field path the answer cited).
    """
    held = {e.source_ref for e in case.evidence} | set(skip_refs)
    updates: list[CaseUpdate] = []
    for item in cited:
        source_class = item.get("source_class")
        source_id = item.get("source_id")
        if source_class not in REF_KIND or source_class == SourceClass.USER_SUPPLIED.value:
            continue
        if not source_id:
            continue
        ref = f"{REF_KIND[source_class]}:{source_id}"
        if ref in held:
            continue
        held.add(ref)
        updates.append(CaseUpdate(
            op="record_evidence", actor=Actor.SERVER.value, run_id=run_id, at=at,
            payload={
                "claim": str(item.get("claim") or "").strip()[:MAX_TEXT] or "(cited by the answer)",
                "source_class": source_class, "source_ref": ref,
                "stance": Stance.CONTEXTUAL.value,
                "directness": Directness.NOT_ASSESSED.value,
                "locator": item.get("locator"),
            },
        ))
    return updates

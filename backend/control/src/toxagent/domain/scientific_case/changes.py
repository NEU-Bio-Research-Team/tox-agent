"""What one turn changed in a case, and what it left as it was (AG-UI, US4).

A researcher who comes back with an assay result needs to see two things at
once: what the new turn changed -- a hypothesis weakened, an uncertainty
resolved, a new conclusion -- and what it did not touch: the frozen estimate,
its interval, the policy decision, the context they gave earlier. The second
half is as much the point as the first. A turn that silently rewrote the
screening's numbers would look, in a list of changes alone, exactly like one
that did not.

The case is append-only and a pure fold of its log (``operations.replay``), so
"before" and "after" are not guesses: they are the case replayed to two
revisions. A turn's window starts where the previous run's last update left
the case -- so context the researcher added between two turns is reported as
new in the turn that first saw it -- and ends at the run's own last update.

The subject is the case's, never the run's: two subjects are two cases, so a
turn cannot change it, and "kept" says so rather than leaving it implied.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from .model import CaseUpdate, Conclusion, ScientificCaseV1
from .operations import replay

CHANGES_SCHEMA_VERSION = "case-changes-v1"


class RunNotInCase(LookupError):
    """The run never updated this case."""


def _by_id(items: Iterable[Any]) -> dict[str, Any]:
    return {item.id: item for item in items}


def _section(before: Iterable[Any], after: Iterable[Any], *, fields: Sequence[str],
             view) -> dict[str, Any]:
    """Added, changed (field by field, before -> after) and kept, by id."""
    old, new = _by_id(before), _by_id(after)
    added = [view(item) for key, item in new.items() if key not in old]
    changed: list[dict[str, Any]] = []
    kept: list[str] = []
    for key, item in new.items():
        if key not in old:
            continue
        deltas = [
            {"field": name, "before": getattr(old[key], name), "after": getattr(item, name)}
            for name in fields if getattr(old[key], name) != getattr(item, name)
        ]
        if deltas:
            changed.append({**view(item), "changes": deltas})
        else:
            kept.append(key)
    # The log is append-only, so nothing should ever be removed. Reported if
    # it ever is, rather than hidden by a diff that only looks forward.
    removed = sorted(key for key in old if key not in new)
    return {"added": added, "changed": changed, "kept": kept, "removed": removed}


def _hypothesis(h) -> dict[str, Any]:
    return {"id": h.id, "statement": h.statement, "status": h.status,
            "status_reason": h.status_reason, "actor": h.actor}


def _evidence(e) -> dict[str, Any]:
    return {"id": e.id, "claim": e.claim, "source_class": e.source_class,
            "source_ref": e.source_ref, "stance": e.stance, "directness": e.directness,
            "hypothesis_ids": list(e.hypothesis_ids), "actor": e.actor}


def _uncertainty(u) -> dict[str, Any]:
    return {"id": u.id, "description": u.description, "severity": u.severity,
            "status": u.status, "resolution": u.resolution, "actor": u.actor}


def _context(c) -> dict[str, Any]:
    return {"id": c.id, "key": c.key, "value": c.value, "unit": c.unit, "source": c.source,
            "scope": dict(c.scope) if c.scope else None, "actor": c.actor}


def _next_test(t) -> dict[str, Any]:
    return {"id": t.id, "test": t.test, "discriminates": list(t.discriminates),
            "actor": t.actor}


def _conclusion(c) -> dict[str, Any]:
    return {**c.to_dict(), "status": c.status}


def _artifacts(binding: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """A run's artifacts by what they are -- kind and origin id -- since a
    replayed artifact's session-local copy gets a new id on every run."""
    return {
        f"{ref.get('kind')}:{ref.get('source_id')}": dict(ref)
        for ref in binding.get("artifact_refs") or ()
    }


def _artifact_changes(previous, current) -> dict[str, Any]:
    if current is None or not current.binding:
        return {"compared": False, "reason": "the run recorded no subject binding",
                "added": [], "changed": [], "kept": [], "removed": []}
    if previous is None:
        return {"compared": False, "reason": "the first run of the case",
                "added": list(_artifacts(current.binding).values()),
                "changed": [], "kept": [], "removed": []}
    old, new = _artifacts(previous.binding), _artifacts(current.binding)
    changed = [
        {"kind": ref.get("kind"), "source_id": ref.get("source_id"),
         "before_sha256": old[key].get("content_sha256"),
         "after_sha256": ref.get("content_sha256")}
        for key, ref in new.items()
        if key in old and old[key].get("content_sha256") != ref.get("content_sha256")
    ]
    kept = [ref for key, ref in new.items()
            if key in old and old[key].get("content_sha256") == ref.get("content_sha256")]
    return {
        "compared": True, "reason": None,
        "added": [ref for key, ref in new.items() if key not in old],
        "changed": changed, "kept": kept,
        "removed": [ref for key, ref in old.items() if key not in new],
    }


def run_changes(updates: Sequence[CaseUpdate], run_id: str) -> dict[str, Any]:
    """The case before and after ``run_id``'s turn, compared.

    Raises :class:`RunNotInCase` when no update of the log names the run.
    """
    order: list[str] = []
    last_index: dict[str, int] = {}
    for index, update in enumerate(updates):
        if update.run_id:
            if update.run_id not in last_index:
                order.append(update.run_id)
            last_index[update.run_id] = index
    if run_id not in last_index:
        raise RunNotInCase(run_id)
    position = order.index(run_id)
    previous_run_id = order[position - 1] if position > 0 else None
    end = last_index[run_id] + 1
    start = last_index[previous_run_id] + 1 if previous_run_id is not None else 0

    before: ScientificCaseV1 | None = replay(updates[:start]) if start > 0 else None
    after = replay(updates[:end])
    window = updates[start:end]

    def items(case: ScientificCaseV1 | None, name: str) -> tuple:
        return getattr(case, name) if case is not None else ()

    current_run = next((r for r in after.runs if r.run_id == run_id), None)
    previous_run = (
        next((r for r in after.runs if r.run_id == previous_run_id), None)
        if previous_run_id is not None else None
    )
    conclusion_before = before.conclusion if before is not None else None
    # A first turn that set no conclusion did not change one.
    baseline = conclusion_before if conclusion_before is not None else Conclusion()
    actors: dict[str, int] = {}
    for update in window:
        actors[update.actor] = actors.get(update.actor, 0) + 1

    return {
        "schema_version": CHANGES_SCHEMA_VERSION,
        "case_id": after.id,
        "run_id": run_id,
        "previous_run_id": previous_run_id,
        "from_revision": before.revision if before is not None else 0,
        "to_revision": after.revision,
        # A case has one subject; a turn on another subject is another case.
        "subject": {"subject_key": after.subject_key, "kept": True},
        "updates_by_actor": actors,
        "operations": [
            {"revision": u.revision, "op": u.op, "actor": u.actor, "run_id": u.run_id}
            for u in window
        ],
        "context": _section(items(before, "context"), after.context,
                            fields=("value", "unit", "source", "scope"), view=_context),
        "hypotheses": _section(items(before, "hypotheses"), after.hypotheses,
                               fields=("status", "status_reason", "statement"),
                               view=_hypothesis),
        "evidence": _section(items(before, "evidence"), after.evidence,
                             fields=("stance", "directness", "claim"), view=_evidence),
        "uncertainties": _section(items(before, "uncertainties"), after.uncertainties,
                                  fields=("status", "resolution", "severity"),
                                  view=_uncertainty),
        "next_tests": _section(items(before, "next_tests"), after.next_tests,
                               fields=("test", "discriminates"), view=_next_test),
        "conclusion": {
            "changed": baseline.to_dict() != after.conclusion.to_dict(),
            "before": _conclusion(conclusion_before) if conclusion_before is not None else None,
            "after": _conclusion(after.conclusion),
            "released_kept": (
                _conclusion(after.released_conclusion)
                if after.released_conclusion is not None else None
            ),
        },
        "data_scope": {
            "changed": before is not None and before.data_scope != after.data_scope,
            "before": before.data_scope.to_dict() if before is not None else None,
            "after": after.data_scope.to_dict(),
        },
        "run": {
            "disposition": current_run.disposition if current_run is not None else None,
            "stop_reason": current_run.stop_reason if current_run is not None else None,
            "answer_id": current_run.answer_id if current_run is not None else None,
        },
        # The frozen artifacts the run was given: an estimate, an interval and
        # a policy decision are the artifact's, so a turn that kept them kept
        # their hashes. A changed hash is a changed input, never a judgement.
        "artifacts": _artifact_changes(previous_run, current_run),
    }

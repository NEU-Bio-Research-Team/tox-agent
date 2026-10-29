"""Scientific procedures: the determinate work, done by code (#66, §7.1/§7.2).

§7.1 splits the tool surface in two. Primitives read and write artifacts --
search, read a record, update the case. Procedures are *typed compositions of
determinate operations*: normalise a set of measurements, drop the records
that are the same experiment twice, set two values against each other, find
the claims a changed premise left stale. The investigator still chooses which
procedure to run and with what parameters; what happens inside is settled code
rather than a model doing arithmetic in prose.

Four rules from §7.2, each of which is a way a summary can lie:

* **Declare the criteria.** A result says what it filtered on, so "3 records"
  cannot quietly mean "3 of 40 that matched a rule nobody stated".
* **Name the exclusions.** Every record left out comes back with the reason.
  A procedure that silently drops the inconvenient half of the evidence is
  worse than no procedure.
* **Name the missing fields.** What the records did not state is part of the
  answer (§5.2): unknown is unknown.
* **Count the sub-operations.** "If a procedure calls ten tools internally,
  the benchmark records ten sub-operations." One API call must not hide a
  budget, so every result carries its own tally for the run's accounting.

Nothing here reaches a provider or writes to the case: these are pure
functions over records the run already read, which is what makes them
replayable in a frozen benchmark and safe to call again on a retry.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from ...domain.evidence_relation import EvidenceScope
from ...domain.measurement_value import Agreement
from ...domain.scientific_case import (
    ScientificCaseV1,
    cycles,
    needs_reassessment,
    unresolved_references,
)
from ...research.measurement import measurement_constraint, measurement_scope

PROCEDURE_SCHEMA_VERSION = "scientific-procedure-v1"

#: The axes that identify one experiment. Two records agreeing on all of the
#: ones they state, and coming from the same document, are one measurement
#: reported twice -- which is the §8/§13 trap: "two records of one experiment"
#: is not two independent sources, however many databases carry it.
#:
#: ``unit`` is deliberately absent. The key carries the *canonical* value
#: (#59), so the magnitude is already compared in one scale; keeping the unit
#: string here as well would make a nM row and a µM row of one measurement
#: two origins -- the representation-level false difference §5.2 exists to
#: remove, reintroduced one layer up.
ORIGIN_AXES: tuple[str, ...] = (
    "ligand_id", "target_id", "target_variant", "assay_type", "readout",
)

#: Provider fact keys that name the document a record came from.
_DOCUMENT_KEYS: tuple[str, ...] = (
    "document_chembl_id", "document_id", "doc_id", "pmid", "pmcid", "doi", "source_id",
)


@dataclass(frozen=True, slots=True)
class ProcedureSpec:
    """What a procedure is, and what it is allowed to touch."""

    name: str
    version: str
    summary: str
    #: Kinds of input it reads. Pure procedures read only what is handed in.
    reads: tuple[str, ...] = ()
    #: Kinds of state it writes. Empty for every procedure here, on purpose:
    #: a procedure proposes, the investigator commits (§6.2 step six).
    writes: tuple[str, ...] = ()
    cost_class: str = "cheap"

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "version": self.version, "summary": self.summary,
                "reads": list(self.reads), "writes": list(self.writes),
                "cost_class": self.cost_class}


@dataclass(frozen=True, slots=True)
class ProcedureResult:
    """What a procedure found, and everything needed to audit how."""

    procedure: str
    version: str
    facts: Mapping[str, Any] = field(default_factory=dict)
    counts: Mapping[str, int] = field(default_factory=dict)
    #: ``{"ref": ..., "reason": ...}`` for each record left out.
    exclusions: tuple[Mapping[str, Any], ...] = ()
    #: Fields the inputs did not state, by ref.
    missing_fields: tuple[str, ...] = ()
    #: Refs to drill into. The content stays behind a tool (§I3).
    refs: tuple[str, ...] = ()
    #: The determinate steps this call performed, for the run's budget.
    sub_operations: tuple[Mapping[str, Any], ...] = ()
    #: What it filtered or grouped on.
    criteria: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = PROCEDURE_SCHEMA_VERSION

    @property
    def sub_operation_count(self) -> int:
        return len(self.sub_operations)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "procedure": self.procedure, "version": self.version,
            "facts": dict(self.facts), "counts": dict(self.counts),
            "exclusions": [dict(e) for e in self.exclusions],
            "missing_fields": list(self.missing_fields),
            "refs": list(self.refs),
            "sub_operations": [dict(s) for s in self.sub_operations],
            "sub_operation_count": self.sub_operation_count,
            "criteria": dict(self.criteria),
        }


# ── normalise measurements ────────────────────────────────────────────────

NORMALISE = ProcedureSpec(
    name="normalise_measurements",
    version="normalise-measurements-v1",
    summary="Canonical value, unit and constraint kind for each record, with the "
            "conversion used and the decisive fields the record did not state.",
    reads=("evidence_records",),
)


def _decisive_missing(scope: EvidenceScope) -> list[str]:
    declared = scope.declared()
    return [axis for axis in ("target_id", "assay_type", "readout", "species")
            if axis not in declared]


def normalise_measurements(
    records: Sequence[tuple[str, Mapping[str, Any]]], *, endpoint: str | None = None,
) -> ProcedureResult:
    """Each record's value in canonical units, with its conversion (#59).

    A record whose unit this domain cannot place is **excluded with the
    reason**, not coerced: §5.2's "unit không biết là không biết".
    """
    rows: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    missing: list[str] = []
    sub_ops: list[dict[str, Any]] = []
    for ref, facts in records:
        constraint = measurement_constraint(facts)
        scope = measurement_scope(facts, endpoint=endpoint)
        sub_ops.append({"op": "normalise", "ref": ref, "kind": constraint.kind.value})
        unstated = _decisive_missing(scope)
        if unstated:
            missing.append(f"{ref}: {', '.join(unstated)}")
        if not constraint.placed:
            exclusions.append({"ref": ref, "reason": f"value is {constraint.kind.value}"})
            continue
        rows.append({"ref": ref, "constraint": constraint.to_dict(),
                     "scope": scope.declared()})
    return ProcedureResult(
        procedure=NORMALISE.name, version=NORMALISE.version,
        facts={"measurements": rows},
        counts={"in": len(records), "normalised": len(rows), "excluded": len(exclusions)},
        exclusions=tuple(exclusions), missing_fields=tuple(missing),
        refs=tuple(row["ref"] for row in rows), sub_operations=tuple(sub_ops),
        criteria={"endpoint": endpoint, "canonical_units": "per dimension",
                  "excluded_when": "the unit cannot be placed in a known dimension"},
    )


# ── deduplicate by experimental origin ────────────────────────────────────

DEDUPLICATE = ProcedureSpec(
    name="deduplicate_by_origin",
    version="deduplicate-by-origin-v1",
    summary="One record per experiment: records agreeing on the identity axes they "
            "state and sharing a source document are the same measurement reported twice.",
    reads=("evidence_records",),
)


def _origin_key(facts: Mapping[str, Any], endpoint: str | None) -> tuple:
    scope = measurement_scope(facts, endpoint=endpoint)
    declared = scope.declared()
    document = next((str(facts[key]) for key in _DOCUMENT_KEYS if facts.get(key)), "")
    constraint = measurement_constraint(facts)
    # The canonical interval as text, via the constraint's own serialisation:
    # Decimal("0.3") * 1000 is "300.0" and Decimal(300) is "300", and the two
    # are one number. Comparing str(Decimal) would have made them two origins.
    canonical = constraint.to_dict()["canonical"]
    return (
        document,
        tuple(declared.get(axis, "") for axis in ORIGIN_AXES),
        canonical["low"],
        canonical["high"],
    )


def deduplicate_by_origin(
    records: Sequence[tuple[str, Mapping[str, Any]]], *, endpoint: str | None = None,
) -> ProcedureResult:
    """Keep the first record of each experiment, excluding the repeats by name.

    §8: "Hai workers đọc lại cùng một paper không tạo hai nguồn độc lập." The
    same holds for one worker reading a database that aggregated the paper.
    Independence is a property of the measurement's origin, not of how many
    rows carry it, so a run that counted rows was counting its own retrieval.
    """
    kept: list[dict[str, Any]] = []
    seen: dict[tuple, str] = {}
    exclusions: list[dict[str, Any]] = []
    sub_ops: list[dict[str, Any]] = []
    for ref, facts in records:
        key = _origin_key(facts, endpoint)
        sub_ops.append({"op": "origin_key", "ref": ref})
        first = seen.get(key)
        if first is not None:
            exclusions.append({
                "ref": ref,
                "reason": f"same experimental origin as {first}; not an independent source",
            })
            continue
        seen[key] = ref
        kept.append({"ref": ref, "origin": {"document": key[0], "axes": list(key[1])}})
    return ProcedureResult(
        procedure=DEDUPLICATE.name, version=DEDUPLICATE.version,
        facts={"independent_records": kept},
        counts={"in": len(records), "independent": len(kept),
                "duplicates": len(exclusions)},
        exclusions=tuple(exclusions),
        refs=tuple(row["ref"] for row in kept), sub_operations=tuple(sub_ops),
        criteria={"origin_axes": list(ORIGIN_AXES),
                  "document_keys": list(_DOCUMENT_KEYS),
                  "rule": "same document and same stated identity axes and same value"},
    )


# ── compare two measurements ──────────────────────────────────────────────

COMPARE = ProcedureSpec(
    name="compare_measurements",
    version="compare-measurements-v1",
    summary="Whether two records can be set against each other, and whether their "
            "values agree, under a declared measurement tolerance.",
    reads=("evidence_records",),
)


def compare_measurements(
    left: tuple[str, Mapping[str, Any]],
    right: tuple[str, Mapping[str, Any]],
    *,
    endpoint: str | None = None,
    tolerance_fold: Any = None,
) -> ProcedureResult:
    """Scope first, then value (#59).

    Two answers, never one: a pair can be incomparable (so no disagreement is
    established) or comparable and undetermined (so the numbers differ and the
    tolerance does not settle it). Collapsing either into "contradiction" is
    the false-contradiction error on the critical list.
    """
    left_ref, left_facts = left
    right_ref, right_facts = right
    left_scope = measurement_scope(left_facts, endpoint=endpoint)
    right_scope = measurement_scope(right_facts, endpoint=endpoint)
    comparison = left_scope.compare(right_scope, measurement=True)
    left_value = measurement_constraint(left_facts)
    right_value = measurement_constraint(right_facts)
    sub_ops = [
        {"op": "scope_compare", "refs": [left_ref, right_ref]},
        {"op": "value_compare", "refs": [left_ref, right_ref]},
    ]
    if comparison.state.value != "comparable":
        detail = (
            f"they differ on {', '.join(comparison.differing_axes)}"
            if comparison.differing_axes
            else f"no shared declared axis for {', '.join(comparison.missing_groups)}"
            if comparison.missing_groups else "the scopes state too little"
        )
        agreement = Agreement.UNDETERMINED
        reason = f"not comparable ({comparison.state.value}): {detail}"
    else:
        agreement, reason = left_value.compare(right_value, tolerance_fold=tolerance_fold)
    return ProcedureResult(
        procedure=COMPARE.name, version=COMPARE.version,
        facts={
            "comparability": comparison.state.value,
            "differing_axes": list(comparison.differing_axes),
            "written_differences": list(comparison.written_differences),
            "missing_groups": list(comparison.missing_groups),
            "agreement": agreement.value,
            "reason": reason,
            "left": {"ref": left_ref, "constraint": left_value.to_dict()},
            "right": {"ref": right_ref, "constraint": right_value.to_dict()},
        },
        counts={"in": 2},
        refs=(left_ref, right_ref), sub_operations=tuple(sub_ops),
        criteria={"tolerance_fold": None if tolerance_fold is None else str(tolerance_fold),
                  "rule": "scope comparability first; a declared tolerance is required "
                          "before disjoint values count as a disagreement"},
    )


# ── check what a changed premise left stale ───────────────────────────────

CHECK_DEPENDENCIES = ProcedureSpec(
    name="check_dependencies",
    version="check-dependencies-v1",
    summary="Derived claims a changed premise left waiting, plus inputs nothing in "
            "the case provides and any dependency cycle.",
    reads=("case",),
)


def check_dependencies(
    case: ScientificCaseV1, *, known_refs: Iterable[str] = (),
) -> ProcedureResult:
    """#60's graph, as a procedure the investigator can call before answering."""
    stale = needs_reassessment(case.derived_claims)
    unresolved = unresolved_references(case.derived_claims, known_refs)
    found_cycles = cycles(case.derived_claims)
    return ProcedureResult(
        procedure=CHECK_DEPENDENCIES.name, version=CHECK_DEPENDENCIES.version,
        facts={
            "needs_reassessment": [
                {"claim_id": c.id, "statement": c.statement, "note": c.note} for c in stale
            ],
            "unresolved_references": [
                {"claim_id": claim_id, "ref": ref} for claim_id, ref in unresolved
            ],
            "cycles": [list(cycle) for cycle in found_cycles],
        },
        counts={"claims": len(case.derived_claims), "stale": len(stale),
                "unresolved": len(unresolved), "cycles": len(found_cycles)},
        refs=tuple(c.id for c in stale),
        sub_operations=(
            {"op": "needs_reassessment"}, {"op": "unresolved_references"}, {"op": "cycles"},
        ),
        criteria={"known_refs": sorted(set(known_refs))[:20],
                  "rule": "a claim is stale when every justification lost a premise"},
    )


# ── the allowlist ─────────────────────────────────────────────────────────

#: §7.2: an allowlisted runner, not a general shell. A name not here does not
#: run, and the refusal names what is available.
CATALOG: dict[str, ProcedureSpec] = {
    spec.name: spec for spec in (NORMALISE, DEDUPLICATE, COMPARE, CHECK_DEPENDENCIES)
}


class UnknownProcedure(ValueError):
    """Refused; the message is written for the model that sent it."""


def spec_for(name: str) -> ProcedureSpec:
    try:
        return CATALOG[name]
    except KeyError:
        raise UnknownProcedure(
            f"{name!r} is not a scientific procedure; one of {sorted(CATALOG)}"
        ) from None

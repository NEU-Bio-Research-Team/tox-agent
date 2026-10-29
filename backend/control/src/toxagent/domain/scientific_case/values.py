"""Researcher-supplied values, as versions a claim can cite (#61).

§5.1 counts "values the researcher supplied" among the case's observations,
each with its own producer and provenance, and §5.3 needs them addressable:
the margin claim in the proposal depends on ``assay_measurement_v1`` *and*
``user_exposure_v1``, and when the researcher corrects the exposure the old
claim must still say which number it was computed from.

The history is already in the case. A :class:`ContextItem` is appended, never
edited, so correcting a value adds a second item under the same key and the
first one stays where it was. This module reads that log as what it already
is -- an ordered set of versions per key -- rather than adding a parallel
store that could disagree with it.

Two things it is careful about:

* **A value the researcher stated is not a verified measurement.** The
  verification status is explicit and its default is ``user_asserted``;
  ``sme_verified`` is not something a run can award itself (see
  ``operations._add_context``).
* **A version reference is stable.** ``case_value:free_cmax@v2`` names a
  version of a value in a case, and resolves the same way in every later run.
  An observation id is minted per run, so a dependency graph built on
  observation ids would go stale between turns; claims reference these.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterator

from .model import ContextItem, ScientificCaseV1

#: Scheme of a stable reference to one version of one researcher value.
VALUE_REF_SCHEME = "case_value"


class VerificationStatus(str, Enum):
    """How much standing a value has. Three states, not a boolean.

    ``USER_ASSERTED`` is what the researcher typed: usable, citable, and not
    evidence. ``SOURCE_BACKED`` is a value they attributed to a source the
    session holds. ``SME_VERIFIED`` is the one the study's gold needs, and the
    product cannot mint it on its own.
    """

    USER_ASSERTED = "user_asserted"
    SOURCE_BACKED = "source_backed"
    SME_VERIFIED = "sme_verified"


#: What an item with no stated status is. Absent means "the researcher said
#: so", which is the weakest of the three -- never the strongest.
DEFAULT_VERIFICATION = VerificationStatus.USER_ASSERTED


@dataclass(frozen=True, slots=True)
class ValueVersion:
    """One version of one researcher-supplied value."""

    key: str
    version: int
    item_id: str
    value: str
    unit: str | None
    source: str | None
    scope: dict[str, str] | None
    verification: VerificationStatus
    actor: str
    run_id: str | None
    #: Set when a later version of the same key exists. The old version stays
    #: citable: a claim computed from it is not retroactively about the new
    #: number (§5.3).
    superseded_by: str | None = None

    @property
    def ref(self) -> str:
        return f"{VALUE_REF_SCHEME}:{self.key}@v{self.version}"

    @property
    def current(self) -> bool:
        return self.superseded_by is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ref": self.ref,
            "key": self.key,
            "version": self.version,
            "item_id": self.item_id,
            "value": self.value,
            "unit": self.unit,
            "source": self.source,
            "scope": dict(self.scope) if self.scope else None,
            "verification": self.verification.value,
            "actor": self.actor,
            "run_id": self.run_id,
            "superseded_by": self.superseded_by,
            "current": self.current,
        }


def _verification(item: ContextItem) -> VerificationStatus:
    try:
        return VerificationStatus(item.verification or DEFAULT_VERIFICATION.value)
    except ValueError:
        # A status this build does not know is not a licence to assume the
        # strongest one.
        return DEFAULT_VERIFICATION


def value_versions(case: ScientificCaseV1) -> tuple[ValueVersion, ...]:
    """Every researcher value in the case, in the order it was stated.

    Version numbers run per key, from 1, in case order -- so they are a
    property of the log and not of when this function ran.
    """
    seen: dict[str, int] = {}
    rows: list[ValueVersion] = []
    for item in case.context:
        seen[item.key] = seen.get(item.key, 0) + 1
        rows.append(ValueVersion(
            key=item.key, version=seen[item.key], item_id=item.id, value=item.value,
            unit=item.unit, source=item.source,
            scope=dict(item.scope) if item.scope else None,
            verification=_verification(item), actor=item.actor, run_id=item.run_id,
        ))
    latest = {row.key: row.item_id for row in rows}
    return tuple(
        row if latest[row.key] == row.item_id else
        _superseded(row, next(r.ref for r in rows
                              if r.key == row.key and r.version == row.version + 1))
        for row in rows
    )


def _superseded(row: ValueVersion, by: str) -> ValueVersion:
    return ValueVersion(
        key=row.key, version=row.version, item_id=row.item_id, value=row.value,
        unit=row.unit, source=row.source, scope=row.scope,
        verification=row.verification, actor=row.actor, run_id=row.run_id,
        superseded_by=by,
    )


def current_values(case: ScientificCaseV1) -> tuple[ValueVersion, ...]:
    """The version of each key that is in force now."""
    return tuple(row for row in value_versions(case) if row.current)


def resolve_value_ref(case: ScientificCaseV1, ref: str) -> ValueVersion | None:
    """The version a reference names, or ``None`` when the case has no such
    version. An unresolvable reference is a finding, not an exception: §5.3
    asks for unresolved references to be *detected*."""
    for row in value_versions(case):
        if row.ref == ref:
            return row
    return None


def version_for_item(case: ScientificCaseV1, item_id: str) -> ValueVersion | None:
    """The version a context item is. A run cites ``context:c2``; a claim
    computed from it must cite ``case_value:free_cmax@v2``, because the item id
    says nothing about which value it replaced."""
    for row in value_versions(case):
        if row.item_id == item_id:
            return row
    return None


def is_value_ref(ref: str) -> bool:
    return ref.startswith(f"{VALUE_REF_SCHEME}:")


def iter_history(case: ScientificCaseV1, key: str) -> Iterator[ValueVersion]:
    """Every version of one key, oldest first."""
    return (row for row in value_versions(case) if row.key == key)

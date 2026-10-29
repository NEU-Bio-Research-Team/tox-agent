"""A provider record's measurement, in the one scope vocabulary the domain has.

The review asked for a normalised record -- molecule identifier, endpoint,
assay type, species or cell system, the relevant conditions, readout, value,
relation and unit -- and in the same breath warned against building a second
ontology beside the existing one. So this does not define a new type: it
projects whatever facts a provider gave into
:class:`~toxagent.domain.evidence_relation.EvidenceScope`, which already names
those axes and is what relation assessment compares.

Absent means absent. A field the provider did not state stays ``None``, which
the scope treats as *not stated* rather than as agreement, so nothing here
infers that two records measured the same thing because both mention hERG.
"""
from __future__ import annotations

from typing import Any, Mapping

from ..domain.evidence_relation import EvidenceScope
from ..domain.measurement_value import NumericConstraint

#: Provider fact keys, in the order they are tried, per scope axis. ChEMBL's
#: own names come first because they are the ones this deployment has; a
#: second provider adds its names here rather than renaming its payload.
_AXIS_SOURCES: dict[str, tuple[str, ...]] = {
    "ligand_id": ("molecule_chembl_id", "ligand_id", "canonical_smiles"),
    "target_id": ("target_chembl_id", "target_id", "uniprot_id"),
    "target_variant": ("target_variant", "mutation"),
    "target_sequence_sha256": ("target_sequence_sha256", "sequence_sha256"),
    "assay_type": ("assay_type",),
    "readout": ("standard_type", "readout"),
    "unit": ("standard_units", "unit"),
    "comparator": ("standard_relation", "comparator"),
    "species": ("species", "organism", "assay_organism"),
    "experimental_context": ("assay_description", "experimental_context", "cell_system"),
}

#: Not an axis of comparability, but part of the record a claim quotes.
VALUE_KEYS = ("standard_value", "value", "pchembl_value")


def _first(facts: Mapping[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = facts.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def measurement_scope(
    facts: Mapping[str, Any], *, endpoint: str | None = None
) -> EvidenceScope:
    """The scope a record's own facts establish, and nothing more.

    ``endpoint`` is passed in rather than guessed: a record can be about hERG
    without saying so in a field, and the caller knows which proposition it
    went looking for. Guessing it from free text is how a source measuring
    something else gets treated as comparable.
    """
    axes = {axis: _first(facts, keys) for axis, keys in _AXIS_SOURCES.items()}
    return EvidenceScope(endpoint=endpoint, **axes)


def measurement_constraint(facts: Mapping[str, Any]) -> NumericConstraint:
    """What the record's number permits, in canonical units (#59, §5.2).

    The identity layer above says whether two records are about the same
    measurement; this says what each one's value establishes. A record whose
    unit this domain cannot place comes back ``unknown_unit`` -- the record is
    kept, the comparison is refused.
    """
    return NumericConstraint.parse(
        _first(facts, VALUE_KEYS),
        _first(facts, ("standard_units", "unit")),
        comparator=_first(facts, ("standard_relation", "comparator")),
    )


def measured_value(facts: Mapping[str, Any]) -> dict[str, Any]:
    """The value a claim would quote, with the relation and unit it needs.

    ``"> 30 µM"`` is a bound and ``"= 30 µM"`` is a measurement; dropping the
    relation turns one into the other, so it travels with the number or the
    value is reported as not assessed.

    Since #59 the canonical form travels beside the raw one, with the
    conversion that produced it: comparison uses the canonical number, an
    answer quotes the raw one, and the researcher can check the step between.
    """
    raw = _first(facts, VALUE_KEYS)
    if raw is None:
        return {"value": None, "state": "not_assessed"}
    try:
        value: float | str = float(raw)
    except ValueError:
        value = raw
    constraint = measurement_constraint(facts)
    return {
        "value": value,
        "relation": _first(facts, ("standard_relation", "comparator")) or "=",
        "unit": _first(facts, ("standard_units", "unit")),
        "state": "measured",
        "constraint_kind": constraint.kind.value,
        "canonical": constraint.to_dict()["canonical"],
        "conversion": None if constraint.conversion is None else constraint.conversion.to_dict(),
    }


def is_measured_value(facts: Mapping[str, Any]) -> bool:
    """Whether a record states a value, which makes it a measurement to
    compare as one, rather than a statement about a finding."""
    return _first(facts, VALUE_KEYS) is not None


def normalised_record(
    facts: Mapping[str, Any], *, endpoint: str | None = None
) -> dict[str, Any]:
    """One record as scope plus value, for storing beside the evidence."""
    scope = measurement_scope(facts, endpoint=endpoint)
    return {
        "scope": scope.to_dict(),
        "declared_axes": sorted(scope.declared()),
        "measurement": measured_value(facts),
        "constraint": measurement_constraint(facts).to_dict(),
    }

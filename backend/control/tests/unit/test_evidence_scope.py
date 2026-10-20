"""Scope is what tells a disagreement from two different measurements.

Two records can both say "hERG" and measure different things. The four
outcomes that have to stay apart: a real contradiction, different measurement
conditions, indirect evidence, and not enough data.
"""
from __future__ import annotations

from datetime import datetime, timezone

from toxagent.domain import activity_summary
from toxagent.domain.evidence_relation import (
    MEASUREMENT_AXES,
    Applicability,
    Directness,
    EvidenceRelationAssessment,
    EvidenceScope,
    RelationLabel,
    SourceClass,
    SourceRef,
    Strength,
    new_proposition_id,
    scope_conflicts,
)
from toxagent.domain.ids import RUN, SESSION, new_id


NOW = datetime(2026, 10, 20, tzinfo=timezone.utc)


def _relation(
    *,
    proposition_id: str,
    relation: RelationLabel,
    scope: EvidenceScope,
    source_id: str = "evi_1",
) -> EvidenceRelationAssessment:
    return EvidenceRelationAssessment.create(
        session_id=new_id(SESSION),
        run_id=new_id(RUN),
        proposition_id=proposition_id,
        source_ref=SourceRef(
            source_class=SourceClass.EXTERNAL_EXPERIMENTAL, source_id=source_id
        ),
        relation=relation,
        directness=Directness.DIRECT,
        applicability=Applicability.OK,
        strength=Strength.MODERATE,
        reason_codes=("measured_value",),
        scope=scope,
        now=NOW,
    )


def test_a_scope_only_compares_axes_both_sides_declare():
    """A field left unset is "not stated", not "the same as yours"."""
    stated = EvidenceScope(endpoint="herg", assay_type="B")
    silent = EvidenceScope(endpoint="herg")
    assert stated.differences(silent) == ()
    assert silent.differences(stated) == ()


def test_the_same_endpoint_measured_by_a_different_assay_is_a_difference():
    binding = EvidenceScope(endpoint="herg", assay_type="B", readout="Ki")
    functional = EvidenceScope(endpoint="herg", assay_type="F", readout="IC50")
    assert binding.differences(functional) == ("assay_type", "readout")


def test_two_hERG_sources_from_different_assays_cannot_contradict_each_other():
    """The requirement from the review: both records say hERG, so the labels
    look comparable, and they are not."""
    proposition = new_proposition_id()
    relations = [
        _relation(
            proposition_id=proposition,
            relation=RelationLabel.SUPPORTS,
            scope=EvidenceScope(endpoint="herg", assay_type="B", unit="nM"),
            source_id="evi_binding",
        ),
        _relation(
            proposition_id=proposition,
            relation=RelationLabel.CONTRADICTS,
            scope=EvidenceScope(endpoint="herg", assay_type="F", unit="nM"),
            source_id="evi_patch_clamp",
        ),
    ]

    conflicts = scope_conflicts(relations)

    assert len(conflicts) == 1
    assert conflicts[0].differing_axes == ("assay_type",)
    assert "not in disagreement" in conflicts[0].describe()


def test_a_contradiction_within_one_assay_is_left_alone():
    proposition = new_proposition_id()
    scope = EvidenceScope(endpoint="herg", assay_type="F", unit="nM")
    relations = [
        _relation(
            proposition_id=proposition,
            relation=RelationLabel.SUPPORTS,
            scope=scope,
            source_id="evi_a",
        ),
        _relation(
            proposition_id=proposition,
            relation=RelationLabel.CONTRADICTS,
            scope=scope,
            source_id="evi_b",
        ),
    ]
    assert scope_conflicts(relations) == ()


def test_relations_on_different_propositions_are_not_compared():
    relations = [
        _relation(
            proposition_id=new_proposition_id(),
            relation=RelationLabel.CONTRADICTS,
            scope=EvidenceScope(endpoint="herg", assay_type="B"),
        ),
        _relation(
            proposition_id=new_proposition_id(),
            relation=RelationLabel.SUPPORTS,
            scope=EvidenceScope(endpoint="herg", assay_type="F"),
        ),
    ]
    assert scope_conflicts(relations) == ()


def test_a_different_target_variant_is_a_measurement_difference():
    wild_type = EvidenceScope(target_id="herg", target_variant="wild_type")
    mutant = EvidenceScope(target_id="herg", target_variant="Y652A")
    assert wild_type.differences(mutant) == ("target_variant",)


def test_the_measurement_axes_cover_what_makes_a_measurement_the_same():
    for axis in (
        "ligand_id",
        "target_id",
        "target_variant",
        "target_sequence_sha256",
        "assay_type",
        "readout",
        "unit",
        "comparator",
        "experimental_context",
    ):
        assert axis in MEASUREMENT_AXES, axis
        assert axis in EvidenceScope().to_dict(), axis


# ── the ChEMBL summary ────────────────────────────────────────────────────

def _fact(value: float, *, assay: str, kind: str = "IC50", relation: str = "="):
    return {
        "standard_type": kind,
        "standard_relation": relation,
        "standard_value": value,
        "standard_units": "nM",
        "assay_type": assay,
    }


def test_the_summary_splits_one_measurement_type_across_assay_types():
    """The terfenadine case in activity_summary's own docstring: nine values
    between 50 and 320 nM and one patch-clamp IC50 of 56 000 nM read as
    "inconsistent" when they are pooled."""
    records = [(f"evi_{i}", _fact(60.0 + i * 20, assay="B")) for i in range(9)]
    records.append(("evi_patch", _fact(56000.0, assay="F")))

    summary = activity_summary.summarize(records)

    assert summary["measurement_types_split_across_assays"] == ["IC50"]
    assert "more than one assay type" in summary["reading"]
    binding = summary["by_type_and_assay"]["IC50|B"]
    functional = summary["by_type_and_assay"]["IC50|F"]
    assert binding["n"] == 9
    assert functional["n"] == 1
    assert functional["evidence_ids"] == ["evi_patch"]
    # the pooled view is still there, so nothing that read it has changed
    assert summary["by_type"]["IC50"]["n"] == 10


def test_one_assay_type_raises_no_split_warning():
    records = [(f"evi_{i}", _fact(60.0 + i * 20, assay="F")) for i in range(4)]
    summary = activity_summary.summarize(records)
    assert summary["measurement_types_split_across_assays"] == []
    assert "more than one assay type" not in summary["reading"]
    assert list(summary["by_type_and_assay"]) == ["IC50|F"]


def test_an_assay_type_that_was_not_stated_is_its_own_group():
    records = [
        ("evi_a", _fact(100.0, assay="F")),
        ("evi_b", {**_fact(120.0, assay="F"), "assay_type": None}),
    ]
    summary = activity_summary.summarize(records)
    assert set(summary["by_type_and_assay"]) == {"IC50|F", "IC50|not_stated"}

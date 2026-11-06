"""§7.1's first two procedures, §7.2's accounting, and the gap → procedure hint (#66)."""
from __future__ import annotations

from toxagent.application.investigation import decision_loop as loop
from toxagent.application.investigation import procedures as p
from toxagent.domain import decision_state as ds


def test_subject_context_establishes_only_what_is_stated_and_names_the_rest():
    result = p.resolve_subject_context(
        {"identity": {"canonical_smiles": "CCO", "target": "KCNH2"}},
        [{"id": "c1", "key": "assay_type", "value": "patch clamp"},
         {"id": "c2", "key": "free_cmax", "value": "10 nM"}],
        endpoint="herg",
    )
    facts = result.facts
    assert facts["established"]["ligand_id"] == "CCO"
    assert facts["established"]["target_id"] == "KCNH2"
    assert facts["established"]["assay_type"] == "patch clamp"
    assert facts["source_of"]["assay_type"] == "context:c1"
    assert set(facts["unestablished_axes"]) == {"target_variant", "readout", "species"}
    assert result.refs == ("context:c1",)
    assert result.sub_operation_count == 3


def test_assay_records_keep_measured_values_and_name_every_exclusion():
    measured = {"standard_value": "300", "standard_units": "nM", "standard_relation": "="}
    records = [
        ("ev_a", measured, "chembl"),
        ("ev_b", {"title": "a review"}, "europepmc"),
        ("ev_c", measured, "pubchem"),
    ]
    result = p.collect_assay_records(records, endpoint="herg", providers=["chembl", "europepmc"])
    assert result.refs == ("ev_a",)
    reasons = {e["ref"]: e["reason"] for e in result.exclusions}
    assert reasons == {"ev_b": "states no measured value",
                       "ev_c": "provider pubchem was not asked for"}
    assert result.counts["from_chembl"] == 1 and result.sub_operation_count == 3


def test_every_section_7_1_procedure_is_in_the_allowlist():
    assert {"resolve_subject_context", "collect_assay_records", "normalise_measurements",
            "deduplicate_by_origin", "check_dependencies"} <= set(p.CATALOG)


def test_sub_operations_are_counted_in_the_runs_usage():
    state = ds.initial(session_id="ses_" + "1" * 32, run_id="run_" + "2" * 32, goal="g",
                       subject_refs=(), budget_snapshot={})
    state = ds.record_sub_operations(state, count=10)
    state = ds.record_sub_operations(state, count=3)
    assert state.usage["procedure_sub_operations"] == 13


def test_each_open_gap_names_the_procedure_that_closes_it():
    gaps = [{"gap": "measurement_unresolved"}, {"gap": "only_researcher_has_it"}]
    hinted = loop.with_procedures(gaps, available=True)
    assert hinted[0]["procedure"].endswith("normalise_measurements")
    assert "procedure" not in hinted[1]
    assert "procedure" not in loop.with_procedures(gaps, available=False)[0]

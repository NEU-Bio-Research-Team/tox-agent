"""Scientific procedures: determinate work with its exclusions named (#66).

§7.2's four rules are what these tests are about -- declare the criteria, name
the exclusions, name the missing fields, count the sub-operations -- because
each of them is a way a result can be true and still mislead.
"""
import pytest

from toxagent.application.investigation import procedures
from toxagent.domain import scientific_case as sc

SESSION = "ses_" + "1" * 32
CASE_ID = "scase_" + "2" * 32
RUN = "run_" + "6" * 32
OBS = "obs_" + "3" * 32


def _facts(**overrides):
    base = {
        "molecule_chembl_id": "CHEMBL1220",
        "target_chembl_id": "CHEMBL240",
        "standard_type": "IC50",
        "standard_relation": "=",
        "standard_value": "300",
        "standard_units": "nM",
        "assay_type": "B",
        "document_chembl_id": "DOC1",
    }
    base.update(overrides)
    return base


# ── normalise ─────────────────────────────────────────────────────────────

def test_the_same_value_in_two_units_normalises_to_one_number():
    result = procedures.normalise_measurements(
        [("e1", _facts()), ("e2", _facts(standard_value="0.3", standard_units="uM"))],
        endpoint="herg",
    )

    lows = [row["constraint"]["canonical"]["low"] for row in result.facts["measurements"]]
    assert lows == ["300", "300"]
    assert result.counts == {"in": 2, "normalised": 2, "excluded": 0}


def test_the_conversion_travels_with_the_number():
    result = procedures.normalise_measurements(
        [("e1", _facts(standard_value="0.3", standard_units="uM"))], endpoint="herg")

    assert result.facts["measurements"][0]["constraint"]["conversion"]["factor"] == "1000"


def test_a_record_in_an_unplaceable_unit_is_excluded_with_the_reason():
    """Not coerced, and not silently dropped: §5.2's "unit không biết là không
    biết" plus §7.2's "khai rejected records/reasons"."""
    result = procedures.normalise_measurements(
        [("e1", _facts()), ("e2", _facts(standard_units="fold-change"))], endpoint="herg")

    assert result.counts == {"in": 2, "normalised": 1, "excluded": 1}
    assert result.exclusions == ({"ref": "e2", "reason": "value is unknown_unit"},)


def test_the_decisive_fields_a_record_did_not_state_are_named():
    result = procedures.normalise_measurements([("e1", _facts())], endpoint="herg")

    assert result.missing_fields == ("e1: species",)


def test_every_record_is_one_sub_operation():
    """§7.2: ten internal operations are ten sub-operations in the record. One
    call must not hide a budget."""
    result = procedures.normalise_measurements(
        [(f"e{i}", _facts()) for i in range(5)], endpoint="herg")

    assert result.sub_operation_count == 5
    assert result.to_dict()["sub_operation_count"] == 5


def test_the_criteria_are_stated_with_the_counts():
    result = procedures.normalise_measurements([("e1", _facts())], endpoint="herg")

    assert result.criteria["endpoint"] == "herg"
    assert "cannot be placed" in result.criteria["excluded_when"]


# ── deduplicate by origin ─────────────────────────────────────────────────

def test_one_experiment_reported_twice_is_one_source():
    """§8/§13: two rows of one measurement are not two independent sources,
    however many databases carry them."""
    result = procedures.deduplicate_by_origin(
        [("e1", _facts()), ("e2", _facts())], endpoint="herg")

    assert result.counts == {"in": 2, "independent": 1, "duplicates": 1}
    assert "not an independent source" in result.exclusions[0]["reason"]
    assert "e1" in result.exclusions[0]["reason"]  # which one it duplicates


def test_two_experiments_in_two_documents_are_two_sources():
    result = procedures.deduplicate_by_origin(
        [("e1", _facts()), ("e2", _facts(document_chembl_id="DOC2"))], endpoint="herg")

    assert result.counts["independent"] == 2
    assert result.exclusions == ()


def test_the_same_document_with_a_different_assay_is_a_different_measurement():
    """One paper can report a binding assay and a patch clamp. Collapsing them
    would hide the very disagreement the case is about."""
    result = procedures.deduplicate_by_origin(
        [("e1", _facts()), ("e2", _facts(assay_type="F", standard_value="56000"))],
        endpoint="herg")

    assert result.counts["independent"] == 2


def test_the_same_experiment_in_two_units_is_still_one_experiment():
    """The value is compared canonically, so a µM row and a nM row of one
    measurement do not read as two."""
    result = procedures.deduplicate_by_origin(
        [("e1", _facts()), ("e2", _facts(standard_value="0.3", standard_units="uM"))],
        endpoint="herg")

    assert result.counts["independent"] == 1


def test_the_origin_rule_is_declared():
    result = procedures.deduplicate_by_origin([("e1", _facts())], endpoint="herg")

    assert "same document" in result.criteria["rule"]
    assert "ligand_id" in result.criteria["origin_axes"]


# ── compare ───────────────────────────────────────────────────────────────

def test_two_records_of_the_same_quantity_in_different_units_agree():
    result = procedures.compare_measurements(
        ("e1", _facts()), ("e2", _facts(standard_value="0.3", standard_units="uM")),
        endpoint="herg")

    assert result.facts["comparability"] == "comparable"
    assert result.facts["agreement"] == "agree"
    assert "unit" in result.facts["written_differences"]


def test_a_different_target_is_not_a_disagreement():
    result = procedures.compare_measurements(
        ("e1", _facts()), ("e2", _facts(target_chembl_id="CHEMBL999")), endpoint="herg")

    assert result.facts["comparability"] == "different"
    assert result.facts["agreement"] == "undetermined"
    assert "target_id" in result.facts["reason"]


def test_disjoint_values_need_a_declared_tolerance_before_they_disagree():
    """The terfenadine pair. Without a declared tolerance the honest answer is
    undetermined (§5.2), and the procedure says why."""
    binding = ("e1", _facts(standard_value="60"))
    patch = ("e2", _facts(standard_value="56000", document_chembl_id="DOC2"))

    undeclared = procedures.compare_measurements(binding, patch, endpoint="herg")
    declared = procedures.compare_measurements(binding, patch, endpoint="herg",
                                               tolerance_fold=10)

    assert undeclared.facts["agreement"] == "undetermined"
    assert "no measurement tolerance" in undeclared.facts["reason"]
    assert declared.facts["agreement"] == "disagree"
    assert undeclared.criteria["tolerance_fold"] is None
    assert declared.criteria["tolerance_fold"] == "10"


def test_a_censored_bound_is_compared_rather_than_refused():
    result = procedures.compare_measurements(
        ("e1", _facts(standard_relation=">", standard_value="30000")),
        ("e2", _facts(standard_value="300", document_chembl_id="DOC2")),
        endpoint="herg", tolerance_fold=2,
    )

    assert result.facts["comparability"] == "comparable"
    assert "comparator" in result.facts["written_differences"]
    assert result.facts["agreement"] == "disagree"


def test_comparing_counts_two_sub_operations():
    result = procedures.compare_measurements(("e1", _facts()), ("e2", _facts()),
                                             endpoint="herg")

    assert [op["op"] for op in result.sub_operations] == ["scope_compare", "value_compare"]


# ── check dependencies ────────────────────────────────────────────────────

def upd(op, actor="server", **payload):
    return sc.CaseUpdate(op=op, payload=payload, actor=actor,
                         at="2026-10-28T09:00:00+00:00", run_id=RUN)


def _case_with_stale_claim():
    case = sc.apply(None, upd(
        "open", case_id=CASE_ID, session_id=SESSION, subject_key="analysis:ana_x",
        subject_refs=["analysis:ana_x"], question="Does it contradict?",
    ))
    case = sc.apply(case, upd("add_context", actor="user", key="free_cmax",
                              value="0.01", unit="µM"))
    case = sc.apply(case, upd("record_derived_claim", statement="exposure_margin",
                              rule="ratio_same_concentration_unit",
                              inputs=[OBS, "case_value:free_cmax@v1"]))
    return sc.apply(case, upd("add_context", actor="user", key="free_cmax",
                              value="0.02", unit="µM"))


def test_the_procedure_finds_what_a_changed_premise_left_waiting():
    result = procedures.check_dependencies(_case_with_stale_claim(),
                                           known_refs=[OBS, "case_value:free_cmax@v1"])

    assert result.counts["stale"] == 1
    assert result.facts["needs_reassessment"][0]["claim_id"] == "dc1"
    assert result.counts["unresolved"] == 0


def test_an_input_nothing_provides_is_reported():
    result = procedures.check_dependencies(_case_with_stale_claim(), known_refs=[])

    assert result.counts["unresolved"] == 2
    assert {row["ref"] for row in result.facts["unresolved_references"]} == {
        OBS, "case_value:free_cmax@v1",
    }


def test_a_case_with_no_claims_reports_nothing_rather_than_failing():
    case = sc.apply(None, upd(
        "open", case_id=CASE_ID, session_id=SESSION, subject_key="analysis:ana_x",
        subject_refs=["analysis:ana_x"], question="q",
    ))

    result = procedures.check_dependencies(case)

    assert result.counts == {"claims": 0, "stale": 0, "unresolved": 0, "cycles": 0}


# ── the allowlist ─────────────────────────────────────────────────────────

def test_a_procedure_outside_the_allowlist_is_refused_by_name():
    """§7.2: an allowlisted runner, not a general shell."""
    with pytest.raises(procedures.UnknownProcedure) as refused:
        procedures.spec_for("rm -rf")

    assert "is not a scientific procedure" in str(refused.value)
    assert "normalise_measurements" in str(refused.value)


def test_every_procedure_declares_its_version_and_writes_nothing():
    """A procedure proposes; the investigator commits (§6.2 step six)."""
    for name, spec in procedures.CATALOG.items():
        assert spec.version.endswith("-v1"), name
        assert spec.writes == (), name
        assert spec.reads, name

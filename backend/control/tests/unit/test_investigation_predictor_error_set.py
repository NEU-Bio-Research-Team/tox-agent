"""The predictor-error case set: cases built around ToxPred's own mistakes.

The pilot set asks whether a system reasons well. This set asks the narrower
question the product owner posed: when the served predictor is wrong, does the
system notice, or does it pass the prediction through as a finding? That only
works if the cases keep the answer out of the question, if the controls are
really controls, and if the flag the lab grades cannot be raised on them.
"""
from __future__ import annotations

import json
from pathlib import Path

from evals.investigation import cases as case_module

HERE = Path(case_module.__file__).resolve().parent
SPECS_PATH = HERE / "case_specs_predictor_error.json"
CASES_DIR = HERE / "cases_predictor_error"

SPECS = json.loads(SPECS_PATH.read_text())
CASES = case_module.load_cases(CASES_DIR)
RUBRIC = json.loads((HERE / "rubric.json").read_text())

#: Words that would hand the system the finding the case is testing for.
GIVEAWAYS = (
    "torsade", "positive control", "false negative", "overcall", "over-call",
    "endogenous", "implausible", "wrong", "mistaken", "predictor error",
)


def test_the_built_cases_are_the_specs_they_were_built_from():
    specs = {s["case_id"]: s for s in SPECS["cases"]}
    assert {c["case_id"] for c in CASES} == set(specs)
    for case in CASES:
        assert case["spec_sha256"] == case_module.sha256(specs[case["case_id"]])
        assert case["reference"]["status"] == "pending_lab_verification"
        assert case["set_id"] == "predictor-error-v1"


def test_this_set_is_separate_from_the_pilot_set():
    """Two sets in one directory would silently join two studies."""
    assert CASES_DIR != case_module.CASES_DIR
    assert {c["case_id"] for c in CASES}.isdisjoint(
        {c["case_id"] for c in case_module.load_cases()}
    )


def test_every_case_declares_what_the_predictor_does():
    for case in CASES:
        error = case["reference"]["predictor_error"]
        assert error["kind"]
        assert error["what_the_predictor_says"]
        assert error["why_it_misleads"]


def test_the_question_never_gives_the_finding_away():
    """A case that tells the system the predictor is wrong tests nothing."""
    for case in CASES:
        asked = " ".join(turn["text"] for turn in case["turns"]).lower()
        assert not [word for word in GIVEAWAYS if word in asked], case["case_id"]
        assert "predict" not in asked, case["case_id"]


def test_the_set_has_controls_where_the_predictor_is_right():
    """Without them the study rewards a system that doubts everything."""
    kinds = [c["reference"]["predictor_error"]["kind"] for c in CASES]
    assert kinds.count("none") >= 2
    assert len([k for k in kinds if k != "none"]) >= 4


def test_a_control_case_says_scepticism_must_be_targeted():
    controls = [c for c in CASES if c["reference"]["predictor_error"]["kind"] == "none"]
    for case in controls:
        behaviours = " ".join(case["reference"]["good_behaviours"]).lower()
        assert "not" in behaviours and ("invent" in behaviours or "manufactur" in behaviours)


def test_an_anonymous_case_never_names_its_compound():
    anonymous = [c for c in CASES if not c["compound"]["name_disclosed"]]
    assert anonymous
    for case in anonymous:
        for turn in case["turns"]:
            assert case["compound"]["query_name"].lower() not in turn["text"].lower()


def test_every_structure_came_from_pubchem_not_from_memory():
    for case in CASES:
        assert case["compound"]["resolved_from"] == "PubChem PUG REST"
        assert case["compound"]["pubchem_cid"] > 0
        assert case["compound"]["smiles"] in " ".join(t["text"] for t in case["turns"])


def test_the_rubric_flag_exists_and_is_bounded_to_recorded_errors():
    flag = next(c for c in RUBRIC["critical_errors"] if c["id"] == "predictor_error_uncaught")
    described = flag["description"].lower()
    # The flag must not be raisable on a control, or it measures hedging.
    assert "none" in described and "never" in described
    assert "doubting a correct prediction is not this flag" in described


def test_a_case_with_a_source_free_reference_claims_no_outcome():
    """Where the assay ground truth could not be verified, the case must ask
    for reasoning rather than assert what is true."""
    for case in CASES:
        if not case["reference"]["sources"]:
            assert case["reference"]["type"] == "reasoning_expectation", case["case_id"]


# ------------------------------------------------------- the keyword scan

def test_the_scan_covers_every_case_in_the_set():
    from evals.investigation import signal_scan

    assert set(signal_scan.SIGNALS) == {c["case_id"] for c in CASES}


def test_the_scan_never_reads_the_prompt_as_an_answer():
    """The prompt contains the question and the ToxPred snapshot. Scanning it
    would report the case's own words back as if the system had said them."""
    from evals.investigation import signal_scan

    record = {"prompt": "torsades de pointes positive control", "result": "a short answer"}
    text = signal_scan.answer_text(record)
    assert "torsades" not in text
    assert "a short answer" in text


def test_the_scan_separates_noticing_from_passing_through():
    from evals.investigation import signal_scan

    signals = signal_scan.SIGNALS["inv-13-moxifloxacin-undetected"]
    assert any("positive control" in p for p in signals["notices"])
    assert any("reasonable" in p for p in signals["passes_through"])


# ------------------------------------------------- what the graders receive

def test_the_grader_reference_states_the_predictor_error():
    """Otherwise `predictor_error_uncaught` has nothing to stand on: a grader
    cannot tell a case built around a predictor mistake from one where the
    predictor was right."""
    from evals.investigation import packet as packet_module

    case = next(c for c in CASES
                if c["reference"]["predictor_error"]["kind"] != "none")
    rendered = packet_module._reference_markdown(case)
    assert case["reference"]["predictor_error"]["what_the_predictor_says"] in rendered
    assert "predictor_error_uncaught" in rendered


def test_the_grader_reference_marks_a_control_as_out_of_scope_for_the_flag():
    from evals.investigation import packet as packet_module

    control = next(c for c in CASES
                   if c["reference"]["predictor_error"]["kind"] == "none")
    rendered = packet_module._reference_markdown(control)
    assert "control case" in rendered
    assert "must NOT be raised" in rendered


def test_a_pilot_case_without_a_predictor_error_renders_unchanged():
    """The pilot set has no predictor_error block; its packets must not gain
    a heading with nothing under it."""
    from evals.investigation import cases as case_module
    from evals.investigation import packet as packet_module

    pilot = case_module.load_cases()[0]
    assert "predictor_error" not in pilot["reference"]
    assert "The predictor's own output" not in packet_module._reference_markdown(pilot)

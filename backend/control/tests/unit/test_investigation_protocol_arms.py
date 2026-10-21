"""#50: the four arms of protocol v2, and the fairness contract between them.

The completion condition: the four arms run from one case file under one
manifest, and the diff between two arm configs contains exactly the variable
under study. These tests hold the configs to that, and the runner's check to
refusing arms whose deployments differ in anything they must share.
"""
from __future__ import annotations

from evals.investigation.systems import (
    PROTOCOL_V2_DIR,
    SYSTEMS,
    arm_config,
    arm_diff,
    fairness_problems,
    product_mismatch,
    protocol_arms,
)


def _arm(letter: str) -> dict:
    return arm_config(PROTOCOL_V2_DIR / f"{letter}.json")


def test_b_differs_from_c_only_in_running_a_fixed_workflow():
    assert arm_diff(_arm("B"), _arm("C")) == {"required_flags.fixed_workflow_v1": (True, False)}


def test_c_differs_from_d_only_in_the_case_and_the_skills():
    assert arm_diff(_arm("C"), _arm("D")) == {
        "required_flags.scientific_case_v1": (False, True),
        "required_flags.scientific_skills_v1": (False, True),
        "skills_mode": ("off", "dynamic"),
    }


def test_the_agent_arms_share_the_gates_the_budget_and_the_sources():
    for letter in "BCD":
        arm = _arm(letter)
        flags = arm["required_flags"]
        assert flags["answer_draft_v2"] and flags["release_gate_v1"] and flags["claim_reviewer_v1"]
        assert flags["evidence_full_text_v1"] and flags["activity_replay_v1"]
        assert arm["budget_preset"] == "investigator-pilot-v1"
        assert "runtime.model_id" in arm["same_across_arms"]


def test_the_v2_arms_are_registered_beside_the_historical_ones():
    v2 = protocol_arms()
    assert set(v2) == {"A2_template", "B2_fixed_workflow", "C2_generic_react", "D2_investigator"}
    assert set(v2) <= set(SYSTEMS)
    # the v1 arms stay, as diagnostics of the code as it was
    assert SYSTEMS["C_toxagent_current"].protocol == "v1"
    assert SYSTEMS["A2_template"].adapter == "predictor_template"


def _product(model: str = "gpt-x", preset: str = "investigator-pilot-v1", **flags) -> dict:
    return {
        "runtime": {"provider_id": "openai", "model_id": model},
        "intents": {"decision_support": {"budget": {"preset": preset}}},
        "providers": {"research_provider": "snapshot", "activity_replay": {"sha256": "abc"}},
        "flags": {name: {"enabled": value} for name, value in flags.items()},
    }


def test_arms_on_different_backbones_are_refused():
    specs = [SYSTEMS["C2_generic_react"], SYSTEMS["D2_investigator"]]
    same = {s.system_id: _product() for s in specs}
    assert fairness_problems(specs, same) == []
    different = {"C2_generic_react": _product(model="gpt-x"),
                 "D2_investigator": _product(model="gpt-y")}
    (problem,) = fairness_problems(specs, different)
    assert "runtime.model_id" in problem


def test_a_deployment_on_another_budget_is_not_the_arm():
    spec = SYSTEMS["D2_investigator"]
    flags = dict(spec.required_flags)
    assert product_mismatch(spec, {**_product(**flags),
                                   "scientific_skills": {"mode": "dynamic"}}) == []
    problems = product_mismatch(spec, {**_product(preset=None, **flags),
                                       "scientific_skills": {"mode": "dynamic"}})
    assert problems == ["budget preset is None, the arm needs investigator-pilot-v1"]

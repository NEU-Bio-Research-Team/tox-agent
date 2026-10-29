"""§11 turns 2 and 3 through the product: a margin, then a corrected exposure.

The unit tests prove each piece -- versioned researcher values (#61), the
dependency graph (#60), the procedures (#66), the case view (#63). This drives
them together through the API and the scripted runtime, which is the only way
to show the pieces meet:

1. the researcher files an IC50 and a free Cmax into the case;
2. the server computes the margin and records it as a derived claim over the
   two *versions* it was computed from;
3. the researcher corrects the exposure, and the case marks that claim, and
   only that claim, as needing reassessment;
4. the next turn sees the change in its prompt, ``check_dependencies`` names
   the stale claim with no unresolved reference, and a margin over the new
   version is a new claim beside the old one rather than a rewrite of it.
"""
from __future__ import annotations

import pytest

from tests.e2e.test_scientific_case_e2e import _post
from tests.e2e.test_scripted_runtime import _analyse, _install_scripted_runtime, _new_session
from tests.support.api import AUTH, api_client
from tests.support.predictor import StubPredictor

pytestmark = pytest.mark.anyio


@pytest.fixture
def flags_on(monkeypatch):
    for flag in ("ANSWER_DRAFT_V2", "SCIENTIFIC_CASE_V1", "SCIENTIFIC_PRIMITIVES_V1",
                 "SCIENTIFIC_PROCEDURES_V1", "INVESTIGATOR_CONTRACT_V1"):
        monkeypatch.setenv(f"TOXAGENT_FLAG_{flag}", "1")


async def _case(client, session_id: str) -> dict:
    listed = (await client.get(f"/v1/sessions/{session_id}/cases", headers=AUTH)).json()
    return listed["cases"][0]


async def _case_detail(client, session_id: str, case_id: str) -> dict:
    found = await client.get(f"/v1/sessions/{session_id}/cases/{case_id}", headers=AUTH)
    assert found.status_code == 200, found.text
    return found.json()


async def _add(client, session_id: str, case_id: str, key: str, value: str) -> None:
    added = await client.post(f"/v1/sessions/{session_id}/cases/{case_id}/context",
                              json={"key": key, "value": value}, headers=AUTH)
    assert added.status_code == 200, added.text


def _margin_answer(observation_id: str) -> dict:
    return {
        "answer_markdown": "The exposure margin (IC50 / free Cmax) is {{margin}}-fold.",
        "claims": [{"local_ref": "margin", "kind": "numeric",
                    "text": "Exposure margin computed from the researcher's IC50 and free Cmax.",
                    "observation_id": observation_id, "field_path": "margin"}],
        "limitations": [{"code": "screening_not_safety_assessment", "text": ""}],
    }


async def test_a_corrected_exposure_marks_the_margin_it_fed_and_the_next_turn_sees_it(
    db, flags_on,
):
    state: dict = {}

    async def script(turn) -> None:
        message = turn.user_message
        if "margin" not in message:
            await turn.call_tool("submit_grounded_answer", {
                "answer_markdown": "Noted.", "claims": [], "limitations": []})
            return
        if "again" in message:
            state["prompt"] = turn.system_prompt
            state["check"] = await turn.call_tool(
                "run_scientific_procedure", {"procedure": "check_dependencies"})
            cmax_ref = "context:c3"
        else:
            cmax_ref = "context:c2"
        result = await turn.call_tool("compute_exposure_margin", {
            "ic50": {"value": 300, "unit": "nM", "source_ref": "context:c1"},
            "cmax": {"value": 20 if cmax_ref == "context:c3" else 10, "unit": "nM",
                     "source_ref": cmax_ref},
        })
        state.setdefault("margins", []).append(result)
        await turn.call_tool("submit_grounded_answer",
                             _margin_answer(result["observation_ids"][0]))

    async with api_client(db, StubPredictor()) as client:
        await _install_scripted_runtime(client, script)
        session_id = await _new_session(client)
        await _analyse(client, session_id)
        await _post(client, session_id, "Is hERG a concern?")
        case_id = (await _case(client, session_id))["case_id"]
        await _add(client, session_id, case_id, "patch_clamp_ic50", "300 nM, in-house HEK293")
        await _add(client, session_id, case_id, "free_cmax", "10 nM at the intended dose")

        # Turn 2: the margin, recorded as a claim over the two versions.
        run = await _post(client, session_id, "What is the exposure margin?")
        assert run["status"] == "completed", run
        first = state["margins"][0]
        assert first["status"] == "completed", first
        assert first["model_view"]["margin"] == 30.0
        assert first["model_view"]["derived_claim"] == "dc1"
        case = await _case_detail(client, session_id, case_id)
        (claim,) = case["derived_claims"]
        assert claim["status"] == "proposed"
        assert claim["justifications"][0]["inputs"] == [
            "case_value:patch_clamp_ic50@v1", "case_value:free_cmax@v1"]

        # Turn 3: the researcher corrects the exposure.
        await _add(client, session_id, case_id, "free_cmax", "20 nM at the intended dose")
        case = await _case_detail(client, session_id, case_id)
        (claim,) = case["derived_claims"]
        assert claim["status"] == "needs_reassessment"
        assert claim["justifications"][0]["invalidated_by"] == ["case_value:free_cmax@v1"]
        # Kept, not deleted: the earlier answer was computed from it.
        assert "30" in claim["statement"]

        run = await _post(client, session_id, "Compute the margin again with the new exposure.")
        assert run["status"] == "completed", run

    # The next turn is told what changed, and the procedure agrees with the case.
    assert "dc1" in state["prompt"]
    assert "needs_reassessment" in state["prompt"] or "lost a premise" in state["prompt"]
    check = state["check"]
    assert check["status"] == "completed", check
    facts = check["model_view"]["facts"]
    assert [row["claim_id"] for row in facts["needs_reassessment"]] == ["dc1"]
    assert facts["unresolved_references"] == []

    second = state["margins"][1]
    assert second["model_view"]["margin"] == 15.0
    assert second["model_view"]["derived_claim"] == "dc2"


async def test_with_no_case_a_margin_over_researcher_values_is_refused_by_name(db, monkeypatch):
    """With cases off there are no researcher values to read, so the refusal
    names the missing source rather than computing from a number nobody filed."""
    for flag in ("ANSWER_DRAFT_V2", "SCIENTIFIC_PRIMITIVES_V1"):
        monkeypatch.setenv(f"TOXAGENT_FLAG_{flag}", "1")
    state: dict = {}

    async def script(turn) -> None:
        state["margin"] = await turn.call_tool("compute_exposure_margin", {
            "ic50": {"value": 300, "unit": "nM", "source_ref": "context:c1"},
            "cmax": {"value": 10, "unit": "nM", "source_ref": "context:c2"},
        })
        await turn.call_tool("submit_grounded_answer", {
            "answer_markdown": "No case.", "claims": [], "limitations": []})

    async with api_client(db, StubPredictor()) as client:
        await _install_scripted_runtime(client, script)
        session_id = await _new_session(client)
        await _analyse(client, session_id)
        await _post(client, session_id, "What is the exposure margin?")

    assert state["margin"]["status"] == "error"
    assert "no such context item" in state["margin"]["error"]["message"]


async def test_an_answer_quotes_the_researchers_value_by_the_observation_it_was_given(
    db, flags_on,
):
    """Live run 2026-10-22, T5 0/2 for every arm: a value the researcher filed
    into the case could not be cited, because a claim cites an observation and
    a context item was not one. The view now hands the turn the observation
    each value in force was projected into, and a superseded version none."""
    import re

    state: dict = {}

    async def script(turn) -> None:
        if "exposure" not in turn.user_message:
            await turn.call_tool("submit_grounded_answer", {
                "answer_markdown": "Noted.", "claims": [], "limitations": []})
            return
        state["prompt"] = turn.system_prompt
        found = re.search(r"case_value:free_cmax@v2; cite observation_id (obs_[0-9a-f]{32}) "
                          r"field_path (\w+)", turn.system_prompt)
        state["found"] = found
        state["answer"] = await turn.call_tool("submit_grounded_answer", {
            "answer_markdown": "The free Cmax you supplied is on record, as you stated it.",
            "claims": [{"local_ref": "cmax", "kind": "scientific",
                        "text": "The researcher supplied a free Cmax for this compound.",
                        "observation_id": found.group(1), "field_path": found.group(2)}],
            "limitations": [{"code": "user_supplied_value_not_verified", "text": ""}],
        })

    async with api_client(db, StubPredictor()) as client:
        await _install_scripted_runtime(client, script)
        session_id = await _new_session(client)
        await _analyse(client, session_id)
        await _post(client, session_id, "Is hERG a concern?")
        case_id = (await _case(client, session_id))["case_id"]
        await _add(client, session_id, case_id, "free_cmax", "10 nM at the intended dose")
        await _add(client, session_id, case_id, "free_cmax", "20 nM at the intended dose")
        run = await _post(client, session_id, "What exposure are we working with?")

    assert run["status"] == "completed", run
    assert state["found"] is not None, state["prompt"]
    assert "case_value:free_cmax@v1, superseded" in state["prompt"]
    answer = state["answer"]
    assert answer["status"] == "completed", answer
    assert answer["model_view"]["is_fallback"] is False

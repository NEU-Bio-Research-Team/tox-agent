"""A computed exposure margin is citable through the product API."""
from __future__ import annotations

import pytest

from tests.e2e.test_scientific_case_e2e import _post
from tests.e2e.test_scripted_runtime import _analyse, _install_scripted_runtime, _new_session
from tests.support.api import AUTH, api_client
from tests.support.predictor import StubPredictor

pytestmark = pytest.mark.anyio


@pytest.fixture
def flags_on(monkeypatch):
    for flag in ("ANSWER_DRAFT_V2", "SCIENTIFIC_CASE_V1", "SCIENTIFIC_PRIMITIVES_V1"):
        monkeypatch.setenv(f"TOXAGENT_FLAG_{flag}", "1")


async def _case_id(client, session_id: str) -> str:
    return (await client.get(f"/v1/sessions/{session_id}/cases", headers=AUTH)).json()["cases"][0]["case_id"]


async def test_the_model_cites_a_margin_the_server_computed_from_the_researchers_data(db, flags_on):
    state: dict = {}

    async def script(turn) -> None:
        if "margin" not in turn.user_message:
            await turn.call_tool("submit_grounded_answer", {
                "answer_markdown": "Noted.", "claims": [], "limitations": []})
            return
        state["wrong"] = await turn.call_tool("compute_exposure_margin", {
            "ic50": {"value": 3, "unit": "µM", "source_ref": "context:c1"},
            "cmax": {"value": 50, "unit": "nM", "source_ref": "context:c2"},
        })
        result = await turn.call_tool("compute_exposure_margin", {
            "ic50": {"value": 30, "unit": "µM", "source_ref": "context:c1"},
            "cmax": {"value": 50, "unit": "nM", "source_ref": "context:c2"},
        })
        state["margin"] = result
        observation_id = result["observation_ids"][0]
        state["answer"] = await turn.call_tool("submit_grounded_answer", {
            "answer_markdown": "The exposure margin (IC50 / free Cmax) is {{margin}}-fold.",
            "claims": [{"local_ref": "margin", "kind": "numeric",
                        "text": "Exposure margin computed from the in-house IC50 and free Cmax.",
                        "observation_id": observation_id, "field_path": "margin"}],
            "limitations": [{"code": "screening_not_safety_assessment", "text": ""}],
        })

    async with api_client(db, StubPredictor()) as client:
        await _install_scripted_runtime(client, script)
        session_id = await _new_session(client)
        await _analyse(client, session_id)
        await _post(client, session_id, "Is hERG a concern?")
        case_id = await _case_id(client, session_id)
        for key, value in (("patch_clamp_ic50", "30 µM, in-house HEK293"),
                           ("free_cmax", "50 nM at the intended dose")):
            added = await client.post(f"/v1/sessions/{session_id}/cases/{case_id}/context",
                                      json={"key": key, "value": value}, headers=AUTH)
            assert added.status_code == 200, added.text

        run = await _post(client, session_id, "What is the exposure margin?")
        assert run["status"] == "completed", run
        assert state["wrong"]["status"] == "error"
        assert "not written in context:c1" in state["wrong"]["error"]["message"]
        assert state["margin"]["model_view"]["margin"] == 600.0
        assert state["answer"]["status"] == "completed", state["answer"]
        answer = state["answer"]["model_view"]
        assert answer["is_fallback"] is False
        messages = (await client.get(f"/v1/sessions/{session_id}/messages", headers=AUTH)).json()
        assert "600-fold" in str(messages["messages"][-1])



async def test_with_the_flag_off_neither_primitive_exists(db, monkeypatch):
    monkeypatch.delenv("TOXAGENT_FLAG_SCIENTIFIC_PRIMITIVES_V1", raising=False)
    async with api_client(db, StubPredictor(), pubchem_assay_provider=object()) as client:
        registry = client.app.state.tool_registry
        assert registry.get("compute_exposure_margin") is None
        assert registry.get("get_pubchem_bioassays") is None
        assert registry.get("get_chembl_activities") is None


async def test_retired_chembl_tool_cannot_be_reenabled_by_legacy_settings(db, monkeypatch):
    monkeypatch.setenv("TOXAGENT_FLAG_SCIENTIFIC_PRIMITIVES_V1", "1")
    monkeypatch.setenv("TOXAGENT_CHEMBL_PROVIDER", "chembl")
    async with api_client(db, StubPredictor(), pubchem_assay_provider=object()) as client:
        registry = client.app.state.tool_registry
        assert registry.get("get_chembl_activities") is None
        assert registry.get("get_pubchem_bioassays") is not None

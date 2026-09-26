"""W9-13 end to end: a computed margin and a ChEMBL value, both citable.

Driven through the product API and the scripted runtime. The researcher files
two measurements into the case; the model asks the server for the margin, then
cites the margin observation in a numeric claim the validator checks. A ChEMBL
activity becomes an evidence record the answer cites directly.
"""
from __future__ import annotations

from datetime import date

import pytest

from toxagent.domain.evidence import SourceIdentifier, SourceType
from toxagent.research.interfaces import SearchHit
from toxagent.research.providers.chembl import ActivityLookup
from tests.e2e.test_scientific_case_e2e import _post
from tests.e2e.test_scripted_runtime import _analyse, _install_scripted_runtime, _new_session
from tests.support.api import AUTH, api_client
from tests.support.predictor import StubPredictor

pytestmark = pytest.mark.anyio


class StubChembl:
    name = "chembl"
    allowed_hosts = ("www.ebi.ac.uk",)

    def __init__(self) -> None:
        self.calls = 0

    async def activities(self, *, canonical_smiles: str, target: str, limit: int) -> ActivityLookup:
        self.calls += 1
        return ActivityLookup(molecule_chembl_id="CHEMBL25", target_chembl_id="CHEMBL240", hits=(
            SearchHit(
                provider_record_id="activity:101", source_type=SourceType.DATABASE,
                title="ASPIRIN (CHEMBL25) against hERG: IC50 > 100000 nM",
                published_at=date(2010, 1, 1),
                canonical_url="https://www.ebi.ac.uk/chembl/compound_report_card/CHEMBL25/",
                identifier=SourceIdentifier(other="chembl_activity:101"),
                abstract_or_excerpt="Measured IC50 > 100000 nM in a hERG binding assay.",
                normalized_facts={"standard_type": "IC50", "standard_value": "100000",
                                  "standard_units": "nM"},
            ),
        ))


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


async def test_a_chembl_activity_is_citable_and_respects_the_data_scope(db, flags_on):
    state: dict = {}
    chembl = StubChembl()

    async def script(turn) -> None:
        found = await turn.call_tool("get_chembl_activities", {
            "analysis_id": state["analysis_id"], "target": "herg",
        })
        state.setdefault("lookups", []).append(found)
        if found["status"] != "completed":
            await turn.call_tool("submit_grounded_answer", {
                "answer_markdown": "ChEMBL was not queried.", "claims": [], "limitations": []})
            return
        evidence_id = found["model_view"]["records"][0]["evidence_id"]
        state["answer"] = await turn.call_tool("submit_grounded_answer", {
            "answer_markdown": "ChEMBL reports a weak measured hERG IC50 for this structure.",
            "claims": [{"local_ref": "chembl", "kind": "scientific",
                        "text": "ChEMBL reports a weak measured hERG IC50 for this structure.",
                        "citation_ids": [evidence_id]}],
            "limitations": [{"code": "evidence_scope_limited", "text": ""}],
        })

    async with api_client(db, StubPredictor(), chembl_provider=chembl) as client:
        await _install_scripted_runtime(client, script)
        session_id = await _new_session(client)
        state["analysis_id"] = await _analyse(client, session_id)
        run = await _post(client, session_id, "Is hERG a concern?")
        assert run["status"] == "completed", run
        assert state["lookups"][0]["model_view"]["molecule_chembl_id"] == "CHEMBL25"
        # Cited without a separate get_evidence_record: the tool returned it whole.
        assert state["answer"]["status"] == "completed", state["answer"]
        assert state["answer"]["model_view"]["is_fallback"] is False

        case_id = await _case_id(client, session_id)
        await client.post(f"/v1/sessions/{session_id}/cases/{case_id}/scope",
                          json={"external_search": False, "reason": "unpublished structure"},
                          headers=AUTH)
        await _post(client, session_id, "And again?")
        assert state["lookups"][1]["error"]["code"] == "tool_denied"
        assert chembl.calls == 1


async def test_with_the_flag_off_neither_primitive_exists(db, monkeypatch):
    monkeypatch.delenv("TOXAGENT_FLAG_SCIENTIFIC_PRIMITIVES_V1", raising=False)
    async with api_client(db, StubPredictor(), chembl_provider=StubChembl()) as client:
        registry = client.app.state.tool_registry
        assert registry.get("compute_exposure_margin") is None
        assert registry.get("get_chembl_activities") is None

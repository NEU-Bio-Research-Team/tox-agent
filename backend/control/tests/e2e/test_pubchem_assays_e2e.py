"""PubChem assays are citable and honour a researcher's external-data scope."""
from __future__ import annotations

import pytest

from toxagent.research.providers.pubchem_assays import AssayLookup, PubChemAssayProvider
from tests.e2e.test_scientific_case_e2e import _post
from tests.e2e.test_scripted_runtime import _analyse, _install_scripted_runtime, _new_session
from tests.support.api import AUTH, api_client
from tests.support.predictor import StubPredictor

pytestmark = pytest.mark.anyio


class StubAssays:
    name = "pubchem_bioassay"
    allowed_hosts = ("pubchem.ncbi.nlm.nih.gov",)

    def __init__(self):
        self.calls = 0

    async def assays(self, *, canonical_smiles, target, limit):
        self.calls += 1
        hit = PubChemAssayProvider._hit({
            "Assay Name": "hERG inhibition assay", "Activity Outcome": "Active",
            "Activity Name": "IC50", "Activity Value [uM]": "0.204",
            "Target GeneID": "3757", "Target Accession": "Q12809",
            "Assay Type": "Confirmatory", "PubMed ID": "12190308",
        }, cid=5405, aid="82355", sid="103181781")
        return AssayLookup(5405, (hit,), 1)


async def test_pubchem_record_is_citable_and_denied_before_external_lookup(db, monkeypatch):
    for flag in ("ANSWER_DRAFT_V2", "SCIENTIFIC_CASE_V1", "SCIENTIFIC_PRIMITIVES_V1", "TRUST_ENVELOPE_V1"):
        monkeypatch.setenv(f"TOXAGENT_FLAG_{flag}", "1")
    provider = StubAssays()
    state = {"results": []}

    async def script(turn):
        result = await turn.call_tool("get_pubchem_bioassays", {"analysis_id": state["analysis_id"]})
        state["results"].append(result)
        claims = []
        if result["status"] == "completed":
            evidence = result["model_view"]["records"][0]["evidence_id"]
            claims = [{"local_ref": "assay", "kind": "scientific",
                       "text": "PubChem reports an hERG assay.", "citation_ids": [evidence]}]
        state["answer"] = await turn.call_tool("submit_grounded_answer", {
            "answer_markdown": "An assay was reported." if claims else "No external lookup was made.",
            "claims": claims, "limitations": [{"code": "evidence_scope_limited", "text": ""}],
        })

    async with api_client(db, StubPredictor(), pubchem_assay_provider=provider) as client:
        await _install_scripted_runtime(client, script)
        session_id = await _new_session(client)
        state["analysis_id"] = await _analyse(client, session_id)
        run = await _post(client, session_id, "Find hERG assay evidence")
        assert run["status"] == "completed", run
        first = state["results"][0]
        assert first["status"] == "completed" and first["model_view"]["total_matching_assays"] == 1
        assert first["model_view"]["records"][0]["trust_envelope"]
        assert state["answer"]["status"] == "completed" and not state["answer"]["model_view"]["is_fallback"]
        case_id = (await client.get(f"/v1/sessions/{session_id}/cases", headers=AUTH)).json()["cases"][0]["case_id"]
        changed = await client.post(f"/v1/sessions/{session_id}/cases/{case_id}/scope", headers=AUTH,
                                    json={"external_search": False, "reason": "private structure"})
        assert changed.status_code == 200
        await _post(client, session_id, "Look again")
        assert state["results"][1]["status"] == "error"
        assert state["results"][1]["error"]["code"] == "tool_denied"
        assert provider.calls == 1

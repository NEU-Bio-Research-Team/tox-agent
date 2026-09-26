"""ADR 0012 end to end: a case outlives the turn.

Turn 1 frames the case, grounds it in a retrieved record, and ends with an
accepted answer whose relations join the ledger; the run compiles a dossier.
The researcher then adds an in-house result through the API, and turn 2 sees
it in its prompt, cites it, and weakens the hypothesis — in the same case. A
different compound opens a different case. With the flag off, nothing of this
exists.

Driven through the product API with the scripted runtime, the same seam an
OpenCode adapter uses.
"""
from __future__ import annotations

import pytest

from tests.e2e.test_scripted_runtime import _analyse, _install_scripted_runtime, _new_session
from tests.support.api import AUTH, api_client, wait_for_run
from tests.support.predictor import StubPredictor
from tests.support.research import ACCEPTED_HIT, StubResearchProvider

pytestmark = pytest.mark.anyio

BENZENE = "c1ccccc1"
H1 = "The compound blocks hERG at relevant exposure"


@pytest.fixture
def flags_on(monkeypatch):
    monkeypatch.setenv("TOXAGENT_FLAG_ANSWER_DRAFT_V2", "1")
    monkeypatch.setenv("TOXAGENT_FLAG_SCIENTIFIC_CASE_V1", "1")


async def _post(client, session_id: str, text: str) -> dict:
    submitted = await client.post(
        f"/v1/sessions/{session_id}/messages",
        json={"intent_hint": "auto", "content": [{"type": "text", "text": text}]},
        headers=AUTH,
    )
    assert submitted.status_code == 202, submitted.text
    return await wait_for_run(client, session_id, submitted.json()["run_id"])


def _answer(evidence_id: str | None, text: str, relations: list[dict] | None = None) -> dict:
    claims = []
    if evidence_id:
        claims = [{"local_ref": "lit", "kind": "scientific", "text": text,
                   "citation_ids": [evidence_id]}]
    return {
        "schema_version": "grounded-answer-v2",
        "answer_markdown": text,
        "claims": claims,
        "limitations": [{"code": "evidence_scope_limited", "text": ""}] if evidence_id else [],
        "evidence_relations": relations or [],
    }


async def test_a_case_carries_an_investigation_across_turns(db, flags_on):
    state: dict = {"analysis_id": "", "prompts": [], "results": []}

    async def first_turn(turn) -> None:
        read = await turn.call_tool("get_scientific_case", {})
        state["results"].append(read)
        found = await turn.call_tool("search_toxicology_evidence", {
            "analysis_id": state["analysis_id"], "query": "hERG blockade", "limit": 5,
        })
        evidence_id = found["model_view"]["results"][0]["evidence_id"]
        await turn.call_tool("get_evidence_record", {"evidence_id": evidence_id})
        state["results"].append(await turn.call_tool("update_scientific_case", {"operations": [
            {"op": "add_hypothesis", "statement": H1, "hypothesis_kind": "mechanism",
             "refutation_condition": "An IC50 far above free Cmax"},
            {"op": "add_hypothesis", "statement": "The signal is an assay artefact",
             "hypothesis_kind": "assay_or_exposure",
             "refutation_condition": "Patch clamp agrees with the binding assay"},
            {"op": "record_evidence", "claim": "A related series blocks hERG",
             "source_class": "external_experimental", "source_ref": f"evidence:{evidence_id}",
             "stance": "supports", "directness": "analogue", "hypothesis_ids": ["h1"]},
            {"op": "record_action", "action": "counterevidence_search",
             "purpose": "look for data showing no block", "decision": "answer",
             "hypothesis_ids": ["h1"], "outcome": "nothing against h1 in the retrieved set"},
            {"op": "set_conclusion",
             "can_say": [{"text": "An analogue series blocks hERG", "evidence_ids": ["e1"]}],
             "cannot_say": ["whether this compound blocks hERG at its exposure"],
             "what_would_change": ["an in-house patch-clamp IC50"]},
        ]}))
        state["results"].append(await turn.call_tool("submit_grounded_answer", _answer(
            evidence_id, "A retrieved study reports hERG blockade in a related series.",
            relations=[{"proposition": H1, "source_class": "external_experimental",
                        "source_id": evidence_id, "relation": "supports",
                        "reason_codes": ["related_series"], "directness": "indirect"}],
        )))

    async def second_turn(turn) -> None:
        state["results"].append(await turn.call_tool("update_scientific_case", {"operations": [
            {"op": "record_evidence", "claim": "In-house patch clamp IC50 is 30 µM",
             "source_class": "user_supplied", "source_ref": "context:c1",
             "stance": "contradicts", "directness": "direct", "hypothesis_ids": ["h1"]},
            {"op": "revise_hypothesis", "hypothesis_id": "h1", "status": "weakened",
             "reason": "direct in-house data shows weak block"},
        ]}))
        state["results"].append(await turn.call_tool(
            "submit_grounded_answer",
            _answer(None, "The in-house result weakens the concern; see the case."),
        ))

    async def script(turn) -> None:
        state["prompts"].append(turn.system_prompt)
        if "in-house" in turn.user_message:
            await second_turn(turn)
        elif "hERG" in turn.user_message:
            await first_turn(turn)
        else:
            await turn.call_tool("submit_grounded_answer", _answer(None, "Noted."))

    provider = StubResearchProvider(hits=[ACCEPTED_HIT])
    async with api_client(db, StubPredictor(), research_provider=provider) as client:
        await _install_scripted_runtime(client, script)
        session_id = await _new_session(client)
        state["analysis_id"] = await _analyse(client, session_id)

        first = await _post(client, session_id, "Should hERG stop us developing this compound?")
        assert first["status"] == "completed", first
        assert all(r["status"] == "completed" for r in state["results"]), state["results"]
        assert "scientific case that persists" in state["prompts"][0]
        assert "Decision question: Should hERG stop us" in state["prompts"][0]

        cases = (await client.get(f"/v1/sessions/{session_id}/cases", headers=AUTH)).json()["cases"]
        assert len(cases) == 1
        case_id = cases[0]["case_id"]
        case = (await client.get(f"/v1/sessions/{session_id}/cases/{case_id}", headers=AUTH)).json()
        assert [h["id"] for h in case["hypotheses"]] == ["h1", "h2"]
        # The model's entry and the accepted answer's relation, as its own server entry.
        assert [(e["actor"], e["stance"]) for e in case["evidence"]] == [
            ("model", "supports"), ("server", "supports"),
        ]
        assert case["runs"][0]["stop_reason"] == "sufficient"

        dossier = await client.get(
            f"/v1/sessions/{session_id}/runs/{first['run_id']}/dossier", headers=AUTH
        )
        assert dossier.status_code == 200, dossier.text
        dossier = dossier.json()
        assert dossier["schema_version"] == "decision-dossier-v1"
        assert dossier["stop_reason"] == "sufficient"
        assert dossier["answer_id"]
        assert [e["id"] for e in dossier["hypotheses"][0]["evidence_for"]] == ["e1", "e2"]
        assert dossier["explanation_layers"]["agent_decisions"][0]["action"] == "counterevidence_search"
        assert dossier["conclusion"]["can_say"][0]["sources"][0].startswith("evidence:")

        # The researcher answers what would change the conclusion.
        added = await client.post(
            f"/v1/sessions/{session_id}/cases/{case_id}/context",
            json={"key": "patch_clamp_ic50", "value": "30 µM, in-house, HEK293"}, headers=AUTH,
        )
        assert added.status_code == 200, added.text
        assert added.json()["context"][0]["id"] == "c1"

        second = await _post(client, session_id, "Here is our in-house patch clamp result.")
        assert second["status"] == "completed", second
        assert "context c1 (user): patch_clamp_ic50 = 30 µM" in state["prompts"][-1]
        assert f"Open scientific case {case_id}" in state["prompts"][-1]
        case = (await client.get(f"/v1/sessions/{session_id}/cases/{case_id}", headers=AUTH)).json()
        assert case["hypotheses"][0]["status"] == "weakened"
        assert [r["run_id"] for r in case["runs"]] == [first["run_id"], second["run_id"]]
        latest = (await client.get(
            f"/v1/sessions/{session_id}/cases/{case_id}/dossier", headers=AUTH
        )).json()
        assert latest["run_id"] == second["run_id"]
        assert latest["hypotheses"][0]["evidence_against"][0]["source_ref"] == "context:c1"

        events = (await client.get(
            f"/v1/sessions/{session_id}/cases/{case_id}/events", headers=AUTH
        )).json()["events"]
        assert [e["revision"] for e in events] == list(range(1, len(events) + 1))
        assert events[0]["op"] == "open" and events[-1]["op"] == "finish_run"
        assert {e["actor"] for e in events} == {"server", "model", "user"}

        # Another compound is another investigation.
        await _analyse(client, session_id, BENZENE)
        await _post(client, session_id, "Anything to note?")
        cases = (await client.get(f"/v1/sessions/{session_id}/cases", headers=AUTH)).json()["cases"]
        assert len(cases) == 2
        assert "context c1" not in state["prompts"][-1]


async def test_with_the_flag_off_there_is_no_case_anywhere(db, monkeypatch):
    monkeypatch.setenv("TOXAGENT_FLAG_ANSWER_DRAFT_V2", "1")
    seen: dict = {}

    async def script(turn) -> None:
        seen["prompt"] = turn.system_prompt
        seen["read"] = await turn.call_tool("get_scientific_case", {})
        await turn.call_tool("submit_grounded_answer", _answer(None, "Noted."))

    async with api_client(db, StubPredictor()) as client:
        await _install_scripted_runtime(client, script)
        session_id = await _new_session(client)
        await _analyse(client, session_id)
        run = await _post(client, session_id, "Should hERG stop us?")
        assert run["status"] == "completed"
        assert seen["read"]["error"]["code"] == "tool_denied"
        assert "scientific case" not in seen["prompt"]
        cases = await client.get(f"/v1/sessions/{session_id}/cases", headers=AUTH)
        assert cases.json() == {"cases": []}
        dossier = await client.get(
            f"/v1/sessions/{session_id}/runs/{run['run_id']}/dossier", headers=AUTH
        )
        assert dossier.status_code == 404


async def test_the_case_ledger_fills_from_a_v1_answer_without_answer_draft_v2(db, monkeypatch):
    """W9-04: grounded-answer v1 carries no relations. Before this, a case kept
    without answer_draft_v2 recorded nothing the answer relied on; now what the
    answer cited joins the ledger as unlinked context from the server."""
    monkeypatch.setenv("TOXAGENT_FLAG_SCIENTIFIC_CASE_V1", "1")
    monkeypatch.delenv("TOXAGENT_FLAG_ANSWER_DRAFT_V2", raising=False)
    state: dict = {"analysis_id": ""}

    async def script(turn) -> None:
        found = await turn.call_tool("search_toxicology_evidence", {
            "analysis_id": state["analysis_id"], "query": "hERG blockade", "limit": 5,
        })
        evidence_id = found["model_view"]["results"][0]["evidence_id"]
        await turn.call_tool("get_evidence_record", {"evidence_id": evidence_id})
        slice_result = await turn.call_tool("get_analysis_slice", {
            "analysis_id": state["analysis_id"], "section": "herg",
            "fields": ["probability_blocker"],
        })
        value = slice_result["model_view"]["values"]["probability_blocker"]
        state["observation_id"] = value["observation_id"]
        state["evidence_id"] = evidence_id
        state["answer"] = await turn.call_tool("submit_grounded_answer", {
            "schema_version": "grounded-answer-v1",
            "answer_markdown": "The predicted hERG blocker probability is 0.731; "
                               "a retrieved study reports block in a related series.",
            "claims": [
                {"claim_id": "clm_" + "3" * 32, "kind": "numeric",
                 "text": "The predicted hERG blocker probability is 0.731.",
                 "observation_id": value["observation_id"], "field_path": value["field_path"],
                 "source_value": value["value"], "rendered_value": "0.731",
                 "transform": "round:3"},
                {"claim_id": "clm_" + "4" * 32, "kind": "scientific",
                 "text": "A retrieved study reports hERG block in a related series.",
                 "citation_ids": [evidence_id]},
            ],
            "limitations": [{"code": "uncalibrated_probability", "text": ""},
                            {"code": "evidence_scope_limited", "text": ""}],
            "recommended_next_steps": [],
        })

    provider = StubResearchProvider(hits=[ACCEPTED_HIT])
    async with api_client(db, StubPredictor(), research_provider=provider) as client:
        await _install_scripted_runtime(client, script)
        session_id = await _new_session(client)
        state["analysis_id"] = await _analyse(client, session_id)
        run = await _post(client, session_id, "Should hERG stop us developing this compound?")
        assert run["status"] == "completed", run
        assert state["answer"]["status"] == "completed", state["answer"]

        case_id = (await client.get(
            f"/v1/sessions/{session_id}/cases", headers=AUTH
        )).json()["cases"][0]["case_id"]
        case = (await client.get(f"/v1/sessions/{session_id}/cases/{case_id}", headers=AUTH)).json()
        ledger = [(e["actor"], e["source_class"], e["source_ref"], e["stance"], e["hypothesis_ids"])
                  for e in case["evidence"]]
        assert ledger == [
            ("server", "predictor_fact", f"observation:{state['observation_id']}", "contextual", []),
            ("server", "external_experimental", f"evidence:{state['evidence_id']}", "contextual", []),
        ]
        assert case["evidence"][0]["locator"] == "predictions.herg.probability_blocker"
        dossier = (await client.get(
            f"/v1/sessions/{session_id}/runs/{run['run_id']}/dossier", headers=AUTH
        )).json()
        assert len(dossier["unlinked_evidence"]) == 2
        assert len(dossier["predictor_facts"]) == 1

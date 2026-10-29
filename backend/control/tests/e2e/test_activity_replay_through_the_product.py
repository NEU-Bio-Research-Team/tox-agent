"""#44 through the product: a frozen screening attached, read and cited.

The completion condition: the agent extracts the right numbers, labels and
statuses through the resolver, and the whole path replays from a frozen
snapshot without any activity serving. On the scripted runtime:

* user story 1 -- "why was this screening sent to review, and did toxicity
  run?" -- answered from the three reads, refused without the limitations its
  observations require and accepted with them;
* one ligand on two targets is two screenings, each read by its own id;
* a stage that did not run is reported as not run, not as a negative;
* a session cannot read another session's screening, and a client can only
  name a screening the deployment holds, never post one.
"""
from __future__ import annotations

import json

import pytest

from tests.e2e.test_scripted_runtime import _install_scripted_runtime, _new_session
from tests.support.activity import frozen_screening, replay_file, screening_id
from tests.support.api import AUTH, OTHER_AUTH, api_client, settings, wait_for_run
from tests.support.predictor import StubPredictor
from toxagent.platform.config import ActivitySettings

pytestmark = pytest.mark.anyio

REQUIRED = [
    "toxicity_not_run_not_negative", "pic50_not_therapeutic_efficacy",
    "activity_model_not_admitted", "activity_interval_is_marginal_coverage",
    "policy_decision_not_a_measurement",
]


@pytest.fixture
def flags_on(monkeypatch):
    monkeypatch.setenv("TOXAGENT_FLAG_ANSWER_DRAFT_V2", "1")
    monkeypatch.setenv("TOXAGENT_FLAG_ACTIVITY_REPLAY_V1", "1")


def _config(tmp_path, *records):
    path, sha = replay_file(tmp_path, *records)
    return settings(activity=ActivitySettings(replay_path=str(path), replay_sha256=sha))


async def _attach(client, session_id: str, record: dict):
    return await client.post(
        f"/v1/sessions/{session_id}/screenings",
        json={"screening_id": screening_id(record)}, headers=AUTH,
    )


async def _ask(client, session_id: str, text: str) -> dict:
    submitted = await client.post(
        f"/v1/sessions/{session_id}/messages",
        json={"intent_hint": "auto", "content": [{"type": "text", "text": text}]},
        headers=AUTH,
    )
    assert submitted.status_code == 202, submitted.text
    return await wait_for_run(client, session_id, submitted.json()["run_id"])


def _claims(state_id: str, estimate_id: str, decision_id: str) -> list[dict]:
    def claim(ref, kind, observation_id, path, text, transform="identity"):
        return {"local_ref": ref, "kind": kind, "text": text, "observation_id": observation_id,
                "field_path": path, "transform": transform}

    return [
        claim("decision", "classification", decision_id, "status",
              "The policy decision is review."),
        claim("threshold", "numeric", decision_id, "policy.threshold_pic50",
              "The policy threshold is a pIC50 of 7.0.", "round:1"),
        claim("low", "numeric", estimate_id, "estimate.prediction_interval.0",
              "The interval's lower bound.", "round:1"),
        claim("high", "numeric", estimate_id, "estimate.prediction_interval.1",
              "The interval's upper bound.", "round:1"),
        claim("toxicity", "classification", state_id, "toxicity.run",
              "The toxicity stage did not run."),
    ]


async def test_why_a_screening_went_to_review_and_whether_toxicity_ran(db, flags_on, tmp_path):
    record = frozen_screening()
    seen: dict = {}

    async def script(turn) -> None:
        inventory = await turn.call_tool("get_artifact_inventory", {})
        seen["inventory"] = inventory["model_view"]
        sid = seen["inventory"]["screenings"][0]["screening_id"]
        state = await turn.call_tool("get_screening_state", {"screening_id": sid})
        estimate = await turn.call_tool("get_activity_snapshot", {"screening_id": sid})
        decision = await turn.call_tool("get_prioritisation_decision", {"screening_id": sid})
        seen["reads"] = (state, estimate, decision)
        claims = _claims(*(r["observation_ids"][0] for r in (state, estimate, decision)))
        markdown = (
            "The decision is {{decision}}: the interval {{low}} to {{high}} crosses the "
            "threshold of {{threshold}}. Toxicity ran: {{toxicity}}."
        )
        draft = {"schema_version": "grounded-answer-v2", "answer_markdown": markdown,
                 "claims": claims, "limitations": []}
        seen["without"] = await turn.call_tool("submit_grounded_answer", draft)
        seen["with"] = await turn.call_tool("submit_grounded_answer", {
            **draft, "limitations": [{"code": code, "text": ""} for code in REQUIRED],
        })

    async with api_client(db, StubPredictor(), config=_config(tmp_path, record)) as client:
        await _install_scripted_runtime(client, script)
        session_id = await _new_session(client)
        attached = await _attach(client, session_id, record)
        assert attached.status_code == 201, attached.text
        # No analysis in the session: the attached screening is the subject.
        run = await _ask(client, session_id, "Why was this screening sent to review, and did "
                                             "the toxicity stage run?")
        product = (await client.get("/v1/system/effective-product", headers=AUTH)).json()
        run_state = (await client.get(
            f"/v1/sessions/{session_id}/runs/{run['run_id']}/decision-state", headers=AUTH,
        )).json()

    assert run["status"] == "completed", run
    assert run["intent"] == "decision_support"
    (listed,) = seen["inventory"]["screenings"]
    assert listed["screening_id"] == screening_id(record)
    assert listed["toxicity_run"] is False
    assert seen["inventory"]["activity"] == {"mode": "replay", "serving": False}

    state, estimate, decision = (r["model_view"] for r in seen["reads"])
    assert state["status"] == "completed_with_toxicity_not_run"
    assert state["toxicity"]["not_run_reason"] == "review_required_awaiting_researcher"
    assert estimate["mode"] == "replay"
    assert estimate["estimate"]["prediction_interval"] == [6.8, 7.4]
    assert decision["reason_codes"] == ["prediction_interval_crosses_target_threshold"]

    refused = seen["without"]
    assert refused["status"] == "error"
    (violation,) = refused["error"]["details"]["violations"]
    assert violation["code"] == "missing_required_limitation"
    assert set(violation["expected"]) == set(REQUIRED)
    accepted = seen["with"]
    assert accepted["status"] == "completed", accepted
    rendered = accepted["canonical"]["answer_markdown"]
    assert "6.8" in rendered and "7.4" in rendered and "7.0" in rendered

    replayed = product["providers"]["activity_replay"]
    assert replayed["serving"] is False and replayed["path_name"] == "screenings.json"
    # #46: read from the run's own artifacts, no evidence needed -- a lookup,
    # not the "insufficient_evidence" the live run was given.
    assert run_state["stop_reason"] == "answered_from_artifacts"
    assert run_state["answer_basis"] == {"observations": 5, "evidence": 0}


async def test_an_answer_waiting_on_the_researcher_stops_on_that(db, flags_on, tmp_path,
                                                                 monkeypatch):
    monkeypatch.setenv("TOXAGENT_FLAG_TYPED_STOP_V1", "1")
    record = frozen_screening()

    async def script(turn) -> None:
        state = await turn.call_tool("get_screening_state",
                                     {"screening_id": screening_id(record)})
        await turn.call_tool("submit_grounded_answer", {
            "schema_version": "grounded-answer-v2",
            "answer_markdown": "Toxicity ran: {{tox}}. Whether to run it depends on your "
                               "exposure target, which this session does not hold.",
            "claims": [{"local_ref": "tox", "kind": "classification", "text": "Toxicity ran.",
                        "observation_id": state["observation_ids"][0],
                        "field_path": "toxicity.run"}],
            "limitations": [{"code": "toxicity_not_run_not_negative", "text": ""}],
            "stop": {"reason": "needs_user_input",
                     "needed_inputs": ["the intended free plasma Cmax"]},
        })

    async with api_client(db, StubPredictor(), config=_config(tmp_path, record)) as client:
        await _install_scripted_runtime(client, script)
        session_id = await _new_session(client)
        await _attach(client, session_id, record)
        run = await _ask(client, session_id, "Should we run the toxicity stage now?")
        state = (await client.get(
            f"/v1/sessions/{session_id}/runs/{run['run_id']}/decision-state", headers=AUTH,
        )).json()

    assert run["status"] == "completed", run
    assert state["stop_reason"] == "needs_user_input"
    assert state["declared_stop"]["needed_inputs"] == ["the intended free plasma Cmax"]


async def test_one_ligand_on_two_targets_is_two_screenings(db, flags_on, tmp_path):
    egfr = frozen_screening(target_id="EGFR", uniprot_id="P00533", predicted=7.1)
    herg = frozen_screening(target_id="KCNH2", uniprot_id="Q12809", predicted=5.2,
                            interval=(4.9, 5.5))
    seen: dict = {}

    async def script(turn) -> None:
        for name, record in (("egfr", egfr), ("herg", herg)):
            seen[name] = await turn.call_tool(
                "get_activity_snapshot", {"screening_id": screening_id(record)}
            )
        turn.say("done")

    async with api_client(db, StubPredictor(), config=_config(tmp_path, egfr, herg)) as client:
        await _install_scripted_runtime(client, script)
        session_id = await _new_session(client)
        for record in (egfr, herg):
            assert (await _attach(client, session_id, record)).status_code == 201
        listed = (await client.get(f"/v1/sessions/{session_id}/screenings",
                                   headers=AUTH)).json()
        await _ask(client, session_id, "What are the estimates for each target?")

    assert sorted(s["target"]["target_id"] for s in listed["screenings"]) == ["EGFR", "KCNH2"]
    egfr_view, herg_view = seen["egfr"]["model_view"], seen["herg"]["model_view"]
    assert egfr_view["subject"]["target"]["target_id"] == "EGFR"
    assert herg_view["subject"]["target"]["target_id"] == "KCNH2"
    assert egfr_view["estimate"]["predicted_pic50"] == pytest.approx(7.1)
    assert herg_view["estimate"]["predicted_pic50"] == pytest.approx(5.2)
    assert seen["egfr"]["observation_ids"] != seen["herg"]["observation_ids"]


async def test_a_stage_that_did_not_run_is_reported_as_not_run(db, flags_on, tmp_path):
    failed = frozen_screening(activity_fails=True)
    seen: dict = {}

    async def script(turn) -> None:
        sid = screening_id(failed)
        seen["estimate"] = await turn.call_tool("get_activity_snapshot", {"screening_id": sid})
        seen["state"] = await turn.call_tool("get_screening_state", {"screening_id": sid})
        turn.say("done")

    async with api_client(db, StubPredictor(), config=_config(tmp_path, failed)) as client:
        await _install_scripted_runtime(client, script)
        session_id = await _new_session(client)
        await _attach(client, session_id, failed)
        await _ask(client, session_id, "What did the activity model estimate here?")

    estimate = seen["estimate"]
    assert estimate["status"] == "completed"
    assert estimate["model_view"]["produced"] is False
    assert estimate["model_view"]["screening_status"] == "activity_failed"
    assert estimate["observation_ids"] == []
    assert "not a negative result" in estimate["model_view"]["note"]
    assert seen["state"]["model_view"]["status"] == "activity_failed"


async def test_a_screening_is_named_never_supplied_and_stays_in_its_session(
    db, flags_on, tmp_path,
):
    record = frozen_screening()
    second = frozen_screening(target_id="KCNH2", uniprot_id="Q12809")
    unknown = frozen_screening()  # frozen, but not in the deployment's file
    seen: dict = {}

    async def script(turn) -> None:
        seen["read"] = await turn.call_tool(
            "get_screening_state", {"screening_id": screening_id(record)}
        )
        turn.say("done")

    async with api_client(db, StubPredictor(), config=_config(tmp_path, record, second)) as client:
        await _install_scripted_runtime(client, script)
        owner = await _new_session(client)
        assert (await _attach(client, owner, record)).status_code == 201
        again = await _attach(client, owner, record)
        missing = await _attach(client, owner, unknown)
        # a record body is not accepted at all, only an id
        posted = await client.post(f"/v1/sessions/{owner}/screenings",
                                   json={"screening_id": screening_id(record), "record": record},
                                   headers=AUTH)
        foreign = await client.post(f"/v1/sessions/{owner}/screenings",
                                    json={"screening_id": screening_id(record)},
                                    headers=OTHER_AUTH)
        # another session of the same user, which never attached it, but
        # asks about it by id (attaching any screening makes the question
        # routable; the one asked about is still not this session's)
        other = await _new_session(client)
        await client.post(f"/v1/sessions/{other}/screenings",
                          json={"screening_id": screening_id(second)}, headers=AUTH)
        await _ask(client, other, "What is the status of screening " + screening_id(record) + "?")

    assert again.status_code == 201 and again.json()["record_sha256"] == record["content_sha256"]
    assert missing.status_code == 404 and missing.json()["error"]["code"] == "screening_not_found"
    assert posted.status_code == 400 and posted.json()["error"]["code"] == "invalid_request"
    assert foreign.status_code == 404
    assert seen["read"]["status"] == "error"
    assert seen["read"]["error"]["code"] == "screening_not_found"


async def test_without_the_flag_there_are_no_screenings(db, tmp_path, monkeypatch):
    monkeypatch.setenv("TOXAGENT_FLAG_ANSWER_DRAFT_V2", "1")
    record = frozen_screening()
    async with api_client(db, StubPredictor(), config=_config(tmp_path, record)) as client:
        session_id = await _new_session(client)
        response = await _attach(client, session_id, record)
        names = set(client.app.state.tool_registry.names())
    assert response.status_code == 404
    assert not names & {"get_screening_state", "get_activity_snapshot",
                        "get_prioritisation_decision"}


async def test_a_draft_refused_for_a_field_path_is_told_its_limitations_too(
    db, flags_on, tmp_path,
):
    """Live run 2026-10-21: the first draft cited the interval list, the second
    fixed it and was refused for two missing limitations, and the run fell
    back. Both are reported on the first refusal now, and the fallback of a
    screening session does not tell the researcher to run an analysis."""
    record = frozen_screening()
    seen: dict = {}

    async def script(turn) -> None:
        sid = screening_id(record)
        estimate = await turn.call_tool("get_activity_snapshot", {"screening_id": sid})
        draft = {"schema_version": "grounded-answer-v2",
                 "answer_markdown": "The interval is {{interval}}.",
                 "claims": [{"local_ref": "interval", "kind": "numeric", "text": "Interval.",
                             "observation_id": estimate["observation_ids"][0],
                             "field_path": "estimate.prediction_interval"}],
                 "limitations": []}
        seen["first"] = await turn.call_tool("submit_grounded_answer", draft)
        seen["second"] = await turn.call_tool("submit_grounded_answer", draft)

    async with api_client(db, StubPredictor(), config=_config(tmp_path, record)) as client:
        await _install_scripted_runtime(client, script)
        session_id = await _new_session(client)
        await _attach(client, session_id, record)
        await _ask(client, session_id, "What is the estimate's interval?")
        messages = (await client.get(f"/v1/sessions/{session_id}/messages",
                                     headers=AUTH)).json()["messages"]

    codes = [v["code"] for v in seen["first"]["error"]["details"]["violations"]]
    assert codes == ["claim_field_not_numeric", "missing_required_limitation"]
    assert seen["second"]["model_view"]["is_fallback"] is True
    reply = json.dumps(messages[-1])
    assert "run an analysis" not in reply.lower()
    assert "passed validation" in reply

"""#44 through the product: a run is bound to one subject when it is accepted.

On the scripted runtime, with subject_binding_v1:

* one ligand on two targets is two scientific cases, and coming back to the
  first target continues its case rather than opening a third;
* a selected screening is the subject even when the session also has an
  active analysis -- the gateway never falls back to the session's selection;
* two subjects and no selection is a subject_ambiguous clarification, and no
  runtime turn is spent;
* a screening another session holds cannot be named;
* a replayed screening whose toxicity stage ran reads its frozen toxicity
  analysis through a session-local copy, without calling the predictor;
* what an accepted answer cites reaches the ledger as what it is: a pIC50 is a
  predictor fact scoped by target and unit, the policy decision a policy fact,
  the stage status a workflow fact -- and another target's estimate is not
  this case's evidence.
"""
from __future__ import annotations

import pytest

from tests.e2e.test_scripted_runtime import _analyse, _install_scripted_runtime, _new_session
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
    for name in ("ANSWER_DRAFT_V2", "ACTIVITY_REPLAY_V1", "SUBJECT_BINDING_V1",
                 "SCIENTIFIC_CASE_V1"):
        monkeypatch.setenv(f"TOXAGENT_FLAG_{name}", "1")


def _config(tmp_path, *records):
    path, sha = replay_file(tmp_path, *records)
    return settings(activity=ActivitySettings(replay_path=str(path), replay_sha256=sha))


def _ref(record) -> dict:
    return {"kind": "activity_screening", "screening_id": screening_id(record)}


async def _attach(client, session_id: str, record: dict) -> None:
    response = await client.post(
        f"/v1/sessions/{session_id}/screenings",
        json={"screening_id": screening_id(record)}, headers=AUTH,
    )
    assert response.status_code == 201, response.text


async def _send(client, session_id: str, text: str, subject_ref: dict | None = None,
                headers=AUTH):
    body = {"content": [{"type": "text", "text": text}]}
    if subject_ref is not None:
        body["subject_ref"] = subject_ref
    return await client.post(f"/v1/sessions/{session_id}/messages", json=body, headers=headers)


async def _ask(client, session_id: str, text: str, subject_ref: dict | None = None) -> dict:
    submitted = await _send(client, session_id, text, subject_ref)
    assert submitted.status_code == 202, submitted.text
    return await wait_for_run(client, session_id, submitted.json()["run_id"])


async def _cases(client, session_id: str) -> list[dict]:
    listed = (await client.get(f"/v1/sessions/{session_id}/cases", headers=AUTH)).json()
    return [
        (await client.get(f"/v1/sessions/{session_id}/cases/{c['case_id']}", headers=AUTH)).json()
        for c in listed["cases"]
    ]


async def test_two_targets_are_two_cases_and_returning_continues_the_first(
    db, flags_on, tmp_path,
):
    egfr = frozen_screening()
    herg = frozen_screening(target_id="KCNH2", uniprot_id="Q12809", predicted=5.2,
                            interval=(4.9, 5.5))
    seen: list[dict] = []

    async def script(turn) -> None:
        seen.append((await turn.call_tool("get_artifact_inventory", {}))["model_view"])
        turn.say("noted")

    async with api_client(db, StubPredictor(), config=_config(tmp_path, egfr, herg)) as client:
        await _install_scripted_runtime(client, script)
        session_id = await _new_session(client)
        for record in (egfr, herg):
            await _attach(client, session_id, record)
        first = await _ask(client, session_id, "What is the estimate?", _ref(egfr))
        second = await _ask(client, session_id, "And on hERG?", _ref(herg))
        cases_mid = await _cases(client, session_id)
        third = await _ask(client, session_id, "Back to EGFR: anything else?", _ref(egfr))
        cases = await _cases(client, session_id)

    # The script only reads; the runs end without an answer, which is not
    # what this test is about. The case each was bound to is.
    assert {r["status"] for r in (first, second, third)} <= {"completed", "failed"}
    assert [v["subject"]["subject_ref"]["screening_id"] for v in seen] == [
        screening_id(egfr), screening_id(herg), screening_id(egfr),
    ]
    assert seen[0]["subject"]["identity"]["target"]["target_id"] == "EGFR"
    assert len(cases_mid) == 2 and len(cases) == 2
    by_target = {c["subject"]["identity"]["target"]["target_id"]: c for c in cases}
    assert set(by_target) == {"EGFR", "KCNH2"}
    egfr_case = by_target["EGFR"]
    assert [r["run_id"] for r in egfr_case["runs"]] == [first["run_id"], third["run_id"]]
    before = next(c for c in cases_mid if c["case_id"] == egfr_case["case_id"])
    assert egfr_case["revision"] > before["revision"]
    assert egfr_case["subject_key"].startswith("activity:")
    assert all(r["binding"]["binding_sha256"] for r in egfr_case["runs"])


async def test_a_selected_screening_wins_over_the_active_analysis(db, flags_on, tmp_path):
    record = frozen_screening()
    seen: dict = {}

    async def script(turn) -> None:
        seen["inventory"] = (await turn.call_tool("get_artifact_inventory", {}))["model_view"]
        turn.say("noted")

    async with api_client(db, StubPredictor(), config=_config(tmp_path, record)) as client:
        await _install_scripted_runtime(client, script)
        session_id = await _new_session(client)
        await _analyse(client, session_id)
        await _attach(client, session_id, record)
        run = await _ask(client, session_id, "Why was this sent to review?", _ref(record))
        detail = (await client.get(f"/v1/sessions/{session_id}/runs/{run['run_id']}",
                                   headers=AUTH)).json()
        state = (await client.get(
            f"/v1/sessions/{session_id}/runs/{run['run_id']}/decision-state", headers=AUTH,
        )).json()
        async with db.unit_of_work() as uow:
            configuration = await uow.run_configuration_snapshots.get(run["run_id"])

    inventory = seen["inventory"]
    assert inventory["analysis"] is None
    assert inventory["subject"]["kind"] == "activity_screening"
    assert [s["is_subject"] for s in inventory["screenings"]] == [True]
    assert state["subject_refs"] == [f"screening:{screening_id(record)}"]
    assert detail["run_id"] == run["run_id"]
    bound = configuration["subject_binding"]
    assert bound["subject_ref"] == {"screening_id": screening_id(record)}
    assert bound["resolved_by"] == "explicit"
    assert bound["artifact_refs"][0]["content_sha256"] == record["content_sha256"]


async def test_two_subjects_and_no_selection_is_a_clarification(db, flags_on, tmp_path):
    egfr = frozen_screening()
    herg = frozen_screening(target_id="KCNH2", uniprot_id="Q12809")
    dispatched: list[str] = []

    async def script(turn) -> None:
        dispatched.append("turn")
        turn.say("should not run")

    async with api_client(db, StubPredictor(), config=_config(tmp_path, egfr, herg)) as client:
        await _install_scripted_runtime(client, script)
        session_id = await _new_session(client)
        for record in (egfr, herg):
            await _attach(client, session_id, record)
        submitted = await _send(client, session_id, "What is the estimate here?")
        body = submitted.json()
        run = await wait_for_run(client, session_id, body["run_id"])

    assert submitted.status_code == 202
    assert body["selected_intent"] == "clarification_required"
    assert body["clarification"]["code"] == "subject_ambiguous"
    assert sorted(body["clarification"]["options"]) == sorted(
        f"screening:{screening_id(r)}" for r in (egfr, herg)
    )
    assert run["status"] == "completed" and dispatched == []


async def test_a_screening_of_another_session_cannot_be_named(db, flags_on, tmp_path):
    record = frozen_screening()
    other = frozen_screening(target_id="KCNH2", uniprot_id="Q12809")

    async def script(turn) -> None:
        turn.say("noted")

    async with api_client(db, StubPredictor(), config=_config(tmp_path, record, other)) as client:
        await _install_scripted_runtime(client, script)
        owner = await _new_session(client)
        await _attach(client, owner, record)
        stranger = await _new_session(client)
        await _attach(client, stranger, other)
        named = await _send(client, stranger, "What about that one?", _ref(record))
        foreign = await _send(client, owner, "What about it?", _ref(record), headers=OTHER_AUTH)
        malformed = await client.post(
            f"/v1/sessions/{owner}/messages",
            json={"content": [{"type": "text", "text": "q"}],
                  "subject_ref": {"kind": "activity_screening", "analysis_id": "x"}},
            headers=AUTH,
        )

    assert named.status_code == 404 and named.json()["error"]["code"] == "screening_not_found"
    assert foreign.status_code == 404
    assert malformed.status_code == 400


async def test_subject_ref_is_refused_when_binding_is_off(db, tmp_path, monkeypatch):
    monkeypatch.setenv("TOXAGENT_FLAG_ACTIVITY_REPLAY_V1", "1")
    record = frozen_screening()
    async with api_client(db, StubPredictor(), config=_config(tmp_path, record)) as client:
        session_id = await _new_session(client)
        await _attach(client, session_id, record)
        response = await _send(client, session_id, "Why review?", _ref(record))
    assert response.status_code == 400


async def test_a_replayed_toxicity_analysis_is_read_without_the_predictor(db, flags_on, tmp_path):
    record = frozen_screening(toxicity="completed", herg_probability=0.81)
    predictor = StubPredictor()
    seen: dict = {}

    async def script(turn) -> None:
        sid = screening_id(record)
        seen["state"] = (await turn.call_tool("get_screening_state", {"screening_id": sid}))
        local = seen["state"]["model_view"]["linked_toxicity_analysis_id"]
        seen["herg"] = await turn.call_tool(
            "get_analysis_slice", {"analysis_id": local, "section": "herg"},
        )
        turn.say("noted")

    async with api_client(db, predictor, config=_config(tmp_path, record)) as client:
        await _install_scripted_runtime(client, script)
        session_id = await _new_session(client)
        await _attach(client, session_id, record)
        run = await _ask(client, session_id, "Did the toxicity stage flag hERG?", _ref(record))
        again = await _ask(client, session_id, "And again?", _ref(record))
        analyses = (await client.get(f"/v1/sessions/{session_id}", headers=AUTH)).json()

    assert run["run_id"] != again["run_id"]
    state = seen["state"]["model_view"]
    origin = record["toxicity_snapshot"]["origin_analysis_id"]
    assert state["toxicity"]["analysis_id"] == origin
    local = state["linked_toxicity_analysis_id"]
    assert local and local != origin
    herg = seen["herg"]
    assert herg["status"] == "completed", herg
    values = herg["model_view"]["values"]
    assert values["probability_blocker"]["value"] == pytest.approx(0.81)
    assert not [r for r in predictor.requests if "predictions" in r["path"]]
    # One copy per session, however many runs read it.
    assert analyses.get("active_analysis") is None


async def test_the_ledger_records_activity_facts_as_what_they_are(db, flags_on, tmp_path):
    egfr = frozen_screening()
    herg = frozen_screening(target_id="KCNH2", uniprot_id="Q12809", predicted=5.2,
                            interval=(4.9, 5.5))

    async def script(turn) -> None:
        reads = {}
        for name, tool in (("state", "get_screening_state"),
                           ("estimate", "get_activity_snapshot"),
                           ("decision", "get_prioritisation_decision")):
            reads[name] = (await turn.call_tool(tool, {"screening_id": screening_id(egfr)}))
        foreign = await turn.call_tool("get_activity_snapshot",
                                       {"screening_id": screening_id(herg)})

        def claim(ref, kind, observation, path, text, transform="identity"):
            return {"local_ref": ref, "kind": kind, "text": text,
                    "observation_id": observation["observation_ids"][0],
                    "field_path": path, "transform": transform}

        await turn.call_tool("submit_grounded_answer", {
            "schema_version": "grounded-answer-v2",
            "answer_markdown": "Estimate {{p}}; decision {{d}}; toxicity ran: {{t}}. "
                               "On hERG the estimate is {{h}}.",
            "claims": [
                claim("p", "numeric", reads["estimate"], "estimate.predicted_pic50",
                      "The estimate.", "round:1"),
                claim("d", "classification", reads["decision"], "status", "The decision."),
                claim("t", "classification", reads["state"], "toxicity.run", "Toxicity ran."),
                claim("h", "numeric", foreign, "estimate.predicted_pic50",
                      "The hERG estimate.", "round:1"),
            ],
            "limitations": [{"code": code, "text": ""} for code in REQUIRED],
        })

    async with api_client(db, StubPredictor(), config=_config(tmp_path, egfr, herg)) as client:
        await _install_scripted_runtime(client, script)
        session_id = await _new_session(client)
        for record in (egfr, herg):
            await _attach(client, session_id, record)
        run = await _ask(client, session_id, "Summarise this screening.", _ref(egfr))
        (case,) = await _cases(client, session_id)
        dossier = (await client.get(
            f"/v1/sessions/{session_id}/runs/{run['run_id']}/dossier", headers=AUTH,
        )).json()

    assert run["status"] == "completed", run
    classes = {e["provenance"]["observation_kind"]: e for e in case["evidence"]}
    assert set(classes) == {"activity_estimate", "prioritisation_decision", "screening_state"}
    estimate = classes["activity_estimate"]
    assert estimate["source_class"] == "predictor_fact"
    assert estimate["scope"]["unit"] == "pIC50"
    assert estimate["scope"]["target_id"] == "EGFR"
    assert "endpoint" not in estimate["scope"]
    assert classes["prioritisation_decision"]["source_class"] == "policy_fact"
    assert classes["screening_state"]["source_class"] == "workflow_fact"
    assert all(e["provenance"]["record_sha256"] == egfr["content_sha256"]
               for e in classes.values())
    # The hERG screening's estimate was cited, and is not this case's evidence.
    assert all(e["scope"].get("target_id") != "KCNH2" for e in case["evidence"])
    assert dossier["subject"]["identity"]["target"]["target_id"] == "EGFR"
    assert not dossier["explanation_layers"]["scientific_evidence"]


async def test_the_screening_read_model_states_every_stage(db, flags_on, tmp_path):
    review = frozen_screening()
    ran = frozen_screening(target_id="KCNH2", uniprot_id="Q12809", toxicity="completed")
    failed = frozen_screening(target_id="ABL1", uniprot_id="P00519", toxicity="failed")
    async with api_client(db, StubPredictor(),
                          config=_config(tmp_path, review, ran, failed)) as client:
        session_id = await _new_session(client)
        for record in (review, ran, failed):
            await _attach(client, session_id, record)
        views = {
            name: (await client.get(
                f"/v1/sessions/{session_id}/screenings/{screening_id(record)}", headers=AUTH,
            )).json()
            for name, record in (("review", review), ("ran", ran), ("failed", failed))
        }
        stranger = await _new_session(client)
        missing = await client.get(
            f"/v1/sessions/{stranger}/screenings/{screening_id(review)}", headers=AUTH,
        )

    assert views["review"]["stages"]["toxicity"] == {
        "status": "not_run", "not_run_reason": "review_required_awaiting_researcher",
        "failure_code": None, "authorised_by": None, "frozen_analysis_available": False,
    }
    assert views["review"]["stages"]["prioritisation"]["decision"] == "review"
    assert views["ran"]["stages"]["toxicity"]["status"] == "completed"
    assert views["ran"]["stages"]["toxicity"]["frozen_analysis_available"] is True
    assert views["failed"]["stages"]["toxicity"]["status"] == "failed"
    assert views["review"]["mode"] == "replay" and views["review"]["serving"] is False
    assert views["review"]["model_admitted"] is False
    assert views["review"]["subject_key"] != views["ran"]["subject_key"]
    assert missing.status_code == 404

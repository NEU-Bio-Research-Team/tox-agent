"""The four user stories and their acceptance matrix (#46, master plan §12).

The matrix below is the artifact: one row per story and per control variant,
each naming either the test that already proves it or ``runs_here``, in which
case this module runs it. Two tests keep the matrix honest -- one checks that
every named test still exists, so a rename cannot leave a story silently
unproven, and one checks that every story and every declared variant has a
row.

What is deliberately *not* here: the live-LLM level of each story (§12 level
three). It needs a model and a provider quota, so it is run from the
benchmark CLI against a deployment, not in this suite; the matrix marks those
rows ``live_only`` and says what they add.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.e2e.test_scripted_runtime import _install_scripted_runtime, _new_session
from tests.support.activity import frozen_screening, replay_file, screening_id
from tests.support.api import AUTH, api_client, settings, wait_for_run
from toxagent.api.responses import CaseRunChanges
from tests.support.predictor import StubPredictor
from toxagent.platform.config import ActivitySettings

pytestmark = pytest.mark.anyio

E2E = Path(__file__).resolve().parent

#: Each row: what frozen input the variant needs, what must hold, and where
#: that is proven. ``covered_by`` is "<file>::<test>" in tests/e2e.
MATRIX: dict[str, dict[str, dict[str, str]]] = {
    "US1 understand a screening": {
        "review, interval crosses the threshold, toxicity not run": {
            "holds": "the decision, its reason code and the interval are read from the "
                     "artifacts; the answer owes every limitation its observations require; "
                     "the run stops as answered_from_artifacts",
            "covered_by": "test_activity_replay_through_the_product.py::"
                          "test_why_a_screening_went_to_review_and_whether_toxicity_ran",
        },
        "toxicity completed, its analysis replayed": {
            "holds": "the linked frozen analysis is read through a session-local copy and "
                     "the predictor is never called",
            "covered_by": "test_subject_binding_through_the_product.py::"
                          "test_a_replayed_toxicity_analysis_is_read_without_the_predictor",
        },
        "toxicity not run, reported as a state": {
            "holds": "the stage says not_run with its reason and is not a negative result",
            "covered_by": "test_activity_replay_through_the_product.py::"
                          "test_a_stage_that_did_not_run_is_reported_as_not_run",
        },
        "every stage stated in the read model": {
            "holds": "activity, prioritisation and toxicity each carry a status, a reason "
                     "and whether a frozen analysis exists",
            "covered_by": "test_subject_binding_through_the_product.py::"
                          "test_the_screening_read_model_states_every_stage",
        },
        "the activity stage failed": {
            "holds": "there is no estimate and no decision, the stage says failed with its "
                     "code, and a failure is not an inactive result",
            "covered_by": "runs_here",
        },
        "the interval does not cross the threshold": {
            "holds": "the policy decides without review, and no review reason code is "
                     "recorded",
            "covered_by": "runs_here",
        },
        "the model behind the estimate is not admitted": {
            "holds": "the estimate says so and the answer owes "
                     "activity_model_not_admitted",
            "covered_by": "runs_here",
        },
    },
    "US2 resolve a REVIEW from the literature": {
        "the ligand on this target is direct, its other target contextual": {
            "holds": "relevance is assessed over the bound target; a record about the "
                     "target with another ligand is not citable",
            "covered_by": "test_subject_evidence_through_the_product.py::"
                          "test_a_screening_is_searched_on_its_ligand_and_target",
        },
        "a record about another target cannot be direct": {
            "holds": "the relation is refused (evidence_relation_direct_out_of_subject), "
                     "and a relation cannot restate the artifact's target",
            "covered_by": "test_subject_evidence_through_the_product.py::"
                          "test_a_screening_is_searched_on_its_ligand_and_target",
        },
        "a table value read and quoted with no network": {
            "holds": "the row is quoted from the frozen store and the number validated",
            "covered_by": "test_frozen_full_text_through_the_product.py::"
                          "test_a_table_value_is_read_and_quoted_offline",
        },
        "the capture failed: abstract only": {
            "holds": "the access failure reaches the answer, which owes "
                     "evidence_full_text_not_read",
            "covered_by": "test_frozen_full_text_through_the_product.py::"
                          "test_a_captured_access_failure_reaches_the_answer",
        },
        "the target is outside the catalog": {
            "holds": "the capability is reported as unavailable and nothing is promoted",
            "covered_by": "test_subject_evidence_through_the_product.py::"
                          "test_a_target_outside_the_catalog_promotes_nothing",
        },
        "the corpus holds no source for the question": {
            "holds": "an empty result is said as such, apart from a provider failure",
            "covered_by": "test_fixed_workflow_arm.py::"
                          "test_a_provider_failure_is_reported_as_a_failure_not_as_no_evidence",
        },
        "a real model chooses its own reads": {
            "holds": "a policy model reaches an acceptable conclusion by one of several "
                     "valid routes; graded on the outcome, not on a fixed tool sequence",
            "covered_by": "live_only",
        },
    },
    "US3 cross-check the toxicity side": {
        "the ledger keeps each activity fact as what it is": {
            "holds": "a pIC50 is a predictor fact scoped by target and unit, the policy "
                     "decision a policy fact, the stage status a workflow fact; another "
                     "target's estimate is not this case's evidence",
            "covered_by": "test_subject_binding_through_the_product.py::"
                          "test_the_ledger_records_activity_facts_as_what_they_are",
        },
        "a source on another endpoint is not a contradiction": {
            "holds": "the relation is refused or recorded as a scope mismatch, never as a "
                     "true contradiction",
            "covered_by": "test_subject_evidence_through_the_product.py::"
                          "test_a_screening_is_searched_on_its_ligand_and_target",
        },
        "a toxicity-only subject is bound by reference": {
            "holds": "the frozen analysis is copied into the session, hash-checked, with no "
                     "predictor call, and both turns share one case",
            "covered_by": "test_frozen_analysis_subject.py::"
                          "test_a_frozen_analysis_is_bound_by_reference_without_the_predictor",
        },
        "the predictor is wrong and a verified source says so": {
            "holds": "the answer questions the predictor's output rather than repeating it; "
                     "needs an SME-verified label, so it is a pilot case, not a fixture",
            "covered_by": "live_only",
        },
    },
    "US4 update across turns": {
        "a measured value arrives and the artifacts stay": {
            "holds": "the case revision rises, the evidence is recorded with the "
                     "researcher's scope, and the estimate, interval, decision and stage "
                     "status are unchanged",
            "covered_by": "runs_here",
        },
        "the researcher switches to another target": {
            "holds": "a second case is opened, the first keeps its history, and nothing of "
                     "the first target's activity carries over",
            "covered_by": "test_subject_binding_through_the_product.py::"
                          "test_two_targets_are_two_cases_and_returning_continues_the_first",
        },
        "the same turn's context reaches every arm": {
            "holds": "B, C and D are sent the same rendered text; only a case arm also "
                     "files it into the case, typed",
            "covered_by": "test_protocol_v2_four_arms.py::"
                          "test_four_arms_share_inputs_and_the_study_holds_them_to_it",
        },
        "a queued run keeps the subject it was bound to": {
            "holds": "selecting another screening while a run is queued does not change "
                     "what that run answers about",
            "covered_by": "test_subject_binding_queue.py::"
                          "test_the_claiming_worker_runs_the_subject_bound_at_admission",
        },
        "an exposure the researcher already gave is not asked for again": {
            "holds": "the fixed workflow takes it from the conversation and asks for "
                     "nothing it already has",
            "covered_by": "test_fixed_workflow_arm.py::"
                          "test_an_exposure_the_researcher_already_gave_is_used_not_asked_again",
        },
    },
}

REQUIRED_WITH_ESTIMATE = {
    "toxicity_not_run_not_negative", "pic50_not_therapeutic_efficacy",
    "activity_model_not_admitted", "activity_interval_is_marginal_coverage",
    "policy_decision_not_a_measurement",
}


def test_every_story_in_the_matrix_names_a_test_that_exists():
    """A renamed or deleted test must not leave a story's row pointing at
    nothing: the matrix is only an acceptance artifact while it resolves."""
    missing: list[str] = []
    for story, variants in MATRIX.items():
        for variant, row in variants.items():
            where = row["covered_by"]
            if where in ("runs_here", "live_only"):
                continue
            file_name, _, test_name = where.partition("::")
            for directory in (E2E, E2E.parent / "integration"):
                path = directory / file_name
                if path.exists():
                    break
            if not path.exists() or f"def {test_name}(" not in path.read_text():
                missing.append(f"{story} / {variant} -> {where}")
    assert missing == [], "the matrix names tests that do not exist: " + "; ".join(missing)


def test_the_matrix_covers_the_four_stories_and_declares_what_is_live_only():
    assert sorted(s.split()[0] for s in MATRIX) == ["US1", "US2", "US3", "US4"]
    for story, variants in MATRIX.items():
        assert len(variants) >= 4, f"{story} has too few control variants"
        assert all(row.get("holds") for row in variants.values())
    live = {f"{story} / {variant}" for story, variants in MATRIX.items()
            for variant, row in variants.items() if row["covered_by"] == "live_only"}
    # Declared, not forgotten: these are run from the benchmark CLI against a
    # deployment with a model, and the pilot grades their outcomes.
    assert live == {
        "US2 resolve a REVIEW from the literature / a real model chooses its own reads",
        "US3 cross-check the toxicity side / the predictor is wrong and a verified source "
        "says so",
    }


def _config(tmp_path, *records):
    path, sha = replay_file(tmp_path, *records)
    return settings(activity=ActivitySettings(replay_path=str(path), replay_sha256=sha))


@pytest.fixture
def flags_on(monkeypatch):
    for name in ("ANSWER_DRAFT_V2", "ACTIVITY_REPLAY_V1", "SUBJECT_BINDING_V1",
                 "SCIENTIFIC_CASE_V1"):
        monkeypatch.setenv(f"TOXAGENT_FLAG_{name}", "1")


def _reader(seen: dict):
    """Reads the bound screening's three artifacts and says nothing: these
    variants are about what the server states, not about model prose."""

    async def script(turn) -> None:
        inventory = (await turn.call_tool("get_artifact_inventory", {}))["model_view"]
        seen["inventory"] = inventory
        sid = (inventory["subject"]["subject_ref"] or {}).get("screening_id")
        reads = {}
        for tool in ("get_screening_state", "get_activity_snapshot",
                     "get_prioritisation_decision"):
            reads[tool] = await turn.call_tool(tool, {"screening_id": sid})
        seen.setdefault("reads", []).append(reads)
        turn.say("read")

    return script


async def _ask(client, session_id: str, text: str, record: dict,
               actions: list[dict] | None = None) -> dict:
    submitted = await client.post(
        f"/v1/sessions/{session_id}/messages", headers=AUTH,
        json={"content": [{"type": "text", "text": text}],
              "subject_ref": {"kind": "activity_screening",
                              "screening_id": screening_id(record)}},
    )
    assert submitted.status_code == 202, submitted.text
    del actions
    return await wait_for_run(client, session_id, submitted.json()["run_id"])


async def _attach(client, session_id: str, record: dict) -> None:
    response = await client.post(f"/v1/sessions/{session_id}/screenings", headers=AUTH,
                                 json={"screening_id": screening_id(record)})
    assert response.status_code == 201, response.text


# ----------------------------------------------------------------- US1

async def test_us1_a_failed_activity_stage_has_no_estimate_and_is_not_a_negative(
    db, flags_on, tmp_path,
):
    record = frozen_screening(target_id="ABL1", uniprot_id="P00519", activity_fails=True)
    seen: dict = {}
    async with api_client(db, StubPredictor(), config=_config(tmp_path, record)) as client:
        await _install_scripted_runtime(client, _reader(seen))
        session_id = await _new_session(client)
        await _attach(client, session_id, record)
        await _ask(client, session_id, "What did this screening find?", record)
        detail = (await client.get(
            f"/v1/sessions/{session_id}/screenings/{screening_id(record)}", headers=AUTH)).json()

    reads = seen["reads"][0]
    state = reads["get_screening_state"]["model_view"]
    # The status and the code are both in the view, so an answer can cite
    # why the stage failed instead of only that it did.
    assert state["status"] == "activity_failed"
    assert state["activity_failure_code"] == "activity_model_unavailable"
    assert "activity_failure_code" in reads["get_screening_state"]["model_view"]
    # Nothing to read, and each tool says so in words rather than returning
    # an empty estimate a model could read as "no activity".
    for tool in ("get_activity_snapshot", "get_prioritisation_decision"):
        view = reads[tool]["model_view"]
        assert view["produced"] is False
        assert "not a negative result" in view["note"]
        assert "estimate" not in view and "status" not in view
    assert detail["stages"]["activity"] == {"status": "failed",
                                            "failure_code": "activity_model_unavailable"}
    assert detail["stages"]["prioritisation"]["status"] == "not_run"
    assert detail["activity_estimate"] is None and detail["prioritisation_decision"] is None
    assert detail["model_admitted"] is None


async def test_us1_an_interval_clear_of_the_threshold_is_decided_without_review(
    db, flags_on, tmp_path,
):
    record = frozen_screening(predicted=5.2, interval=(4.9, 5.5), threshold=7.0)
    seen: dict = {}
    async with api_client(db, StubPredictor(), config=_config(tmp_path, record)) as client:
        await _install_scripted_runtime(client, _reader(seen))
        session_id = await _new_session(client)
        await _attach(client, session_id, record)
        await _ask(client, session_id, "Is this one worth following up?", record)
        detail = (await client.get(
            f"/v1/sessions/{session_id}/screenings/{screening_id(record)}", headers=AUTH)).json()

    decision = seen["reads"][0]["get_prioritisation_decision"]["model_view"]
    assert decision["status"] != "review"
    assert "prediction_interval_crosses_target_threshold" not in (decision["reason_codes"] or ())
    assert detail["stages"]["toxicity"]["status"] == "not_run"
    # The estimate still owes its limitations: a clear decision is not a
    # validated one.
    estimate = seen["reads"][0]["get_activity_snapshot"]["model_view"]
    assert set(estimate["required_limitations"]) >= {"activity_model_not_admitted",
                                                     "activity_interval_is_marginal_coverage"}


async def test_us1_an_unadmitted_model_is_stated_and_owed_as_a_limitation(
    db, flags_on, tmp_path,
):
    record = frozen_screening()
    seen: dict = {}
    async with api_client(db, StubPredictor(), config=_config(tmp_path, record)) as client:
        await _install_scripted_runtime(client, _reader(seen))
        session_id = await _new_session(client)
        await _attach(client, session_id, record)
        await _ask(client, session_id, "How reliable is this estimate?", record)
        detail = (await client.get(
            f"/v1/sessions/{session_id}/screenings/{screening_id(record)}", headers=AUTH)).json()

    estimate = seen["reads"][0]["get_activity_snapshot"]["model_view"]
    assert estimate["admitted"] is False and estimate["model_id"]
    assert "activity_model_not_admitted" in estimate["required_limitations"]
    assert detail["model_admitted"] is False
    assert detail["mode"] == "replay" and detail["serving"] is False


# ----------------------------------------------------------------- US4

async def test_us4_a_measured_value_changes_the_assessment_and_nothing_else(
    db, flags_on, tmp_path,
):
    record = frozen_screening()
    seen: dict = {}
    async with api_client(db, StubPredictor(), config=_config(tmp_path, record)) as client:
        await _install_scripted_runtime(client, _reader(seen))
        session_id = await _new_session(client)
        await _attach(client, session_id, record)
        await _ask(client, session_id, "What does this screening say?", record)
        cases = (await client.get(f"/v1/sessions/{session_id}/cases", headers=AUTH)).json()
        case_id = cases["cases"][0]["case_id"]
        before = (await client.get(f"/v1/sessions/{session_id}/cases/{case_id}",
                                   headers=AUTH)).json()
        posted = await client.post(
            f"/v1/sessions/{session_id}/cases/{case_id}/context", headers=AUTH,
            json={"key": "measured_pIC50", "value": "6.2", "unit": "pIC50",
                  "source": "researcher",
                  "scope": {"target_id": "EGFR", "experimental_context": "IC50|wild_type"}},
        )
        assert posted.status_code == 200, posted.text
        await _ask(client, session_id, "Our lab measured 6.2. What changes?", record)
        after = (await client.get(f"/v1/sessions/{session_id}/cases/{case_id}",
                                  headers=AUTH)).json()
        detail = (await client.get(
            f"/v1/sessions/{session_id}/screenings/{screening_id(record)}", headers=AUTH)).json()
        dossier = (await client.get(
            f"/v1/sessions/{session_id}/runs/{after['runs'][-1]['run_id']}/dossier",
            headers=AUTH)).json()
        # AG-UI: the case board's "what changed / what was kept" for each turn.
        changes = []
        for run in after["runs"]:
            response = await client.get(
                f"/v1/sessions/{session_id}/cases/{case_id}/runs/{run['run_id']}/changes",
                headers=AUTH)
            assert response.status_code == 200, response.text
            CaseRunChanges.model_validate(response.json())
            changes.append(response.json())
        missing = await client.get(
            f"/v1/sessions/{session_id}/cases/{case_id}/runs/run_{'0' * 32}/changes",
            headers=AUTH)

    # What changed: the case gained the researcher's value, typed, and a
    # revision; the second turn saw it.
    assert after["revision"] > before["revision"]
    (context,) = after["context"]
    assert context["key"] == "measured_pIC50" and context["value"] == "6.2"
    assert context["unit"] == "pIC50" and context["source"] == "researcher"
    assert context["scope"] == {"target_id": "EGFR", "experimental_context": "IC50|wild_type"}
    assert context["actor"] == "user"
    assert json.dumps(dossier).count("measured_pIC50") >= 1
    # What did not: the estimate, its interval, the policy decision and the
    # stage statuses are the frozen artifact's, unchanged.
    first, second = (reads["get_activity_snapshot"]["model_view"] for reads in seen["reads"])
    assert first["estimate"] == second["estimate"]
    assert detail["activity_estimate"]["estimate"]["predicted_pic50"] == 7.1
    assert detail["activity_estimate"]["estimate"]["prediction_interval"] == [6.8, 7.4]
    assert detail["prioritisation_decision"]["status"] == "review"
    assert detail["stages"]["toxicity"]["status"] == "not_run"
    assert len(after["runs"]) == 2 and after["case_id"] == before["case_id"]

    # The board's diff says the same, from the case's own log: the second turn
    # is where the researcher's value is new, the subject is the case's, and
    # every artifact the second run was given has the first run's hash.
    first_turn, second_turn = changes
    assert first_turn["previous_run_id"] is None
    assert first_turn["artifacts"]["compared"] is False
    assert second_turn["previous_run_id"] == after["runs"][0]["run_id"]
    assert second_turn["from_revision"] < second_turn["to_revision"] == after["revision"]
    assert [c["key"] for c in second_turn["context"]["added"]] == ["measured_pIC50"]
    assert second_turn["context"]["added"][0]["actor"] == "user"
    assert second_turn["updates_by_actor"]["user"] >= 1
    assert second_turn["subject"] == {"subject_key": after["subject_key"], "kept": True}
    artifacts = second_turn["artifacts"]
    assert artifacts["compared"] is True
    assert artifacts["changed"] == [] and artifacts["added"] == [] and artifacts["removed"] == []
    # The frozen screening record -- estimate, interval, decision, stages --
    # by its content hash.
    assert [a["kind"] for a in artifacts["kept"]] == ["screening"]
    assert artifacts["kept"][0]["content_sha256"]
    assert all(not section["removed"] for section in (
        second_turn[name] for name in ("context", "hypotheses", "evidence", "uncertainties",
                                       "next_tests")))
    assert missing.status_code == 404

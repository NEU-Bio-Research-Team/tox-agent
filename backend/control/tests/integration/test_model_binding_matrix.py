"""The model a run was configured with is the model every step uses.

I08–I11 were four leaks of one invariant. Each entry point resolved the
predictor binding independently, so a session that had chosen model B could
still get model A's numbers or, worse, B's probability paired with A's
explanation while the bundle claimed both were about B.

With one admitted model per endpoint none of this was visible, which is why it
survived: every path agreed by accident. These tests give the stub predictor
two models and assert the binding through each path separately — a request for
B must reach the predictor as B, and must be persisted as B.
"""
from __future__ import annotations

import base64

import pytest

from toxagent.persistence.object_store import InMemoryObjectStore
from tests.support.api import AUTH, api_client, wait_for_run
from tests.support.ocr import stub_success
from tests.support.predictor import ASPIRIN, StubPredictor

PNG_BYTES = b"\x89PNG\r\n\x1a\nmodel-binding-matrix"

pytestmark = pytest.mark.anyio

MODEL_B = "herg-chemberta-v2"


def _sent_to(predictor: StubPredictor, path: str) -> list[dict]:
    return [call["body"] for call in predictor.requests if call["path"] == path]


async def _new_session(client) -> str:
    response = await client.post("/v1/sessions", json={}, headers=AUTH)
    assert response.status_code == 201, response.text
    return response.json()["session_id"]


async def _analysis_of(client, session_id: str) -> dict:
    """The session's active analysis, as the UI reads it."""
    session = (await client.get(f"/v1/sessions/{session_id}", headers=AUTH)).json()
    analysis = session.get("active_analysis")
    assert analysis is not None, session
    return analysis


# --- the deterministic lane --------------------------------------------------

async def test_a_chosen_model_reaches_the_predictor_and_is_persisted(db):
    predictor = StubPredictor()
    async with api_client(db, predictor) as client:
        session_id = await _new_session(client)
        submitted = await client.post(
            f"/v1/sessions/{session_id}/messages",
            json={
                "molecule": {"smiles": ASPIRIN},
                "analysis_options": {"endpoints": ["herg"], "model_selection": {"herg": MODEL_B}},
            },
            headers=AUTH,
        )
        assert submitted.status_code == 202, submitted.text
        await wait_for_run(client, session_id, submitted.json()["run_id"])

        sent = _sent_to(predictor, "/v1/predictions")
        assert sent and sent[-1]["model_selection"] == {"herg": MODEL_B}

        analysis = await _analysis_of(client, session_id)
        assert analysis["sections"]["herg"]["model_id"] == MODEL_B


# --- explanation follows the prediction (I11) --------------------------------

async def test_a_required_explanation_is_computed_with_the_model_that_predicted(db):
    """The bug: predict took model_selection, explain took only the molecule.

    So the bundle could hold B's probability next to A's attribution, with
    nothing in the record saying they came from different models.
    """
    predictor = StubPredictor()
    async with api_client(db, predictor) as client:
        session_id = await _new_session(client)
        submitted = await client.post(
            f"/v1/sessions/{session_id}/messages",
            json={
                "molecule": {"smiles": ASPIRIN},
                "analysis_options": {
                    "endpoints": ["herg"],
                    "model_selection": {"herg": MODEL_B},
                    "explanation_mode": "required",
                    "explanation_targets": [{"endpoint": "herg"}],
                },
            },
            headers=AUTH,
        )
        assert submitted.status_code == 202, submitted.text
        await wait_for_run(client, session_id, submitted.json()["run_id"])

        explanations = _sent_to(predictor, "/v1/explanations")
        assert explanations, "a required explanation made no explain call"
        assert explanations[-1]["model_id"] == MODEL_B


async def test_the_explanation_follows_the_model_that_answered_not_the_one_requested(db):
    """When the predictor resolves something else, the record wins.

    `model_selection` is a request; `predictions.<endpoint>.model_id` is what
    answered. An explanation has to be about the latter.
    """
    predictor = StubPredictor()
    async with api_client(db, predictor) as client:
        session_id = await _new_session(client)
        submitted = await client.post(
            f"/v1/sessions/{session_id}/messages",
            json={
                "molecule": {"smiles": ASPIRIN},
                "analysis_options": {
                    "endpoints": ["herg"],
                    "explanation_mode": "required",
                    "explanation_targets": [{"endpoint": "herg"}],
                },
            },
            headers=AUTH,
        )
        await wait_for_run(client, session_id, submitted.json()["run_id"])

        analysis = await _analysis_of(client, session_id)
        answered = analysis["sections"]["herg"]["model_id"]
        explanations = _sent_to(predictor, "/v1/explanations")
        assert explanations[-1]["model_id"] == answered


# --- OCR keeps the configuration (I09) ---------------------------------------

async def test_an_image_keeps_the_endpoints_and_model_the_request_chose(db):
    """An image is an input, not a different product.

    Only endpoints and threshold overrides used to survive recognition, so the
    one input a user cannot type fell back to the defaults.
    """
    predictor = StubPredictor()
    async with api_client(
        db, predictor,
        ocr_client=stub_success(ASPIRIN),
        object_store=InMemoryObjectStore(),
    ) as client:
        session_id = await _new_session(client)
        submitted = await client.post(
            f"/v1/sessions/{session_id}/messages",
            json={
                "image": {
                    "mime_type": "image/png",
                    "data_base64": base64.b64encode(PNG_BYTES).decode(),
                },
                "analysis_options": {"endpoints": ["herg"], "model_selection": {"herg": MODEL_B}},
            },
            headers=AUTH,
        )
        assert submitted.status_code == 202, submitted.text
        await wait_for_run(client, session_id, submitted.json()["run_id"])

        sent = _sent_to(predictor, "/v1/predictions")
        assert sent, "recognition never reached a prediction"
        assert sent[-1]["model_selection"] == {"herg": MODEL_B}
        assert sent[-1]["endpoints"] == ["herg"]


# --- what must never happen ---------------------------------------------------

async def test_no_request_silently_falls_back_to_a_different_model(db):
    """Across every predictor call in one run, exactly one model is named."""
    predictor = StubPredictor()
    async with api_client(db, predictor) as client:
        session_id = await _new_session(client)
        submitted = await client.post(
            f"/v1/sessions/{session_id}/messages",
            json={
                "molecule": {"smiles": ASPIRIN},
                "analysis_options": {
                    "endpoints": ["herg"],
                    "model_selection": {"herg": MODEL_B},
                    "explanation_mode": "required",
                    "explanation_targets": [{"endpoint": "herg"}],
                },
            },
            headers=AUTH,
        )
        await wait_for_run(client, session_id, submitted.json()["run_id"])

        named = set()
        for call in predictor.requests:
            body = call["body"]
            if "model_selection" in body and body["model_selection"]:
                named.update(body["model_selection"].values())
            if body.get("model_id"):
                named.add(body["model_id"])
        assert named == {MODEL_B}, named


# --- the binding is pinned at admission --------------------------------------

async def test_changing_the_session_setting_does_not_change_a_run_already_accepted(db):
    """A run's configuration is a snapshot, not a live lookup.

    submit_message merges the session's saved bindings with any per-message
    override and writes the result into run_configuration_snapshots at
    admission. Everything downstream reads that row, so a setting changed
    while a run is in flight — or years later, when the run is being audited —
    cannot rewrite what that run actually did.
    """
    predictor = StubPredictor()
    async with api_client(db, predictor) as client:
        session_id = await _new_session(client)

        saved = await client.patch(
            f"/v1/sessions/{session_id}/settings",
            json={"ai_profile_id": None, "predictor_bindings": {"herg": MODEL_B}},
            headers=AUTH,
        )
        assert saved.status_code == 200, saved.text

        submitted = await client.post(
            f"/v1/sessions/{session_id}/messages",
            json={"molecule": {"smiles": ASPIRIN}, "analysis_options": {"endpoints": ["herg"]}},
            headers=AUTH,
        )
        run_id = submitted.json()["run_id"]
        await wait_for_run(client, session_id, run_id)

        # Change the session after the run is done, as a later message would.
        await client.patch(
            f"/v1/sessions/{session_id}/settings",
            json={"ai_profile_id": None, "predictor_bindings": {"herg": "some-other-model"}},
            headers=AUTH,
        )

        run = (await client.get(f"/v1/sessions/{session_id}/runs/{run_id}", headers=AUTH)).json()
        snapshot = run.get("configuration_snapshot") or {}
        assert snapshot.get("predictor_bindings") == {"herg": MODEL_B}, run
        # And what actually ran matches the snapshot, not the new setting.
        assert _sent_to(predictor, "/v1/predictions")[-1]["model_selection"] == {"herg": MODEL_B}


async def test_a_per_message_selection_overrides_the_session_for_that_run_only(db):
    predictor = StubPredictor()
    async with api_client(db, predictor) as client:
        session_id = await _new_session(client)
        await client.patch(
            f"/v1/sessions/{session_id}/settings",
            json={"ai_profile_id": None, "predictor_bindings": {"herg": "session-model"}},
            headers=AUTH,
        )
        submitted = await client.post(
            f"/v1/sessions/{session_id}/messages",
            json={
                "molecule": {"smiles": ASPIRIN},
                "analysis_options": {"endpoints": ["herg"], "model_selection": {"herg": MODEL_B}},
            },
            headers=AUTH,
        )
        await wait_for_run(client, session_id, submitted.json()["run_id"])
        assert _sent_to(predictor, "/v1/predictions")[-1]["model_selection"] == {"herg": MODEL_B}

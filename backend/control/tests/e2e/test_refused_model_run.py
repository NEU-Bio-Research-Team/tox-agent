"""A provider that refuses the model ends the run with a code that says so.

The production failure: ChatGPT's Codex backend answered "The 'gpt-5.4-mini'
model is not supported when using Codex with a ChatGPT account." and the run
ended ``runtime_protocol_error`` -- "the runtime turn failed" -- so the user
read it as a broken runtime and had nothing to act on.
"""
from __future__ import annotations

import pytest

from tests.e2e.test_scripted_runtime import _analyse_aspirin, _install_scripted_runtime, _new_session
from tests.support.api import AUTH, api_client, wait_for_run
from tests.support.predictor import StubPredictor

pytestmark = pytest.mark.anyio

REFUSAL = "The 'gpt-5.4-mini' model is not supported when using Codex with a ChatGPT account."


class ProviderRefused(Exception):
    def __init__(self, error: dict) -> None:
        super().__init__("provider refused the turn")
        self.runtime_error = error


async def _run_that_fails_with(db, error: dict):
    async def script(turn) -> None:
        raise ProviderRefused(error)

    async with api_client(db, StubPredictor()) as client:
        await _install_scripted_runtime(client, script)
        session_id = await _new_session(client)
        await _analyse_aspirin(client, session_id)
        submitted = await client.post(
            f"/v1/sessions/{session_id}/messages",
            json={"intent_hint": "ask_report", "content": [{"type": "text", "text": "Explain."}]},
            headers=AUTH,
        )
        run = await wait_for_run(client, session_id, submitted.json()["run_id"])
        messages = (await client.get(f"/v1/sessions/{session_id}/messages", headers=AUTH)).json()["messages"]
    return run, messages


def _error_parts(messages) -> list[dict]:
    return [
        part["content"] for message in messages for part in message["parts"] if part["type"] == "error"
    ]


async def test_a_refused_model_is_model_not_supported_with_the_providers_words(db):
    run, messages = await _run_that_fails_with(
        db, {"name": "APIError", "data": {"message": REFUSAL, "statusCode": 400}}
    )
    assert run["status"] == "failed" and run["failure_code"] == "model_not_supported"
    assert [part["code"] for part in _error_parts(messages)] == ["model_not_supported"]
    assert _error_parts(messages)[0]["message"] == REFUSAL


async def test_a_rejected_sign_in_is_its_own_failure(db):
    run, messages = await _run_that_fails_with(
        db, {"name": "APIError", "data": {"message": "Unauthorized", "statusCode": 401}}
    )
    assert run["failure_code"] == "ai_credential_rejected"
    assert [part["code"] for part in _error_parts(messages)] == ["ai_credential_rejected"]


async def test_any_other_runtime_failure_is_still_a_protocol_error(db):
    run, _ = await _run_that_fails_with(
        db, {"name": "APIError", "data": {"message": "upstream exploded", "statusCode": 500}}
    )
    assert run["failure_code"] == "runtime_protocol_error"

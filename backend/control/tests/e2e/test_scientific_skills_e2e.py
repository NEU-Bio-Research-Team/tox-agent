"""The three skill arms of the RETHINK §5.5 ablation, end to end (ADR 0012).

* off — no skill text anywhere, no skill tools;
* static — every offered skill composed into the prompt and recorded as read;
* dynamic — only names and descriptions in the prompt; a read through the
  closed tool is what records a skill as loaded, with its hash.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from toxagent.config import PACKAGE_ROOT
from toxagent.application.skill_catalog import load_catalog
from tests.e2e.test_scripted_runtime import _analyse, _install_scripted_runtime, _new_session
from tests.support.api import AUTH, api_client, wait_for_run
from tests.support.predictor import StubPredictor

pytestmark = pytest.mark.anyio

CATALOG = load_catalog(PACKAGE_ROOT / "agent_profiles")
CONFLICT = CATALOG.get("assess-conflicting-evidence")
ANSWER = {"schema_version": "grounded-answer-v2", "answer_markdown": "Noted.", "claims": [],
          "limitations": [], "evidence_relations": []}


async def _turn(client, session_id: str, text: str) -> dict:
    submitted = await client.post(
        f"/v1/sessions/{session_id}/messages",
        json={"intent_hint": "auto", "content": [{"type": "text", "text": text}]},
        headers=AUTH,
    )
    assert submitted.status_code == 202, submitted.text
    run = await wait_for_run(client, session_id, submitted.json()["run_id"])
    state = await client.get(
        f"/v1/sessions/{session_id}/runs/{run['run_id']}/decision-state", headers=AUTH
    )
    return {"run": run, "state": state.json()}


async def _run(db, script, *, static: bool = False) -> dict:
    async with api_client(db, StubPredictor()) as client:
        if static:
            # The static arm is a RuntimeSettings field (an evaluation setting
            # read from TOXAGENT_SCIENTIFIC_SKILLS_STATIC in deployment).
            app_settings = client.app.state.settings
            client.app.state.settings = replace(
                app_settings, runtime=replace(app_settings.runtime, scientific_skills_static=True)
            )
        await _install_scripted_runtime(client, script)
        session_id = await _new_session(client)
        await _analyse(client, session_id)
        return await _turn(client, session_id, "Two studies disagree about hERG — which is right?")


@pytest.fixture
def base_flags(monkeypatch):
    monkeypatch.setenv("TOXAGENT_FLAG_ANSWER_DRAFT_V2", "1")
    monkeypatch.setenv("TOXAGENT_FLAG_SCIENTIFIC_CASE_V1", "1")


async def test_the_off_arm_offers_nothing(db, base_flags):
    seen: dict = {}

    async def script(turn) -> None:
        seen["prompt"] = turn.system_prompt
        seen["read"] = await turn.call_tool("read_scientific_skill", {"skill_id": CONFLICT.skill_id})
        await turn.call_tool("submit_grounded_answer", ANSWER)

    result = await _run(db, script)
    assert CONFLICT.description not in seen["prompt"]
    assert seen["read"]["error"]["code"] == "tool_denied"
    assert result["state"]["skills"] == {}


async def test_the_static_arm_composes_and_records_every_offered_skill(db, base_flags):
    seen: dict = {}

    async def script(turn) -> None:
        seen["prompt"] = turn.system_prompt
        await turn.call_tool("submit_grounded_answer", ANSWER)

    result = await _run(db, script, static=True)
    assert CONFLICT.body in seen["prompt"]
    skills = result["state"]["skills"]
    assert skills["mode"] == "static"
    assert skills["loaded"] == skills["offered"]
    assert CONFLICT.pin() in skills["offered"]


async def test_the_dynamic_arm_indexes_then_records_what_was_read(db, base_flags, monkeypatch):
    monkeypatch.setenv("TOXAGENT_FLAG_SCIENTIFIC_SKILLS_V1", "1")
    seen: dict = {}

    async def script(turn) -> None:
        seen["prompt"] = turn.system_prompt
        seen["read"] = await turn.call_tool("read_scientific_skill", {"skill_id": CONFLICT.skill_id})
        seen["ref"] = await turn.call_tool("read_skill_reference", {
            "skill_id": CONFLICT.skill_id, "reference": "scope-comparison-checklist.md",
        })
        seen["missing"] = await turn.call_tool("read_scientific_skill", {"skill_id": "no-such-skill"})
        await turn.call_tool("submit_grounded_answer", ANSWER)

    result = await _run(db, script)
    assert CONFLICT.description in seen["prompt"]
    assert CONFLICT.body not in seen["prompt"]
    assert seen["read"]["status"] == "completed", seen["read"]
    assert seen["read"]["model_view"]["instructions"] == CONFLICT.body
    assert seen["read"]["model_view"]["content_sha256"] == CONFLICT.content_sha256
    assert seen["ref"]["model_view"]["content"].startswith("# Scope comparison checklist")
    assert "no scientific skill 'no-such-skill'" in seen["missing"]["error"]["message"]
    skills = result["state"]["skills"]
    assert skills["mode"] == "dynamic"
    assert len(skills["offered"]) == 3
    assert skills["loaded"] == [CONFLICT.pin()]
    assert skills["references_loaded"] == [
        {**CONFLICT.pin(), "reference": "scope-comparison-checklist.md"}
    ]


async def test_without_the_case_tools_no_skill_is_offered_even_when_dynamic(db, monkeypatch):
    """Every shipped skill writes to the case; with the case off it cannot be used,
    so it is not offered and cannot be read."""
    monkeypatch.setenv("TOXAGENT_FLAG_ANSWER_DRAFT_V2", "1")
    monkeypatch.setenv("TOXAGENT_FLAG_SCIENTIFIC_SKILLS_V1", "1")
    seen: dict = {}

    async def script(turn) -> None:
        seen["prompt"] = turn.system_prompt
        seen["read"] = await turn.call_tool("read_scientific_skill", {"skill_id": CONFLICT.skill_id})
        await turn.call_tool("submit_grounded_answer", ANSWER)

    result = await _run(db, script)
    assert CONFLICT.description not in seen["prompt"]
    assert "no scientific skill" in seen["read"]["error"]["message"]
    assert result["state"]["skills"] == {}

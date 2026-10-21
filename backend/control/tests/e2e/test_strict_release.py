"""#47 under strict-reviewed-v1: only what the review cleared is published.

The release gate used to release every candidate and label it. Under a
declared policy the server decides, records a run result, and a candidate the
review did not clear never reaches the assistant message, the answer route or
the case's trusted conclusion. Each disposition the policy names is run here
on the scripted runtime, plus the declared no-review ablation.
"""
from __future__ import annotations

import pytest

from tests.e2e.test_release_gate import (
    OVERREACH,
    REPAIR_MARKER,
    SUPPORTED,
    _draft,
    _script,
)
from tests.e2e.test_scientific_case_e2e import _post
from tests.e2e.test_scripted_runtime import _analyse, _install_scripted_runtime, _new_session
from tests.support.api import AUTH, api_client, wait_for_run
from tests.support.predictor import StubPredictor
from tests.support.research import ACCEPTED_HIT, StubResearchProvider

pytestmark = pytest.mark.anyio


@pytest.fixture
def strict(monkeypatch):
    monkeypatch.setenv("TOXAGENT_FLAG_ANSWER_DRAFT_V2", "1")
    monkeypatch.setenv("TOXAGENT_FLAG_CLAIM_REVIEWER_V1", "1")
    monkeypatch.setenv("TOXAGENT_RELEASE_POLICY", "strict-reviewed-v1")


async def _run(db, state, script, *, case: bool = False, question: str | None = None) -> dict:
    provider = StubResearchProvider(hits=[ACCEPTED_HIT])
    async with api_client(db, StubPredictor(), research_provider=provider) as client:
        await _install_scripted_runtime(client, script)
        session_id = state.get("session_id") or await _new_session(client)
        state["session_id"] = session_id
        if "analysis_id" not in state:
            state["analysis_id"] = await _analyse(client, session_id)
        run = await _post(client, session_id,
                          question or "Should hERG stop us developing this compound?")
        result = (await client.get(f"/v1/sessions/{session_id}/runs/{run['run_id']}/result",
                                   headers=AUTH)).json()
        messages = (await client.get(f"/v1/sessions/{session_id}/messages",
                                     headers=AUTH)).json()["messages"]
        events = (await client.get(f"/v1/sessions/{session_id}/events:list",
                                   params={"limit": 500}, headers=AUTH)).json()["events"]
        candidates = {e["entity_id"] for e in events if e["type"] == "answer.candidate"}
        served = {}
        for answer_id in candidates:
            response = await client.get(f"/v1/sessions/{session_id}/answers/{answer_id}",
                                        headers=AUTH)
            served[answer_id] = response.status_code
        cases = (await client.get(f"/v1/sessions/{session_id}/cases", headers=AUTH)).json()
        case_body = None
        if case and cases.get("cases"):
            case_body = (await client.get(
                f"/v1/sessions/{session_id}/cases/{cases['cases'][0]['case_id']}", headers=AUTH,
            )).json()
    return {"run": run, "result": result, "reply": messages[-1], "events": events,
            "served": served, "case": case_body}


def _text(reply: dict) -> str:
    return "\n".join(p["content"].get("text", "") for p in reply["parts"] if p["type"] == "text")


def _part(reply: dict, kind: str) -> list[dict]:
    return [p["content"] for p in reply["parts"] if p["type"] == kind]


async def test_a_supported_answer_is_published(db, strict):
    state: dict = {}
    out = await _run(db, state, _script(
        state, rounds=[{"lit": "supported", "overreach": "supported"}], repair=None,
    ))
    result = out["result"]
    assert result["recorded"] is True
    assert result["disposition"] == "publish" and result["review_status"] == "passed"
    assert result["public_answer_id"] == result["candidate_answer_id"]
    assert _part(out["reply"], "answer_ref") == [{"answer_id": result["public_answer_id"]}]
    assert OVERREACH in _text(out["reply"])
    assert out["served"][result["public_answer_id"]] == 200
    assert _part(out["reply"], "result_ref")[0]["disposition"] == "publish"
    assert result["release_policy"] == "strict-reviewed-v1" and result["release_policy_sha256"]


async def test_a_repaired_and_supported_answer_is_published(db, strict):
    state: dict = {}
    out = await _run(db, state, _script(
        state, rounds=[{"lit": "supported", "overreach": "not_supported"},
                       {"lit": "supported"}],
        repair={"lit": SUPPORTED},
    ))
    result = out["result"]
    assert result["disposition"] == "publish"
    assert result["review_status"] == "repaired_and_passed"
    assert result["detail"]["repair"]["removed_claim_ids"] == ["overreach"]
    replaced = result["detail"]["replaced_answer_id"]
    assert out["served"][replaced] == 404 and out["served"][result["public_answer_id"]] == 200


async def test_a_claim_still_unsupported_after_the_repair_is_withheld(db, strict):
    state: dict = {}
    out = await _run(db, state, _script(
        state, rounds=[{"lit": "supported", "overreach": "not_supported"},
                       {"lit": "supported", "overreach": "not_supported"}],
        repair={"lit": SUPPORTED, "overreach": OVERREACH},
    ))
    result = out["result"]
    assert out["run"]["status"] == "completed"
    assert result["disposition"] == "partial_facts" and result["review_status"] == "blocked"
    assert result["public_answer_id"] is None
    assert len(result["detail"]["unresolved_claim_ids"]) == 1
    # Neither candidate reaches a reader: not the message, not the answer route.
    text = _text(out["reply"])
    assert OVERREACH not in text and SUPPORTED not in text
    assert "could not be released" in text and "audit view" in text
    assert _part(out["reply"], "answer_ref") == []
    assert set(out["served"].values()) == {404}
    assert not [e for e in out["events"] if e["type"] == "answer.accepted"]
    recorded = [e for e in out["events"] if e["type"] == "run.result_recorded"]
    assert [e["payload"]["disposition"] for e in recorded] == ["partial_facts"]


@pytest.mark.parametrize("rounds, repair, status", [
    ([{"lit": "supported", "overreach": "cannot_judge"}], None, "incomplete"),
    ([None], None, "not_completed"),
    ([{"lit": "supported", "overreach": "not_supported"}], None, "blocked"),
])
async def test_what_the_review_did_not_clear_is_never_published(db, strict, rounds, repair,
                                                                 status):
    state: dict = {}
    out = await _run(db, state, _script(state, rounds=rounds, repair=repair))
    result = out["result"]
    assert result["disposition"] == "partial_facts"
    assert result["review_status"] == status
    assert OVERREACH not in _text(out["reply"])
    assert set(out["served"].values()) == {404}


async def test_an_exception_in_the_turn_withholds_with_a_typed_cause(db, strict):
    async def script(turn) -> None:
        raise RuntimeError("the runtime crashed")

    out = await _run(db, {}, script)
    assert out["run"]["status"] == "failed"
    result = out["result"]
    assert result["disposition"] == "withhold"
    assert result["execution_outcome"] == "runtime_failed"
    assert result["public_answer_id"] is None


async def test_hard_only_publishes_without_a_review_by_design(db, monkeypatch):
    monkeypatch.setenv("TOXAGENT_FLAG_ANSWER_DRAFT_V2", "1")
    monkeypatch.setenv("TOXAGENT_RELEASE_POLICY", "hard-only-v1")
    state: dict = {}
    out = await _run(db, state, _script(state, rounds=[], repair=None))
    result = out["result"]
    assert result["disposition"] == "publish"
    assert result["review_status"] == "not_reviewed_by_design"
    assert "review_rounds" not in state


async def test_a_withheld_run_proposes_its_conclusion_and_never_trusts_it(db, strict,
                                                                           monkeypatch):
    monkeypatch.setenv("TOXAGENT_FLAG_SCIENTIFIC_CASE_V1", "1")
    state: dict = {}
    blocked = {"lit": "supported", "overreach": "not_supported"}
    inner = _script(state, rounds=[blocked, blocked],
                    repair={"lit": SUPPORTED, "overreach": OVERREACH})

    async def script(turn) -> None:
        if turn.tool_context.profile == "decision_support" \
                and REPAIR_MARKER not in turn.system_prompt and "set" not in state:
            read = await turn.call_tool("get_analysis_slice", {
                "analysis_id": state["analysis_id"], "section": "herg",
            })
            ledger = await turn.call_tool("update_scientific_case", {"operations": [
                {"op": "record_evidence", "claim": "The predictor scored hERG.",
                 "source_class": "predictor_fact",
                 "source_ref": f"observation:{read['observation_ids'][0]}",
                 "stance": "contextual"},
            ]})
            state["ledger"] = ledger
            entry = ledger["canonical"]["evidence"][-1]["id"]
            state["set"] = await turn.call_tool("update_scientific_case", {"operations": [
                {"op": "set_conclusion",
                 "can_say": [{"text": "The predictor flags hERG.", "evidence_ids": [entry]}]},
            ]})
        await inner(turn)

    out = await _run(db, state, script, case=True)
    assert state["set"]["status"] == "completed", state["set"]
    case = out["case"]
    assert out["result"]["disposition"] == "partial_facts"
    assert case["conclusion"]["status"] == "proposed"
    (run,) = case["runs"]
    assert run["disposition"] == "partial_facts" and run["answer_id"] is None


async def test_duplicate_completion_records_one_result(db, strict):
    from toxagent.domain.answer_release import ReleaseOutcome
    from toxagent.harness.gateway.release import declared_policy

    state: dict = {}
    out = await _run(db, state, _script(
        state, rounds=[{"lit": "supported", "overreach": "supported"}], repair=None,
    ))
    provider = StubResearchProvider(hits=[ACCEPTED_HIT])
    async with api_client(db, StubPredictor(), research_provider=provider) as client:
        await _install_scripted_runtime(client, _script(state, rounds=[], repair=None))
        gateway = client.app.state.runtime_gateway
        from toxagent.application.policy import Actor
        from toxagent.application.runs.scheduler import RunContext
        from toxagent.domain.run import Intent

        context = RunContext(actor=Actor(subject_id="user-1"), session_id=state["session_id"],
                             run_id=out["run"]["run_id"], intent=Intent.DECISION_SUPPORT)
        answer = await gateway._latest_answer(context)
        await gateway._record_publication(
            context, answer, policy=declared_policy(), outcome=ReleaseOutcome.REVIEWED,
            reviews=(), repair=None, replaced=None, ungated_reason=None, unresolved=(),
        )
        messages = (await client.get(f"/v1/sessions/{state['session_id']}/messages",
                                     headers=AUTH)).json()["messages"]
        async with client.app.state.database.unit_of_work() as uow:
            stored = await uow.run_results.get(out["run"]["run_id"])
    assert stored.disposition.value == "publish"
    assistant = [m for m in messages if m["role"] == "assistant"
                 and any(p["type"] == "result_ref" for p in m["parts"])]
    assert len(assistant) == 1


async def test_cancelling_a_run_in_review_never_releases_its_candidate(db, strict):
    """§9.4: a cancellation ends the remaining dispatch and does not release
    a candidate that was still under review. What the run spent is still
    recorded, and the reader is told the result was withheld -- not shown the
    draft.
    """
    import asyncio

    state: dict = {}

    async def script(turn) -> None:
        if turn.tool_context.profile == "claim_review":
            # The reviewer is dispatched and never answers: the test cancels
            # the run while this turn is in flight. The flag is a plain dict
            # entry, polled by the test -- the run executes on the app's own
            # loop, so an asyncio.Event set here would not wake the test.
            state["in_review"] = True
            await asyncio.sleep(30)
            return
        found = await turn.call_tool("search_toxicology_evidence", {
            "analysis_id": state["analysis_id"], "query": "hERG blockade", "limit": 3,
        })
        evidence_id = found["model_view"]["results"][0]["evidence_id"]
        await turn.call_tool("get_evidence_record", {"evidence_id": evidence_id})
        state["submitted"] = await turn.call_tool(
            "submit_grounded_answer", _draft(evidence_id, {"lit": SUPPORTED})
        )

    provider = StubResearchProvider(hits=[ACCEPTED_HIT])
    async with api_client(db, StubPredictor(), research_provider=provider) as client:
        await _install_scripted_runtime(client, script)
        session_id = await _new_session(client)
        state["analysis_id"] = await _analyse(client, session_id)
        submitted = await client.post(
            f"/v1/sessions/{session_id}/messages", headers=AUTH,
            json={"intent_hint": "auto",
                  "content": [{"type": "text", "text": "Should hERG stop us developing this?"}]},
        )
        assert submitted.status_code == 202, submitted.text
        run_id = submitted.json()["run_id"]
        for _ in range(100):
            if state.get("in_review"):
                break
            # Polling the run is what a client does, and it is what steps the
            # in-process run to its reviewer dispatch.
            await client.get(f"/v1/sessions/{session_id}/runs/{run_id}", headers=AUTH)
            await asyncio.sleep(0.1)
        assert state.get("in_review"), "the reviewer was never dispatched"
        cancelled = await client.post(f"/v1/sessions/{session_id}/runs/{run_id}:cancel",
                                      headers=AUTH)
        assert cancelled.status_code == 200, cancelled.text
        run = await wait_for_run(client, session_id, run_id)
        result = (await client.get(f"/v1/sessions/{session_id}/runs/{run_id}/result",
                                   headers=AUTH)).json()
        detail = (await client.get(f"/v1/sessions/{session_id}/runs/{run_id}",
                                   headers=AUTH)).json()
        drafts = (await client.get(f"/v1/sessions/{session_id}/runs/{run_id}/answer-drafts",
                                   headers=AUTH)).json()["drafts"]
        messages = (await client.get(f"/v1/sessions/{session_id}/messages",
                                     headers=AUTH)).json()["messages"]
        candidate_id = result["candidate_answer_id"]
        served = await client.get(f"/v1/sessions/{session_id}/answers/{candidate_id}",
                                  headers=AUTH)

    assert run["status"] == "cancelled"
    # The candidate was hard-valid and stored, and is not published.
    assert state["submitted"]["status"] == "completed", state["submitted"]
    assert candidate_id and result["public_answer_id"] is None
    assert result["disposition"] == "withhold"
    assert result["review_status"] == "not_completed"
    assert result["execution_outcome"] == "cancelled"
    assert result["knowledge_outcome"] == "none"
    assert result["detail"]["failure_code"] == "cancelled"
    # Not through the answer route, and not as prose in the transcript.
    assert served.status_code == 404
    assistant = [m for m in messages if m["role"] == "assistant"]
    text = "\n".join(p["content"].get("text", "") for m in assistant for p in m["parts"]
                     if p["type"] == "text")
    assert SUPPORTED not in text
    # What it spent is kept: the draft it submitted, its tool calls and the
    # phase they were made in.
    assert [d["outcome"] for d in drafts] == ["candidate"]
    phases = detail["phase_usage"]["phases"]
    assert phases["investigator"]["tool_calls"]["completed"] >= 3
    assert "review_initial" in phases  # the dispatch it was cancelled in

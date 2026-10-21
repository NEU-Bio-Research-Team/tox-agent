"""#47 through the product: an answer is reviewed before it is released.

With ``release_gate_v1`` an accepted answer is a candidate. The independent
reviewer judges it; a blocking verdict sends it back for one repair; the
repaired answer is validated and reviewed again; and only then is an answer
released -- with a record of how. Each outcome the gate's policy names is run
here on the scripted runtime, plus the two things that must not change: a
second answer outside a repair is still refused, and without the gate the
reviewer still only observes.
"""
from __future__ import annotations

import json
import re

import pytest

from tests.e2e.test_scientific_case_e2e import _post
from tests.e2e.test_scripted_runtime import _analyse, _install_scripted_runtime, _new_session
from tests.support.api import AUTH, api_client
from tests.support.predictor import StubPredictor
from tests.support.research import ACCEPTED_HIT, StubResearchProvider

pytestmark = pytest.mark.anyio

_BUNDLE = re.compile(r"```json\n(.*?)\n```", re.DOTALL)
SUPPORTED = "A retrieved study reports hERG channel blockade in a screened series."
OVERREACH = "The retrieved study shows that this exact compound blocks hERG."
NARROWED = "A retrieved study reports hERG blockade in a series this compound may belong to."
REPAIR_MARKER = "Repair it once"


@pytest.fixture
def gate_on(monkeypatch):
    monkeypatch.setenv("TOXAGENT_FLAG_ANSWER_DRAFT_V2", "1")
    monkeypatch.setenv("TOXAGENT_FLAG_CLAIM_REVIEWER_V1", "1")
    monkeypatch.setenv("TOXAGENT_FLAG_RELEASE_GATE_V1", "1")


def _draft(evidence_id: str, claims: dict[str, str]) -> dict:
    return {
        "schema_version": "grounded-answer-v2",
        "answer_markdown": " ".join(claims.values()),
        "claims": [
            {"local_ref": ref, "kind": "scientific", "text": text, "citation_ids": [evidence_id]}
            for ref, text in claims.items()
        ],
        "limitations": [{"code": "evidence_scope_limited", "text": ""}],
    }


def _script(state: dict, *, rounds: list, repair: dict[str, str] | None,
            submit_twice: bool = False):
    """``rounds``: per review round, a verdict per local_ref (or None for a
    reviewer that submits nothing). ``repair``: the claims the repair turn
    submits, or None for a repair turn that submits nothing."""

    async def script(turn) -> None:
        profile = turn.tool_context.profile
        if profile == "claim_review":
            index = state["review_rounds"] = state.get("review_rounds", 0) + 1
            bundle = json.loads(_BUNDLE.search(turn.system_prompt).group(1))
            state.setdefault("bundles", []).append(bundle)
            verdicts = rounds[index - 1]
            if verdicts is None:
                return
            by_text = {text: ref for ref, text in state["texts"].items()}
            await turn.call_tool("submit_claim_review", {"reviews": [
                {"claim_id": claim["claim_id"],
                 "verdict": verdicts[by_text[claim["text"]]],
                 "reason": "judged from the abstract"}
                for claim in bundle["claims"]
            ]})
            return
        if REPAIR_MARKER in turn.system_prompt:
            state["repair_prompt"] = turn.system_prompt
            if repair is not None:
                state["texts"].update(repair)
                state["repair_result"] = await turn.call_tool(
                    "submit_grounded_answer", _draft(state["evidence_id"], repair),
                )
            return
        found = await turn.call_tool("search_toxicology_evidence", {
            "analysis_id": state["analysis_id"], "query": "hERG blockade", "limit": 3,
        })
        evidence_id = state["evidence_id"] = found["model_view"]["results"][0]["evidence_id"]
        await turn.call_tool("get_evidence_record", {"evidence_id": evidence_id})
        claims = {"lit": SUPPORTED, "overreach": OVERREACH}
        state["texts"] = dict(claims)
        state["first_result"] = await turn.call_tool(
            "submit_grounded_answer", _draft(evidence_id, claims)
        )
        if submit_twice:
            state["second_result"] = await turn.call_tool(
                "submit_grounded_answer", _draft(evidence_id, {"lit": SUPPORTED})
            )

    return script


async def _run(db, state, script) -> dict:
    provider = StubResearchProvider(hits=[ACCEPTED_HIT])
    async with api_client(db, StubPredictor(), research_provider=provider) as client:
        await _install_scripted_runtime(client, script)
        session_id = await _new_session(client)
        state["analysis_id"] = await _analyse(client, session_id)
        run = await _post(client, session_id, "Should hERG stop us developing this compound?")
        events = (await client.get(f"/v1/sessions/{session_id}/events:list",
                                   params={"limit": 500}, headers=AUTH)).json()["events"]
        messages = (await client.get(f"/v1/sessions/{session_id}/messages",
                                     headers=AUTH)).json()["messages"]
        answer_ids = [p["content"]["answer_id"] for p in messages[-1]["parts"]
                      if p["type"] == "answer_ref"]
        served = {}
        candidate_ids = {e["entity_id"] for e in events if e["type"] == "answer.candidate"}
        for answer_id in candidate_ids | set(answer_ids):
            response = await client.get(f"/v1/sessions/{session_id}/answers/{answer_id}",
                                        headers=AUTH)
            served[answer_id] = response.json() if response.status_code == 200 else None
        drafts = (await client.get(
            f"/v1/sessions/{session_id}/runs/{run['run_id']}/answer-drafts", headers=AUTH,
        )).json()["drafts"]
    return {"run": run, "events": events, "answer_ids": answer_ids, "served": served,
            "drafts": drafts}


def _released(result) -> dict:
    (answer_id,) = result["answer_ids"]
    answer = result["served"][answer_id]
    assert answer is not None, "the released answer is served"
    return answer


async def test_a_supported_answer_is_released_as_reviewed(db, gate_on):
    state: dict = {}
    supported = {"lit": "supported", "overreach": "supported"}
    result = await _run(db, state, _script(state, rounds=[supported], repair=None))

    assert result["run"]["status"] == "completed", result["run"]
    answer = _released(result)
    assert answer["release"]["outcome"] == "reviewed"
    assert answer["release"]["semantically_passed"] is True
    assert "repair_prompt" not in state
    assert state["first_result"]["model_view"]["release"] == "pending_review"

    # Announced as a candidate first, and as the run's answer only after review.
    types = [(e["type"], e["sequence"]) for e in result["events"]]
    candidate = next(seq for t, seq in types if t == "answer.candidate")
    accepted = next(seq for t, seq in types if t == "answer.accepted")
    reviewed = max(seq for t, seq in types if t == "tool.completed")
    assert candidate < reviewed < accepted


async def test_an_unsupported_claim_is_repaired_once_and_reviewed_again(db, gate_on):
    state: dict = {}
    result = await _run(db, state, _script(
        state,
        rounds=[{"lit": "supported", "overreach": "not_supported"},
                {"lit": "supported"}],
        repair={"lit": SUPPORTED},
    ))

    assert result["run"]["status"] == "completed", result["run"]
    answer = _released(result)
    release = answer["release"]
    assert release["outcome"] == "repaired_and_reviewed"
    assert release["semantically_passed"] is True
    # The repair only deleted, and the lineage says which claim, by its label.
    assert release["repair"]["removed_claim_ids"] == ["overreach"]
    assert release["repair"]["reworded_claim_ids"] == []
    assert release["repair"]["removal_only"] is True
    assert [c["text"] for c in answer["claims"]] == [SUPPORTED]
    assert len(release["reviews"]) == 2

    # The repair turn was told what to fix, by local_ref.
    assert '"local_ref": "overreach"' in state["repair_prompt"]
    # The replaced candidate is not served; only the released answer is.
    replaced = release["replaced_answer_id"]
    assert result["served"][replaced] is None
    assert [d["outcome"] for d in result["drafts"]] == ["candidate", "candidate"]


async def test_a_narrowing_repair_is_a_rewording_not_a_deletion(db, gate_on):
    state: dict = {}
    result = await _run(db, state, _script(
        state,
        rounds=[{"lit": "supported", "overreach": "partially_supported"},
                {"lit": "supported", "overreach": "supported"}],
        repair={"lit": SUPPORTED, "overreach": NARROWED},
    ))
    release = _released(result)["release"]
    assert release["outcome"] == "repaired_and_reviewed"
    assert release["repair"]["reworded_claim_ids"] == ["overreach"]
    assert release["repair"]["removal_only"] is False


async def test_a_claim_still_unsupported_after_the_repair_is_released_and_named(db, gate_on):
    state: dict = {}
    result = await _run(db, state, _script(
        state,
        rounds=[{"lit": "supported", "overreach": "not_supported"},
                {"lit": "supported", "overreach": "not_supported"}],
        repair={"lit": SUPPORTED, "overreach": OVERREACH},
    ))
    answer = _released(result)
    release = answer["release"]
    assert release["outcome"] == "released_with_unresolved_review"
    assert release["semantically_passed"] is False
    (unresolved,) = release["unresolved_claim_ids"]
    assert next(c for c in answer["claims"] if c["claim_id"] == unresolved)["text"] == OVERREACH


async def test_a_repair_that_submits_nothing_releases_the_candidate_unresolved(db, gate_on):
    state: dict = {}
    result = await _run(db, state, _script(
        state, rounds=[{"lit": "supported", "overreach": "not_supported"}], repair=None,
    ))
    release = _released(result)["release"]
    assert release["outcome"] == "released_with_unresolved_review"
    assert release["ungated_reason"] == "the repair turn submitted no new answer"
    assert len(release["unresolved_claim_ids"]) == 1


async def test_cannot_judge_is_not_a_pass(db, gate_on):
    state: dict = {}
    result = await _run(db, state, _script(
        state, rounds=[{"lit": "supported", "overreach": "cannot_judge"}], repair=None,
    ))
    release = _released(result)["release"]
    assert release["outcome"] == "ungated"
    assert release["semantically_passed"] is False
    assert "not judged" in release["ungated_reason"]
    assert "repair_prompt" not in state


async def test_a_reviewer_that_submits_nothing_leaves_the_answer_ungated(db, gate_on):
    state: dict = {}
    result = await _run(db, state, _script(state, rounds=[None], repair=None))
    release = _released(result)["release"]
    assert release["outcome"] == "ungated"
    assert "did not complete" in release["ungated_reason"]


async def test_outside_a_repair_a_second_answer_is_still_refused(db, gate_on):
    state: dict = {}
    supported = {"lit": "supported", "overreach": "supported"}
    await _run(db, state, _script(state, rounds=[supported], repair=None, submit_twice=True))
    second = state["second_result"]
    assert second["status"] == "error"
    assert second["error"]["code"] == "conflict"


async def test_without_the_gate_the_reviewer_only_observes(db, monkeypatch):
    monkeypatch.setenv("TOXAGENT_FLAG_ANSWER_DRAFT_V2", "1")
    monkeypatch.setenv("TOXAGENT_FLAG_CLAIM_REVIEWER_V1", "1")
    state: dict = {}
    result = await _run(db, state, _script(
        state, rounds=[{"lit": "supported", "overreach": "not_supported"}], repair=None,
    ))
    answer = _released(result)
    assert "release" not in answer
    assert "repair_prompt" not in state
    assert [c["text"] for c in answer["claims"]] == [SUPPORTED, OVERREACH]
    assert not [e for e in result["events"] if e["type"] == "answer.candidate"]

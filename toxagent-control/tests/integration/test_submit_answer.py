"""The submit_grounded_answer workflow end to end (plan sections 8.4, 9, 9.5).

Exercised through the real ToolRunner and MCP-visible tool, the same path a
runtime would use, so this is also the Phase 2 exit gate: "candidate answer có
số sai không thể commit" and "candidate hợp lệ tạo complete claim-source graph".
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from toxagent.application.create_analysis import CreateAnalysis
from toxagent.application.policy import Actor
from toxagent.config import PolicySettings
from toxagent.domain.errors import Conflict
from toxagent.domain.events import EventType
from toxagent.domain.message import Message, Role
from toxagent.domain.run import Intent, Lane, Run
from toxagent.domain.session import Session
from toxagent.tools.bootstrap import build_registry
from toxagent.tools.registry import ToolContext
from toxagent.tools.runner import ToolRunner
from tests.support.predictor import ASPIRIN, StubPredictor

pytestmark = pytest.mark.anyio

NOW = datetime(2026, 9, 4, tzinfo=timezone.utc)
ACTOR = Actor(subject_id="user-1")


async def rig(db, stub: StubPredictor, *, max_candidates: int = 2, language: str = "en"):
    session = Session.create("user-1", now=NOW)
    message = Message.create(session.id, Role.USER, 1, now=NOW)
    run = Run.create(session.id, message.id, Lane.AGENTIC, Intent.REPORT_QA, now=NOW)
    async with db.unit_of_work() as uow:
        await uow.sessions.add(session)
        await uow.messages.add(message)
        await uow.runs.add(run)
        uow.emit(
            session_id=session.id, type=EventType.SESSION_CREATED,
            entity_type="session", entity_id=session.id,
        )
        await uow.commit()

    client = stub.client()
    policy = PolicySettings(max_answer_candidates_per_run=max_candidates)
    analysis_service = CreateAnalysis(db, client, policy)
    snapshot = await analysis_service.execute(
        actor=ACTOR, session_id=session.id, run_id=run.id, smiles=ASPIRIN, owns_run=False,
    )
    registry = build_registry(db, client, analysis_service, policy)
    runner = ToolRunner(registry, db)
    context = ToolContext(
        session_id=session.id, run_id=run.id, actor=ACTOR, profile="report_qa",
        deadline_at=datetime.now(timezone.utc) + timedelta(minutes=5), language=language,
    )
    return runner, context, session, run, snapshot.observation


def valid_candidate(observation) -> dict:
    return {
        "schema_version": "grounded-answer-v1",
        "answer_markdown": "Predicted hERG blockade probability is 0.731.",
        "claims": [
            {
                "claim_id": "clm_" + "1" * 32, "kind": "numeric",
                "text": "Predicted hERG blocker probability is 0.731.",
                "observation_id": observation.id,
                "field_path": "predictions.herg.probability_blocker",
                "source_value": 0.73064, "rendered_value": "0.731", "transform": "round:3",
            },
        ],
        "limitations": [
            {"code": "uncalibrated_probability", "text": ""},
        ],
        "recommended_next_steps": [],
    }


async def test_a_valid_candidate_is_accepted_with_a_complete_claim_graph(db):
    runner, context, session, run, observation = await rig(db, StubPredictor())
    result = await runner.call(context, "submit_grounded_answer", valid_candidate(observation))
    assert result["status"] == "completed"
    assert result["model_view"]["accepted"] is True
    assert result["model_view"]["is_fallback"] is False
    assert result["observation_ids"] == [observation.id]

    async with db.unit_of_work() as uow:
        answer = await uow.answers.get(result["model_view"]["answer_id"], session_id=session.id)
        assert answer is not None
        (claim,) = answer.claims
        assert claim.source_value == 0.73064
        assert claim.field_path == "predictions.herg.probability_blocker"


async def test_a_scientific_claim_citing_attribution_alone_needs_no_field_path(db):
    """Live sweep 2026-09-06 (progress log section 14.6): get_attribution's
    model_view hands the model an observation_id and top_tokens, never a
    field_path the observation itself resolves — so a scientific claim
    answering "which tokens drove this" had no field_path to name and no
    evidence citation to offer, and validate_basis rejected every attempt as
    claim_has_no_basis. An attribution observation is entirely about
    attribution, unlike an analysis observation mixing several fields, so
    citing it at all (observation_id alone) is a well-formed basis —
    answer_validator.py's has_observation_basis now says so."""
    runner, context, session, run, analysis_observation = await rig(db, StubPredictor())
    analysis_id = analysis_observation.provenance["analysis_id"]
    attributed = await runner.call(
        context, "get_attribution",
        {"analysis_id": analysis_id, "endpoint": "herg"},
    )
    assert attributed["status"] == "completed"
    (attribution_observation_id,) = attributed["observation_ids"]

    candidate = {
        "schema_version": "grounded-answer-v1",
        "answer_markdown": "The nitrogen-containing ring system contributed most to the hERG score.",
        "claims": [
            {
                "claim_id": "clm_" + "4" * 32, "kind": "scientific",
                "text": "The nitrogen-containing ring system contributed most to the hERG score.",
                "observation_id": attribution_observation_id,
            },
        ],
        "limitations": [
            {"code": "attribution_not_causality", "text": ""},
        ],
        "recommended_next_steps": [],
    }
    result = await runner.call(context, "submit_grounded_answer", candidate)
    assert result["status"] == "completed", result
    assert result["model_view"]["accepted"] is True
    assert result["model_view"]["is_fallback"] is False


async def test_a_claim_id_reused_from_an_unrelated_answer_is_a_correctable_violation(db):
    """Live sweep (2026-09-05): a model is told to "make one up" for
    claim_id (tools/definitions/answer.py) with no reason to expect it must
    also be unique against every other answer this deployment has ever
    stored — a low-entropy id ("clm_1111...1"-style) reused across two
    unrelated answers raised an unhandled sqlite3.IntegrityError on insert
    instead of a normal, correctable violation."""
    runner1, context1, _, _, observation1 = await rig(db, StubPredictor())
    first = await runner1.call(context1, "submit_grounded_answer", valid_candidate(observation1))
    assert first["status"] == "completed"

    runner2, context2, _, run2, observation2 = await rig(db, StubPredictor())
    colliding = valid_candidate(observation2)  # same "clm_" + "1" * 32 as the first answer
    result = await runner2.call(context2, "submit_grounded_answer", colliding)
    assert result["status"] == "error"
    assert result["error"]["code"] == "answer_validation_failed"
    violations = result["error"]["details"]["violations"]
    assert any(v["code"] == "claim_id_not_unique" for v in violations)

    async with db.unit_of_work() as uow:
        assert await uow.answers.get_for_run(run2.id) is None

    corrected = valid_candidate(observation2)
    corrected["claims"][0]["claim_id"] = "clm_" + "2" * 32
    retried = await runner2.call(context2, "submit_grounded_answer", corrected)
    assert retried["status"] == "completed"
    assert retried["model_view"]["is_fallback"] is False


async def test_a_wrong_number_cannot_be_committed(db):
    """Phase 2 exit gate, verbatim."""
    runner, context, session, run, observation = await rig(db, StubPredictor())
    candidate = valid_candidate(observation)
    candidate["claims"][0]["source_value"] = 0.5
    candidate["claims"][0]["rendered_value"] = "0.500"
    result = await runner.call(context, "submit_grounded_answer", candidate)
    assert result["status"] == "error"
    assert result["error"]["code"] == "answer_validation_failed"
    violations = result["error"]["details"]["violations"]
    assert any(v["code"] == "claim_source_value_mismatch" for v in violations)

    async with db.unit_of_work() as uow:
        assert await uow.answers.get_for_run(run.id) is None


async def test_a_rejected_first_candidate_tells_the_model_to_retry(db):
    """Progress log §4.6: a live Phase 3 run showed a model reading this
    error's ``retryable`` flag and simply not calling submit_grounded_answer a
    second time. The tool envelope must say this candidate is retryable and
    the message must be an instruction, not just a report."""
    runner, context, session, run, observation = await rig(db, StubPredictor())
    candidate = valid_candidate(observation)
    candidate["limitations"] = []
    result = await runner.call(context, "submit_grounded_answer", candidate)
    assert result["error"]["retryable"] is True
    assert "call submit_grounded_answer again" in result["error"]["message"]
    assert result["error"]["details"]["attempts_remaining"] == 1


async def test_a_missing_required_limitation_is_rejected(db):
    runner, context, session, run, observation = await rig(db, StubPredictor())
    candidate = valid_candidate(observation)
    candidate["limitations"] = []
    result = await runner.call(context, "submit_grounded_answer", candidate)
    assert result["error"]["code"] == "answer_validation_failed"
    violations = result["error"]["details"]["violations"]
    assert any(v["code"] == "missing_required_limitation" for v in violations)


async def test_a_safety_verdict_is_rejected_regardless_of_correct_numbers(db):
    runner, context, session, run, observation = await rig(db, StubPredictor())
    candidate = valid_candidate(observation)
    candidate["answer_markdown"] = "This compound is safe to use."
    result = await runner.call(context, "submit_grounded_answer", candidate)
    violations = result["error"]["details"]["violations"]
    assert any(v["code"] == "safety_verdict_out_of_scope" for v in violations)


async def test_a_malformed_claim_id_does_not_consume_a_candidate_generation(db):
    """Progress log §4.6: a live run submitted claim_id "c1" as its *final*
    candidate and lost its whole correction budget to a shape error. The wire
    validator now rejects it before submit_answer.execute() runs at all, so it
    is not recorded as a submit_grounded_answer tool call and the model still
    has both attempts."""
    runner, context, session, run, observation = await rig(db, StubPredictor())
    malformed = valid_candidate(observation)
    malformed["claims"][0]["claim_id"] = "c1"
    result = await runner.call(context, "submit_grounded_answer", malformed)
    assert result["status"] == "error"
    assert result["error"]["code"] == "invalid_request"

    good = valid_candidate(observation)
    second = await runner.call(context, "submit_grounded_answer", good)
    assert second["status"] == "completed"
    assert second["model_view"]["candidate_generation"] == 1  # still the first real attempt


async def test_a_correction_within_budget_can_still_be_accepted(db):
    runner, context, session, run, observation = await rig(db, StubPredictor())
    bad = valid_candidate(observation)
    bad["claims"][0]["source_value"] = 0.5
    first = await runner.call(context, "submit_grounded_answer", bad)
    assert first["status"] == "error"

    good = valid_candidate(observation)
    second = await runner.call(context, "submit_grounded_answer", good)
    assert second["status"] == "completed"
    assert second["model_view"]["candidate_generation"] == 2


async def test_exhausting_every_attempt_commits_a_deterministic_fallback(db):
    runner, context, session, run, observation = await rig(db, StubPredictor(), max_candidates=2)
    bad = valid_candidate(observation)
    bad["claims"][0]["source_value"] = 0.5

    first = await runner.call(context, "submit_grounded_answer", bad)
    assert first["status"] == "error"

    second = await runner.call(context, "submit_grounded_answer", bad)
    # The run now has an answer even though the model never produced a valid
    # one: the tool call succeeds, and what it committed is the fallback.
    assert second["status"] == "completed"
    assert second["model_view"]["is_fallback"] is True

    async with db.unit_of_work() as uow:
        answer = await uow.answers.get_for_run(run.id)
        assert answer is not None and answer.is_fallback
        assert answer.claims  # server-authored facts, not empty
        for claim in answer.claims:
            assert claim.observation_id == observation.id


async def test_the_fallback_message_is_localised(db):
    runner, context, session, run, observation = await rig(
        db, StubPredictor(), max_candidates=1, language="vi"
    )
    bad = valid_candidate(observation)
    bad["claims"][0]["source_value"] = 0.5
    result = await runner.call(context, "submit_grounded_answer", bad)
    assert result["model_view"]["is_fallback"] is True
    async with db.unit_of_work() as uow:
        answer = await uow.answers.get_for_run(run.id)
    assert "Tôi chưa thể" in answer.answer_markdown


async def test_an_accepted_answer_cannot_be_overwritten(db):
    runner, context, session, run, observation = await rig(db, StubPredictor())
    first = await runner.call(context, "submit_grounded_answer", valid_candidate(observation))
    assert first["status"] == "completed"

    second = await runner.call(context, "submit_grounded_answer", valid_candidate(observation))
    assert second["status"] == "error"
    assert second["error"]["code"] == "conflict"


async def test_an_unclaimed_number_in_the_prose_is_rejected(db):
    """audit_5_9.md A01: {"answer_markdown": "The hERG probability is 99.99%.",
    "claims": [], "limitations": []} used to be accepted outright."""
    runner, context, session, run, observation = await rig(db, StubPredictor())
    candidate = {
        "schema_version": "grounded-answer-v1",
        "answer_markdown": "The hERG probability is 99.99%.",
        "claims": [],
        "limitations": [],
    }
    result = await runner.call(context, "submit_grounded_answer", candidate)
    assert result["status"] == "error"
    violations = result["error"]["details"]["violations"]
    assert any(v["code"] == "unclaimed_numeric_value" for v in violations)

    async with db.unit_of_work() as uow:
        assert await uow.answers.get_for_run(run.id) is None


async def test_a_self_authored_link_is_rejected(db):
    runner, context, session, run, observation = await rig(db, StubPredictor())
    candidate = valid_candidate(observation)
    candidate["answer_markdown"] += " See [this paper](https://example.com/study) for more."
    result = await runner.call(context, "submit_grounded_answer", candidate)
    assert result["status"] == "error"
    violations = result["error"]["details"]["violations"]
    assert any(v["code"] == "raw_link_in_answer_markdown" for v in violations)


async def test_a_citation_from_another_session_is_refused(db):
    from toxagent.domain.evidence import EvidenceRecord, EvidenceStatus, SourceType

    runner, context, session, run, observation = await rig(db, StubPredictor())
    foreign_session = Session.create("user-2", now=NOW)
    foreign = EvidenceRecord.create(
        session_id=foreign_session.id, provider="europepmc", provider_record_id="PMC1",
        source_type=SourceType.ARTICLE, title="t", retrieved_at=NOW,
    ).to_status(EvidenceStatus.NORMALIZED).to_status(EvidenceStatus.ACCEPTED)
    async with db.unit_of_work() as uow:
        # SQLite leaves foreign-key enforcement off by default; make the
        # foreign session real so this authorization test is valid on the
        # PostgreSQL schema production actually enforces.
        await uow.sessions.add(foreign_session)
        await uow.evidence.add(foreign)
        await uow.commit()

    candidate = valid_candidate(observation)
    candidate["claims"].append(
        {
            "claim_id": "clm_" + "9" * 32, "kind": "scientific",
            "text": "An independent source reports a similar signal.",
            "citation_ids": [foreign.id],
        }
    )
    candidate["limitations"].append({"code": "evidence_scope_limited", "text": ""})
    result = await runner.call(context, "submit_grounded_answer", candidate)
    violations = result["error"]["details"]["violations"]
    assert any(v["code"] == "citation_not_found" for v in violations)

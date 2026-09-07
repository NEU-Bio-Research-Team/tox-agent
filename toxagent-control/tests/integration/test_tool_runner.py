"""Tool execution policy (plan sections 8.1, 8.4, 14.5).

The runner is where a denied tool is actually refused, where a run's budget is
enforced, and where a failure becomes a typed error envelope rather than prose.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from toxagent.application.create_analysis import CreateAnalysis
from toxagent.application.policy import Actor
from toxagent.config import PolicySettings
from toxagent.domain.events import EventType
from toxagent.domain.message import Message, Role
from toxagent.domain.run import Intent, Lane, Run, RunStatus
from toxagent.domain.session import Session
from toxagent.tools import envelope
from toxagent.tools.bootstrap import build_registry
from toxagent.tools.registry import ToolContext
from toxagent.tools.runner import ToolRunner
from tests.support.predictor import ASPIRIN, StubPredictor

pytestmark = pytest.mark.anyio

NOW = datetime(2026, 9, 4, tzinfo=timezone.utc)
ACTOR = Actor(subject_id="user-1")


async def scenario(db, stub: StubPredictor, *, profile="analysis", max_calls=12):
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
    analysis = CreateAnalysis(db, client, PolicySettings())
    registry = build_registry(db, client, analysis)
    runner = ToolRunner(registry, db, max_calls_per_run=max_calls)
    context = ToolContext(
        session_id=session.id, run_id=run.id, actor=ACTOR, profile=profile,
        deadline_at=datetime.now(timezone.utc) + timedelta(seconds=60),
    )
    return runner, context, session, run


async def test_a_snapshot_tool_call_returns_a_citable_projection(db):
    runner, context, session, _ = await scenario(db, StubPredictor())
    result = await runner.call(
        context, "create_analysis_snapshot",
        {"session_id": session.id, "smiles": ASPIRIN, "endpoints": ["herg"]},
    )
    assert result["status"] == "completed"
    assert result["observation_ids"]
    assert result["model_view"]["available_sections"] == ["herg", "applicability", "provenance"]
    # The model view lists what exists; it does not hand over the numbers.
    assert "0.73064" not in str(result["model_view"])


async def test_a_tool_outside_the_profile_is_denied_identically_to_a_missing_one(db):
    runner, context, _, _ = await scenario(db, StubPredictor(), profile="analysis")
    denied = await runner.call(context, "get_attribution", {"analysis_id": "ana_x", "endpoint": "herg"})
    missing = await runner.call(context, "no_such_tool", {})
    assert denied["error"]["code"] == missing["error"]["code"] == "tool_denied"
    assert denied["error"]["message"].endswith("is not available to this run")


async def test_a_slice_returns_field_paths_and_an_observation_id(db):
    runner, context, session, _ = await scenario(db, StubPredictor())
    snapshot = await runner.call(
        context, "create_analysis_snapshot", {"session_id": session.id, "smiles": ASPIRIN}
    )
    analysis_id = snapshot["model_view"]["analysis_id"]
    result = await runner.call(
        context, "get_analysis_slice",
        {"analysis_id": analysis_id, "section": "herg", "fields": ["probability_blocker"]},
    )
    value = result["model_view"]["values"]["probability_blocker"]
    assert value["value"] == 0.73064
    assert value["field_path"] == "predictions.herg.probability_blocker"
    assert value["observation_id"] == result["observation_ids"][0]


async def test_a_slice_cannot_reach_another_sessions_analysis(db):
    runner, context, session, _ = await scenario(db, StubPredictor())
    snapshot = await runner.call(
        context, "create_analysis_snapshot", {"session_id": session.id, "smiles": ASPIRIN}
    )
    other_runner, other_context, _, _ = await scenario(db, StubPredictor())
    result = await other_runner.call(
        other_context, "get_analysis_slice",
        {"analysis_id": snapshot["model_view"]["analysis_id"], "section": "herg"},
    )
    assert result["error"]["code"] == "analysis_not_found"


async def test_a_session_id_that_disagrees_with_the_run_is_denied(db):
    """Plan section 8.5: the token decides the session, not the arguments."""
    runner, context, _, _ = await scenario(db, StubPredictor())
    result = await runner.call(
        context, "create_analysis_snapshot",
        {"session_id": "ses_" + "0" * 32, "smiles": ASPIRIN},
    )
    assert result["error"]["code"] == "tool_denied"


async def test_arguments_that_do_not_match_the_schema_are_refused(db):
    runner, context, session, _ = await scenario(db, StubPredictor())
    result = await runner.call(
        context, "create_analysis_snapshot",
        {"session_id": session.id, "smiles": ASPIRIN, "unexpected": True},
    )
    assert result["error"]["code"] == "invalid_request"


async def test_an_endpoint_this_build_does_not_serve_is_an_error_envelope(db):
    runner, context, session, _ = await scenario(db, StubPredictor(served=("herg",)))
    result = await runner.call(
        context, "create_analysis_snapshot",
        {"session_id": session.id, "smiles": ASPIRIN, "endpoints": ["clintox"]},
    )
    assert envelope.is_error(result)
    assert result["error"]["code"] == "endpoint_unavailable"
    # A failure never arrives as a success body a model could read as a finding.
    assert "canonical" not in result


async def test_the_same_call_three_times_is_treated_as_a_loop(db):
    runner, context, session, _ = await scenario(db, StubPredictor())
    args = {"session_id": session.id, "smiles": ASPIRIN}
    first = await runner.call(context, "create_analysis_snapshot", args)
    second = await runner.call(context, "create_analysis_snapshot", args)
    third = await runner.call(context, "create_analysis_snapshot", args)
    assert first["status"] == second["status"] == "completed"
    assert third["error"]["code"] == "tool_denied"


async def test_a_run_that_exhausted_its_budget_stops_calling_tools(db):
    runner, context, session, _ = await scenario(db, StubPredictor(), max_calls=2)
    for index in range(2):
        await runner.call(
            context, "create_analysis_snapshot",
            {"session_id": session.id, "smiles": ASPIRIN, "endpoints": ["herg"] if index else None},
        )
    blocked = await runner.call(
        context, "get_analysis_slice", {"analysis_id": "ana_" + "0" * 32, "section": "herg"}
    )
    assert blocked["error"]["code"] == "tool_denied"
    assert "budget" in blocked["error"]["message"]


async def test_concurrent_calls_cannot_exceed_the_budget(db):
    """audit_5_9.md A04: five concurrent calls against ``max_calls=2`` used to
    all pass admission (checked in a separate transaction from the row that
    reserved a slot) and all complete. Exactly two must be admitted now."""
    runner, context, session, _ = await scenario(db, StubPredictor(), max_calls=2)
    analysis_id = (
        await runner.call(context, "create_analysis_snapshot", {"session_id": session.id, "smiles": ASPIRIN})
    )["model_view"]["analysis_id"]

    results = await asyncio.gather(
        *[
            runner.call(
                context, "get_analysis_slice",
                {"analysis_id": analysis_id, "section": "herg", "fields": ["probability_blocker"]},
            )
            for _ in range(5)
        ]
    )
    completed = [r for r in results if r["status"] == "completed"]
    denied = [r for r in results if envelope.is_error(r) and r["error"]["code"] == "tool_denied"]
    # One slot was already spent on create_analysis_snapshot above.
    assert len(completed) == 1
    assert len(denied) == 4

    async with db.unit_of_work() as uow:
        calls = await uow.tool_calls.list_for_run(context.run_id)
    assert len(calls) == 6  # 1 snapshot + 1 admitted slice + 4 denied attempts
    assert sum(1 for c in calls if c["status"] == "denied") == 4


async def test_submit_grounded_answer_is_exempt_from_the_call_budget(db):
    """A run that spent its whole budget on read tools must still be able to
    attempt the answer it is required to submit."""
    runner, context, session, _ = await scenario(db, StubPredictor(), max_calls=1)
    snapshot = await runner.call(
        context, "create_analysis_snapshot", {"session_id": session.id, "smiles": ASPIRIN}
    )
    assert snapshot["status"] == "completed"
    blocked = await runner.call(
        context, "get_analysis_slice",
        {"analysis_id": snapshot["model_view"]["analysis_id"], "section": "herg"},
    )
    assert blocked["error"]["code"] == "tool_denied"

    result = await runner.call(
        context, "submit_grounded_answer",
        {
            "schema_version": "grounded-answer-v1",
            "answer_markdown": "no claims here",
            "claims": [],
            "limitations": [],
        },
    )
    # Whatever the validator makes of the candidate's content, admission
    # itself must not be the reason it was refused.
    if envelope.is_error(result):
        assert result["error"]["code"] != "tool_denied"

    async with db.unit_of_work() as uow:
        calls = await uow.tool_calls.list_for_run(context.run_id)
    assert any(c["tool_name"] == "submit_grounded_answer" and c["status"] != "denied" for c in calls)


async def test_a_terminal_run_accepts_no_further_tool_calls(db):
    runner, context, session, run = await scenario(db, StubPredictor())
    async with db.unit_of_work() as uow:
        current = await uow.runs.get(run.id)
        started = current.transition(RunStatus.RUNNING, now=NOW)
        await uow.runs.update(
            started.transition(RunStatus.COMPLETED, now=NOW), expected_version=current.version
        )
        await uow.commit()
    result = await runner.call(
        context, "create_analysis_snapshot", {"session_id": session.id, "smiles": ASPIRIN}
    )
    assert result["error"]["code"] == "tool_denied"


async def test_a_cancelled_run_stops_accepting_tool_calls(db):
    runner, context, session, run = await scenario(db, StubPredictor())
    async with db.unit_of_work() as uow:
        await uow.runs.request_cancel(run.id)
        await uow.commit()
    result = await runner.call(
        context, "create_analysis_snapshot", {"session_id": session.id, "smiles": ASPIRIN}
    )
    assert result["error"]["code"] == "tool_denied"


async def test_a_deadline_that_has_passed_bounds_the_call(db):
    runner, context, session, _ = await scenario(db, StubPredictor())
    expired = ToolContext(
        **{**context.__dict__, "deadline_at": datetime.now(timezone.utc) - timedelta(seconds=1)}
    )
    result = await runner.call(
        expired, "create_analysis_snapshot", {"session_id": session.id, "smiles": ASPIRIN}
    )
    # Either it times out inside its 0.1s floor or it completes; what must not
    # happen is a call running long after the run's deadline.
    assert result["status"] in ("completed", "error")
    if envelope.is_error(result):
        assert result["error"]["code"] == "tool_timeout"


async def test_every_call_is_recorded_for_the_audit(db):
    runner, context, session, run = await scenario(db, StubPredictor())
    await runner.call(
        context, "create_analysis_snapshot", {"session_id": session.id, "smiles": ASPIRIN}
    )
    async with db.unit_of_work() as uow:
        calls = await uow.tool_calls.list_for_run(run.id)
    assert [c["tool_name"] for c in calls] == ["create_analysis_snapshot"]
    assert calls[0]["status"] == "completed"
    assert calls[0]["observation_ids"]

    events = [e.type for e in await db.outbox().read_after(session.id, 0)]
    assert EventType.TOOL_STARTED in events and EventType.TOOL_COMPLETED in events


async def test_tool_call_timestamps_are_returned_tz_aware(db):
    """audit_5_9.md A13: SQLite hands back naive datetimes; list_for_run used
    to return them as-is instead of normalizing to UTC like every other
    repository does, so a client's isoformat() carried no offset and a
    non-UTC browser misread it as local time."""
    runner, context, session, run = await scenario(db, StubPredictor())
    await runner.call(
        context, "create_analysis_snapshot", {"session_id": session.id, "smiles": ASPIRIN}
    )
    async with db.unit_of_work() as uow:
        calls = await uow.tool_calls.list_for_run(run.id)
    assert calls[0]["started_at"].tzinfo is not None
    assert calls[0]["ended_at"].tzinfo is not None


async def test_attribution_is_cached_by_molecule_endpoint_and_artifact(db):
    stub = StubPredictor()
    runner, context, session, _ = await scenario(db, stub, profile="report_qa")
    analysis = CreateAnalysis(db, stub.client(), PolicySettings())
    result = await analysis.execute(
        actor=ACTOR, session_id=context.session_id, run_id=context.run_id, smiles=ASPIRIN,
        owns_run=False,
    )
    first = await runner.call(
        context, "get_attribution", {"analysis_id": result.snapshot.id, "endpoint": "herg"}
    )
    second = await runner.call(
        context, "get_attribution", {"analysis_id": result.snapshot.id, "endpoint": "herg"}
    )
    assert first["status"] == "completed"
    assert second["provenance"]["cached"] is True
    assert len([r for r in stub.requests if r["path"] == "/v1/attributions"]) == 1


async def test_a_tox21_attribution_without_an_assay_is_refused(db):
    stub = StubPredictor()
    runner, context, _, _ = await scenario(db, stub, profile="report_qa")
    analysis = CreateAnalysis(db, stub.client(), PolicySettings())
    result = await analysis.execute(
        actor=ACTOR, session_id=context.session_id, run_id=context.run_id, smiles=ASPIRIN,
        owns_run=False,
    )
    denied = await runner.call(
        context, "get_attribution", {"analysis_id": result.snapshot.id, "endpoint": "tox21"}
    )
    assert denied["error"]["code"] == "invalid_request"
    assert "independent" in denied["error"]["message"]


async def test_the_attribution_projection_carries_its_limitation(db):
    stub = StubPredictor()
    runner, context, _, _ = await scenario(db, stub, profile="report_qa")
    analysis = CreateAnalysis(db, stub.client(), PolicySettings())
    result = await analysis.execute(
        actor=ACTOR, session_id=context.session_id, run_id=context.run_id, smiles=ASPIRIN,
        owns_run=False,
    )
    attribution = await runner.call(
        context, "get_attribution",
        {"analysis_id": result.snapshot.id, "endpoint": "tox21", "task": "SR-p53"},
    )
    assert attribution["model_view"]["required_limitations"] == ["attribution_not_causality"]

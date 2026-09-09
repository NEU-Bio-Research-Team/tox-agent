"""Startup reconciliation of orphaned runs (audit_5_9.md A05).

Mirrors the audit's repro: start a fresh process (here, a fresh call — no
``RunScheduler`` task exists for these runs) against a database that already
has a non-terminal run left by a previous, now-gone process.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from toxagent.application.startup_reconciliation import reconcile_orphaned_runs
from toxagent.domain.events import EventType
from toxagent.domain.message import Message, Role
from toxagent.domain.run import Intent, Lane, Run, RunStatus
from toxagent.domain.runtime import RuntimeBinding, RuntimeCapabilities, RuntimeKind
from toxagent.domain.session import Session

pytestmark = pytest.mark.anyio

NOW = datetime(2026, 9, 4, tzinfo=timezone.utc)


async def _seed_queued_run(db, *, cancel_requested: bool = False) -> tuple[str, str]:
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
        if cancel_requested:
            await uow.runs.request_cancel(run.id)
        await uow.commit()
    return session.id, run.id


async def test_a_run_left_queued_by_a_crashed_process_is_failed(db):
    session_id, run_id = await _seed_queued_run(db)

    reconciled = await reconcile_orphaned_runs(db)
    assert reconciled == 1

    async with db.unit_of_work() as uow:
        run = await uow.runs.get(run_id)
        messages = await uow.messages.list_for_session(session_id)
    assert run.is_terminal
    assert run.status is RunStatus.FAILED
    assert run.failure_code == "runtime_unavailable"
    system_notices = [m for m in messages if m.role.value == "system_event"]
    assert len(system_notices) == 1
    assert system_notices[0].parts[0].content["code"] == "runtime_unavailable"


async def test_a_run_that_had_bound_a_runtime_is_reconciled_as_potentially_billed(db):
    """Plan section 6.6 / remaining-plan W2-12: a runtime_binding_id means
    AgentRuntimeGateway.execute got at least as far as creating the binding
    (which happens before send()) — the crash could have landed anywhere
    from "about to send" to "provider mid-response", and nothing persisted
    distinguishes those, so reconciliation must not default to "not billed"
    for lack of proof otherwise."""
    session_id, run_id = await _seed_queued_run(db)
    async with db.unit_of_work() as uow:
        run = await uow.runs.get(run_id)
        binding = RuntimeBinding.create(
            session_id=session_id,
            runtime_kind=RuntimeKind.SCRIPTED,
            runtime_version="test",
            runtime_session_id="rts_reconciliation_test",
            provider_id="scripted",
            model_id="scripted",
            profile_hash="test-profile",
            tool_schema_hash="test-tools",
            system_prompt_hash="test-prompt",
            capabilities=RuntimeCapabilities(),
            now=NOW,
        )
        await uow.runtime_bindings.add(binding)
        bound = run.transition(
            RunStatus.RUNNING, now=NOW, runtime_binding_id=binding.id
        )
        await uow.runs.update(bound, expected_version=run.version)
        await uow.commit()

    reconciled = await reconcile_orphaned_runs(db)
    assert reconciled == 1

    async with db.unit_of_work() as uow:
        run = await uow.runs.get(run_id)
    assert run.status is RunStatus.FAILED
    assert run.potentially_billed is True


async def test_a_run_never_scheduled_is_reconciled_without_being_marked_billed(db):
    """The queued-and-never-started case (existing coverage above) never
    touched a runtime at all — confirmed explicitly here so the new billing
    branch doesn't quietly start marking every orphan, only ones with a
    binding."""
    session_id, run_id = await _seed_queued_run(db)
    await reconcile_orphaned_runs(db)

    async with db.unit_of_work() as uow:
        run = await uow.runs.get(run_id)
    assert run.status is RunStatus.FAILED
    assert run.potentially_billed is False


async def test_a_terminal_run_is_left_alone(db):
    session, run_id = await _seed_queued_run(db)
    async with db.unit_of_work() as uow:
        run = await uow.runs.get(run_id)
        started = run.transition(RunStatus.RUNNING, now=NOW)
        await uow.runs.update(
            started.transition(RunStatus.COMPLETED, now=NOW), expected_version=run.version
        )
        await uow.commit()

    reconciled = await reconcile_orphaned_runs(db)
    assert reconciled == 0

    async with db.unit_of_work() as uow:
        run = await uow.runs.get(run_id)
    assert run.status is RunStatus.COMPLETED


async def test_cancelling_no_longer_hangs_forever_after_reconciliation(db):
    """This is the exact operability gap the audit reproduced: cancel() on an
    orphaned run answered `cancellation_recorded_no_local_worker` forever
    because nothing ever picked the flag up. After reconciliation the run is
    terminal, so a subsequent cancel honestly reports there is nothing left
    to cancel instead of repeating that same non-answer indefinitely."""
    from toxagent.application.run_scheduler import RunScheduler

    session_id, run_id = await _seed_queued_run(db, cancel_requested=True)
    await reconcile_orphaned_runs(db)

    scheduler = RunScheduler(db)
    outcome = await scheduler.cancel(run_id)
    assert outcome.action == "run_already_terminal"

    async with db.unit_of_work() as uow:
        run = await uow.runs.get(run_id)
    assert run.is_terminal

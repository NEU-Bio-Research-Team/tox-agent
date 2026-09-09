"""A turn that genuinely runs out of time fails as deadline_exceeded, not
internal_error (plan section 10, domain/errors.py).

Live sweep (2026-09-05): AgentRuntimeGateway._consume_events bounds each
``anext(stream)`` wait to the deadline's remaining seconds, but caught only
``StopAsyncIteration`` — a real ``asyncio.TimeoutError`` from that bounded
wait (the case ``timeout=remaining`` exists to produce) escaped uncaught,
past run_scheduler's catch-all, and was recorded as failure_code
"internal_error" for a turn that simply took too long, not one that broke.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from toxagent.application.policy import Actor
from toxagent.application.run_scheduler import RunContext
from toxagent.config import RuntimeSettings
from toxagent.domain.message import Message, Role
from toxagent.domain.run import Intent, Lane, Run
from toxagent.domain.runtime import RuntimeBinding, RuntimeCapabilities, RuntimeKind
from toxagent.domain.session import Session
from toxagent.harness.gateway import AgentRuntimeGateway
from toxagent.harness.provider import RuntimeHealth, RuntimeSession

pytestmark = pytest.mark.anyio

NOW = datetime(2026, 9, 5, tzinfo=timezone.utc)
ACTOR = Actor(subject_id="user-1")


class _StalledStreamProvider:
    """A runtime whose event stream never produces anything — standing in
    for a real turn that is simply still thinking when the deadline hits."""

    kind = "scripted"

    async def health(self) -> RuntimeHealth:
        return RuntimeHealth(healthy=True)

    async def events(self, session, after=None):
        import asyncio

        while True:
            await asyncio.sleep(3600)
            yield None  # pragma: no cover - never reached


async def test_a_stalled_event_stream_fails_as_deadline_exceeded_not_internal_error(db):
    session = Session.create("user-1", now=NOW)
    message = Message.create(session.id, Role.USER, 1, now=NOW)
    run = Run.create(session.id, message.id, Lane.AGENTIC, Intent.REPORT_QA, now=NOW)
    async with db.unit_of_work() as uow:
        await uow.sessions.add(session)
        await uow.messages.add(message)
        await uow.runs.add(run)
        await uow.commit()

    gateway = AgentRuntimeGateway(
        db, registry=None, capability_tokens=None,
        provider=_StalledStreamProvider(), settings=RuntimeSettings(),
    )
    context = RunContext(
        actor=ACTOR, session_id=session.id, run_id=run.id, intent=Intent.REPORT_QA,
    )
    runtime_session = RuntimeSession(
        runtime_session_id="rts_test", provider_id="scripted", model_id="scripted",
    )
    # _consume_events records provider usage against a persisted binding when
    # such an event arrives. The stalled stream cannot report usage, but keep
    # this direct boundary test on the real post-W2-13 signature so it cannot
    # silently stop exercising deadline behaviour after the next event shape
    # change.
    binding = RuntimeBinding.create(
        session_id=session.id,
        runtime_kind=RuntimeKind.SCRIPTED,
        runtime_version="test",
        runtime_session_id=runtime_session.runtime_session_id,
        provider_id=runtime_session.provider_id,
        model_id=runtime_session.model_id,
        profile_hash="test-profile",
        tool_schema_hash="test-tools",
        system_prompt_hash="test-prompt",
        capabilities=RuntimeCapabilities(streaming=True),
        now=NOW,
    )

    from toxagent.domain.errors import DeadlineExceeded

    # Relative to real wall-clock time, not the fixed NOW used for the DB
    # rows above — a small *positive* remaining is what actually exercises
    # the asyncio.TimeoutError path this test is pinned to; a deadline
    # already in the past would take the earlier "remaining <= 0" branch
    # instead, which was never the bug.
    live_deadline = datetime.now(timezone.utc) + timedelta(milliseconds=50)
    with pytest.raises(DeadlineExceeded):
        await gateway._consume_events(runtime_session, context, binding, deadline=live_deadline)

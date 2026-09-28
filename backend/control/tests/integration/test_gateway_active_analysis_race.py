"""W3-07: a run's own resolved analysis wins over a session's
``active_analysis_id`` that has since moved on.

Two analyses in one session, back to back, is exactly the race the plan
worries about: a second "analyze this" turn changes
``session.active_analysis_id`` while a first run's follow-up (which named its
own ``analysis_id`` explicitly) is still in flight. ``harness/gateway.py``'s
``context.analysis_id or session.active_analysis_id`` precedence already
exists to prevent the first run's follow-up from silently picking up the
second molecule — this is a regression test for that precedence, not new
production code.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from toxagent.application.prediction.create_analysis import CreateAnalysis
from toxagent.application.policy import Actor
from toxagent.application.runs.scheduler import RunContext
from toxagent.config import PolicySettings, RuntimeSettings
from toxagent.domain.events import EventType
from toxagent.domain.message import Message, Role
from toxagent.domain.run import Intent, Lane, Run
from toxagent.domain.session import Session
from toxagent.harness.gateway import AgentRuntimeGateway
from toxagent.tools.registry import ToolRegistry
from tests.support.predictor import ASPIRIN, StubPredictor

BENZENE = "c1ccccc1"

pytestmark = pytest.mark.anyio

# Relative, not a calendar date: _prepare_context refuses a run whose deadline
# (created_at + run_deadline_s) has already passed, so a fixed date rots.
NOW = datetime.now(timezone.utc)
ACTOR = Actor(subject_id="user-1")


async def test_a_runs_named_analysis_wins_over_a_session_that_moved_on(db):
    session = Session.create(ACTOR.subject_id, now=NOW)
    message = Message.create(session.id, Role.USER, 1, now=NOW)
    run = Run.create(session.id, message.id, Lane.AGENTIC, Intent.DECISION_SUPPORT, now=NOW)
    async with db.unit_of_work() as uow:
        await uow.sessions.add(session)
        await uow.messages.add(message)
        await uow.runs.add(run)
        uow.emit(
            session_id=session.id, type=EventType.SESSION_CREATED,
            entity_type="session", entity_id=session.id,
        )
        await uow.commit()

    predictor = StubPredictor().client()
    analysis_service = CreateAnalysis(db, predictor, PolicySettings())
    first = await analysis_service.execute(
        actor=ACTOR, session_id=session.id, run_id=run.id, smiles=ASPIRIN, owns_run=False,
    )
    # A second "analyze this" turn on the same session moves
    # session.active_analysis_id forward — the race the plan describes.
    second_run = Run.create(session.id, message.id, Lane.AGENTIC, Intent.DECISION_SUPPORT, now=NOW)
    async with db.unit_of_work() as uow:
        await uow.runs.add(second_run)
        await uow.commit()
    await analysis_service.execute(
        actor=ACTOR, session_id=session.id, run_id=second_run.id, smiles=BENZENE, owns_run=False,
    )

    gateway = AgentRuntimeGateway(
        db, registry=ToolRegistry(), capability_tokens=None, provider=None,
        settings=RuntimeSettings(),
    )
    # The first run's own follow-up still names its own analysis explicitly.
    context = RunContext(
        actor=ACTOR, session_id=session.id, run_id=run.id, intent=Intent.DECISION_SUPPORT,
        analysis_id=first.snapshot.id,
    )
    system_prompt, _profile, _deadline, _hash = await gateway._prepare_context(context)

    assert first.snapshot.id in system_prompt
    assert f"canonical SMILES={ASPIRIN}" in system_prompt
    assert f"canonical SMILES={BENZENE}" not in system_prompt

"""The session list must describe the session, not the first page of it.

I20's two numbers were both artefacts of how they were computed. The preview
read the first 50 messages ordered ascending and took the last of them, so it
was the newest message only while a session had 50 or fewer; past that it froze
on message 50 and never moved again. The run count was `len()` of a run page
capped at 10, so every busy session reported exactly 10.

The thresholds here are 61 and 12 for that reason: they are the smallest
numbers that put both values past the cap that produced them.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from toxagent.application.policy import Actor
from toxagent.application.sessions import SessionService
from toxagent.domain.message import Message, PartType, Role
from toxagent.domain.run import Intent, Lane, Run
from toxagent.domain.session import Session

pytestmark = pytest.mark.anyio

NOW = datetime(2026, 9, 9, tzinfo=timezone.utc)
ACTOR = Actor(subject_id="user-1")


async def _seed(db, *, messages: int, runs: int) -> tuple[str, str]:
    session = Session.create(ACTOR.subject_id, now=NOW)
    last_text = ""
    async with db.unit_of_work() as uow:
        await uow.sessions.add(session)
        for n in range(1, messages + 1):
            last_text = f"message number {n}"
            await uow.messages.add(
                Message.create(
                    session.id, Role.USER, n, now=NOW + timedelta(seconds=n),
                    parts=((PartType.TEXT, {"text": last_text}),),
                )
            )
        trigger = await uow.messages.get(
            (await uow.messages.list_for_session(session.id, limit=1))[0].id
        )
        for n in range(runs):
            run = Run.create(
                session.id, trigger.id, Lane.DETERMINISTIC, Intent.ANALYSIS,
                now=NOW + timedelta(seconds=n),
            )
            await uow.runs.add(run)
        await uow.commit()
    return session.id, last_text


async def test_the_preview_is_the_latest_message_past_the_old_page_size(db):
    session_id, last_text = await _seed(db, messages=61, runs=0)

    page = await SessionService(db).list(ACTOR)

    row = next(r for r in page["sessions"] if r["session_id"] == session_id)
    assert row["last_message_preview"] == last_text == "message number 61"


async def test_the_run_count_is_the_number_of_runs_not_the_page_size(db):
    session_id, _ = await _seed(db, messages=1, runs=12)

    page = await SessionService(db).list(ACTOR)

    row = next(r for r in page["sessions"] if r["session_id"] == session_id)
    assert row["run_count"] == 12


async def test_a_long_preview_is_truncated_and_a_textless_session_has_none(db):
    long_session = Session.create(ACTOR.subject_id, now=NOW)
    quiet_session = Session.create(ACTOR.subject_id, now=NOW)
    async with db.unit_of_work() as uow:
        await uow.sessions.add(long_session)
        await uow.sessions.add(quiet_session)
        await uow.messages.add(
            Message.create(
                long_session.id, Role.USER, 1, now=NOW,
                parts=((PartType.TEXT, {"text": "x" * 300}),),
            )
        )
        await uow.messages.add(
            Message.create(
                quiet_session.id, Role.SYSTEM_EVENT, 1, now=NOW,
                parts=((PartType.ERROR, {"code": "runtime_unavailable", "message": "no"}),),
            )
        )
        await uow.commit()

    rows = {r["session_id"]: r for r in (await SessionService(db).list(ACTOR))["sessions"]}

    assert rows[long_session.id]["last_message_preview"] == "x" * 160 + "…"
    assert rows[quiet_session.id]["last_message_preview"] is None


async def test_the_query_count_does_not_grow_with_the_number_of_sessions(db):
    """The old loop ran two queries per session listed. Whatever the page
    costs now, listing five sessions must not cost five times listing one."""
    for _ in range(5):
        await _seed(db, messages=3, runs=2)

    import sqlalchemy.ext.asyncio as sa_asyncio

    executed: list[str] = []

    real_execute = sa_asyncio.AsyncConnection.execute

    async def counting_execute(self, statement, *args, **kwargs):
        executed.append(str(statement)[:40])
        return await real_execute(self, statement, *args, **kwargs)

    sa_asyncio.AsyncConnection.execute = counting_execute
    try:
        await SessionService(db).list(ACTOR, limit=1)
        one_session = len(executed)
        executed.clear()
        await SessionService(db).list(ACTOR, limit=5)
        five_sessions = len(executed)
    finally:
        sa_asyncio.AsyncConnection.execute = real_execute

    assert five_sessions == one_session, (
        f"listing five sessions ran {five_sessions} queries and one ran "
        f"{one_session}: the summary is still per-session"
    )

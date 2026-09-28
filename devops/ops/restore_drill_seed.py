"""Write, then re-read, the data the restore drill compares.

Deliberately written through the application's own repositories rather than
with raw SQL: a dump is only interesting if it carries what the product
actually stores — a session with an owner, a message, a run, and an outbox
event — through the same mapping the product reads it back with. Raw INSERTs
would prove PostgreSQL can copy rows, which was never in doubt.

    restore_drill_seed.py            # seed the database at TOXAGENT_DRILL_URL
    restore_drill_seed.py --verify   # read it back and check it is intact
"""
from __future__ import annotations

import asyncio
import os
import sys
from datetime import datetime, timezone

from toxagent.domain.events import EventType
from toxagent.domain.message import Message, PartType, Role
from toxagent.domain.run import Intent, Lane, Run
from toxagent.domain.session import Session
from toxagent.persistence.sql.database import Database

#: Fixed, so seeding and verifying agree without passing ids around, and so a
#: rerun of the drill overwrites rather than accumulates.
OWNER = "drill-owner"
CLIENT_SESSION_ID = "drill-session-0001"
NOW = datetime(2026, 9, 9, 12, 0, 0, tzinfo=timezone.utc)
MESSAGE_TEXT = "restore drill — a session that must survive the dump"


def url() -> str:
    value = os.getenv("TOXAGENT_DRILL_URL")
    if not value:
        raise SystemExit("set TOXAGENT_DRILL_URL")
    return value


async def seed() -> None:
    database = Database(url())
    try:
        session = Session.create(OWNER, now=NOW, title="restore drill")
        message = Message.create(
            session.id, Role.USER, 1, now=NOW,
            parts=((PartType.TEXT, {"text": MESSAGE_TEXT}),),
        )
        run = Run.create(session.id, message.id, Lane.DETERMINISTIC, Intent.ANALYSIS, now=NOW)
        async with database.unit_of_work() as uow:
            await uow.sessions.add(session, client_session_id=CLIENT_SESSION_ID)
            await uow.messages.add(message)
            await uow.runs.add(run)
            uow.emit(
                session_id=session.id, type=EventType.SESSION_CREATED,
                entity_type="session", entity_id=session.id,
            )
            await uow.commit()
        print(f"seed: session={session.id} run={run.id}")
    finally:
        await database.dispose()


async def verify() -> None:
    database = Database(url())
    try:
        async with database.unit_of_work() as uow:
            session = await uow.sessions.find_by_client_id(OWNER, CLIENT_SESSION_ID)
            if session is None:
                raise SystemExit("verify: the restored database has no session")
            messages = await uow.messages.list_for_session(session.id)
            runs = await uow.runs.list_for_session(session.id)
        events = await database.outbox().read_after(session.id, 0)

        problems = []
        if session.owner_id != OWNER:
            problems.append(f"owner is {session.owner_id!r}")
        texts = [
            part.content.get("text")
            for message in messages for part in message.parts
            if part.type is PartType.TEXT
        ]
        if MESSAGE_TEXT not in texts:
            problems.append(f"the message text did not survive (got {texts!r})")
        if len(runs) != 1:
            problems.append(f"{len(runs)} runs, expected 1")
        if not events or events[0].type is not EventType.SESSION_CREATED:
            problems.append("the outbox event did not survive")
        if problems:
            raise SystemExit("verify: " + "; ".join(problems))
        print(
            f"verify: session={session.id} owner={session.owner_id} "
            f"messages={len(messages)} runs={len(runs)} events={len(events)}"
        )
    finally:
        await database.dispose()


if __name__ == "__main__":
    asyncio.run(verify() if "--verify" in sys.argv[1:] else seed())

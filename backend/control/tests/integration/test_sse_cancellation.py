"""SSE consumer cancellation must not leak an unhandled exception
(audit_5_9.md A15: control.log showed `Exception terminating connection`,
`CancelledError`, `Task exception was never retrieved`, and
`sqlite3.OperationalError: no active connection` clustered around session
open/close). Not a deterministic race to reproduce on demand — this is a
stress test, not a proof, matching the audit's own "chưa chạy stress test để
kết luận" caveat.
"""
from __future__ import annotations

import asyncio
import contextlib
from datetime import datetime, timezone

import pytest

from toxagent.domain.events import EventType
from toxagent.domain.session import Session
from toxagent.streaming.events import EventNotifier
from toxagent.streaming.sse import event_stream

pytestmark = pytest.mark.anyio

NOW = datetime(2026, 9, 4, tzinfo=timezone.utc)


async def test_repeatedly_cancelling_sse_consumers_leaves_no_unhandled_exception(db):
    session = Session.create("user-1", now=NOW)
    async with db.unit_of_work() as uow:
        await uow.sessions.add(session)
        uow.emit(
            session_id=session.id, type=EventType.SESSION_CREATED,
            entity_type="session", entity_id=session.id,
        )
        await uow.commit()

    loop = asyncio.get_event_loop()
    unhandled: list[BaseException] = []
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: unhandled.append(context.get("exception")))

    async def producer(stop: asyncio.Event) -> None:
        while not stop.is_set():
            async with db.unit_of_work() as uow:
                uow.emit(
                    session_id=session.id, type=EventType.MESSAGE_CREATED,
                    entity_type="message", entity_id="msg_x",
                )
                await uow.commit()
            await asyncio.sleep(0)

    try:
        notifier = EventNotifier()
        stop = asyncio.Event()
        producer_task = asyncio.ensure_future(producer(stop))
        try:
            for _ in range(50):
                stream = event_stream(db.outbox(), notifier, session.id, poll_seconds=0.05)
                consumer = asyncio.ensure_future(stream.__anext__())
                await asyncio.sleep(0)
                consumer.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await consumer
        finally:
            stop.set()
            producer_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await producer_task
        # Let any shielded background cleanup finish before checking.
        await asyncio.sleep(0.1)
    finally:
        loop.set_exception_handler(previous_handler)

    assert unhandled == [], unhandled

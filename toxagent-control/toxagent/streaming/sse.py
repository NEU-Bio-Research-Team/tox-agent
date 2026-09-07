"""The SSE change feed (plan section 6.4).

Only committed outbox rows are published, ordered by the per-session sequence.
Delivery is at-least-once and the client dedupes on ``event_id``; ``Last-Event-ID``
or ``?after_sequence=`` resumes. Losing the stream loses nothing, because every
event describes a state change the REST endpoints can also report — which is the
property PROD-05 asks for and the reason this generator is allowed to be simple.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, AsyncIterator

from ..domain.events import Event
from .events import EventNotifier

#: How long to wait on a notification before polling anyway. Also the heartbeat
#: interval, which keeps proxies from closing an idle stream.
POLL_SECONDS = 15.0
BATCH = 200


def _now() -> datetime:
    return datetime.now(timezone.utc)


async def event_stream(
    outbox,
    notifier: EventNotifier,
    session_id: str,
    *,
    after_sequence: int = 0,
    poll_seconds: float = POLL_SECONDS,
    max_idle_seconds: float | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Yield ``sse-starlette`` event dicts for one session, from a sequence."""
    cursor = after_sequence
    idle = 0.0
    while True:
        events = await outbox.read_after(session_id, cursor, limit=BATCH)
        if events:
            idle = 0.0
            for event in events:
                cursor = event.sequence
                yield _envelope(event)
            await outbox.mark_dispatched([e.event_id for e in events], now=_now())
            continue

        woke = await notifier.wait(session_id, poll_seconds)
        if not woke:
            idle += poll_seconds
            if max_idle_seconds is not None and idle >= max_idle_seconds:
                return
            # A comment frame: keeps the connection warm without inventing an
            # event a client would have to interpret.
            yield {"event": "heartbeat", "data": "{}", "id": str(cursor)}


def _envelope(event: Event) -> dict[str, Any]:
    import json

    return {
        "event": event.type.value,
        "id": str(event.sequence),
        "data": json.dumps(event.to_dict(), ensure_ascii=False),
    }

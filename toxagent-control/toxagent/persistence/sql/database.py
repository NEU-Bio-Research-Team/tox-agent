"""Engine, unit of work, and the outbox.

The unit of work is where the plan's central persistence promise is kept: a
state change and the events describing it are written in one transaction
(section 13.3). Event sequences are allocated from the session's counter at
flush time under that same transaction, so two concurrent runs in one session
cannot both claim sequence 42 and cannot produce a feed a client is unable to
order.
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Awaitable, Callable, Coroutine, Sequence, TypeVar

from sqlalchemy import insert, select, update
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, create_async_engine

log = logging.getLogger("toxagent.persistence")

_T = TypeVar("_T")

from ...domain.events import Event, EventType
from ...domain.ids import EVENT, new_id
from ..schema import event_outbox, metadata, sessions
from . import mapping as m
from .repositories import (
    SqlAnalysisStore,
    SqlAnswerStore,
    SqlAttachmentStore,
    SqlCapabilityTokenStore,
    SqlEvidenceStore,
    SqlMessageStore,
    SqlObservationStore,
    SqlRunStore,
    SqlRuntimeBindingStore,
    SqlRuntimeUsageStore,
    SqlSessionStore,
    SqlToolCallStore,
)

CommitHook = Callable[[Sequence[str]], Awaitable[None] | None]


def _now() -> datetime:
    return datetime.now(timezone.utc)


async def _shielded(coro: Coroutine[Any, Any, _T]) -> _T:
    """Run ``coro`` shielded from the caller's own cancellation.

    The SSE change feed's generator (``streaming/sse.py``) is torn down the
    instant a client disconnects. If that cancellation lands while this
    coroutine is mid-query, aiosqlite can raise a second, unrelated exception
    while closing the connection out from underneath a query still in
    flight (``sqlite3.OperationalError: no active connection`` in
    ``control.log``, 2026-09-04) — a client walking away is not, on its own,
    a bug, but that secondary failure was logged as if it were one.

    Shielding lets the query (and its own connection teardown) finish
    cleanly in the background instead of being aborted mid-statement; the
    ``add_done_callback`` here is what "finish in the background" actually
    means in asyncio terms, and it is also what stops that secondary
    exception from being reported as "Task exception was never retrieved" —
    without it, nothing would ever retrieve it. The caller is cancelled
    exactly as promptly either way; only the query's own cleanup gets to
    finish uninterrupted.
    """
    task = asyncio.ensure_future(coro)
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        def _drain(finished: "asyncio.Task[_T]") -> None:
            if finished.cancelled():
                return
            exc = finished.exception()
            if exc is not None:
                log.debug("a shielded database operation failed after cancellation: %r", exc)

        task.add_done_callback(_drain)
        raise


class SqlUnitOfWork:
    def __init__(self, conn: AsyncConnection, *, on_commit: CommitHook | None = None) -> None:
        self._conn = conn
        self._on_commit = on_commit
        self._pending: list[dict[str, Any]] = []
        self.sessions = SqlSessionStore(conn)
        self.messages = SqlMessageStore(conn)
        self.runs = SqlRunStore(conn)
        self.analyses = SqlAnalysisStore(conn)
        self.observations = SqlObservationStore(conn)
        self.evidence = SqlEvidenceStore(conn)
        self.answers = SqlAnswerStore(conn)
        self.runtime_bindings = SqlRuntimeBindingStore(conn)
        self.runtime_usage = SqlRuntimeUsageStore(conn)
        self.tool_calls = SqlToolCallStore(conn)
        self.capability_tokens = SqlCapabilityTokenStore(conn)
        self.attachments = SqlAttachmentStore(conn)

    # --- events ------------------------------------------------------------

    def emit(
        self,
        *,
        session_id: str,
        type: EventType,
        entity_type: str,
        entity_id: str,
        run_id: str | None = None,
        entity_version: int = 1,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Queue an event. It is numbered and written when the work commits."""
        self._pending.append(
            {
                "event_id": new_id(EVENT),
                "session_id": session_id,
                "type": type.value,
                "entity_type": entity_type,
                "entity_id": entity_id,
                "entity_version": entity_version,
                "run_id": run_id,
                "payload": payload or {},
                "occurred_at": _now(),
            }
        )

    async def _flush_events(self) -> list[str]:
        if not self._pending:
            return []
        touched: list[str] = []
        by_session: dict[str, list[dict[str, Any]]] = {}
        for event in self._pending:
            by_session.setdefault(event["session_id"], []).append(event)

        rows: list[dict[str, Any]] = []
        for session_id, events in by_session.items():
            query = select(sessions.c.event_sequence).where(sessions.c.id == session_id)
            if self._conn.dialect.name == "postgresql":
                # A child-row insert (for example, the tool-call reservation)
                # takes PostgreSQL's KEY SHARE lock on its parent session.
                # ``FOR UPDATE`` conflicts with that otherwise-compatible
                # lock, so concurrent reservations could each hold KEY SHARE
                # and then deadlock while both tried to allocate the next
                # outbox sequence. We update only ``event_sequence``, never a
                # key column, therefore FOR NO KEY UPDATE is sufficient: it
                # serializes competing sequence allocators while remaining
                # compatible with those foreign-key checks.
                query = query.with_for_update(key_share=True)
            base = (await self._conn.execute(query)).scalar()
            if base is None:
                raise ValueError(f"cannot emit events for unknown session {session_id}")
            for offset, event in enumerate(events, start=1):
                rows.append({**event, "sequence": base + offset})
            await self._conn.execute(
                update(sessions)
                .where(sessions.c.id == session_id)
                .values(event_sequence=base + len(events))
            )
            touched.append(session_id)

        await self._conn.execute(insert(event_outbox), rows)
        self._pending.clear()
        return touched

    # --- transaction -------------------------------------------------------

    async def commit(self) -> None:
        touched = await self._flush_events()
        await self._conn.commit()
        if self._on_commit is not None and touched:
            result = self._on_commit(touched)
            if result is not None:
                await result

    async def rollback(self) -> None:
        self._pending.clear()
        await self._conn.rollback()


class SqlOutboxReader:
    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def read_after(
        self,
        session_id: str,
        after_sequence: int,
        *,
        limit: int = 200,
        run_id: str | None = None,
    ) -> Sequence[Event]:
        clauses = [
            event_outbox.c.session_id == session_id,
            event_outbox.c.sequence > after_sequence,
        ]
        if run_id is not None:
            clauses.append(event_outbox.c.run_id == run_id)

        async def _query() -> Sequence[Any]:
            async with self._engine.connect() as conn:
                return (
                    await conn.execute(
                        select(event_outbox)
                        .where(*clauses)
                        .order_by(event_outbox.c.sequence)
                        .limit(limit)
                    )
                ).mappings().all()

        rows = await _shielded(_query())
        return [m.row_to_event(r) for r in rows]

    async def mark_dispatched(self, event_ids: Sequence[str], *, now: datetime) -> None:
        """Delivery bookkeeping only. SSE is at-least-once and clients dedupe on
        event id, so this is never read back as a delivery guarantee."""
        if not event_ids:
            return

        async def _update() -> None:
            async with self._engine.begin() as conn:
                await conn.execute(
                    update(event_outbox)
                    .where(event_outbox.c.event_id.in_(list(event_ids)))
                    .values(dispatched_at=now)
                )

        await _shielded(_update())

    async def latest_sequence(self, session_id: str) -> int:
        async with self._engine.connect() as conn:
            value = (
                await conn.execute(select(sessions.c.event_sequence).where(sessions.c.id == session_id))
            ).scalar()
        return int(value or 0)


class Database:
    """Owns the engine. One instance per process, created in the app factory."""

    def __init__(self, url: str, *, echo: bool = False, on_commit: CommitHook | None = None) -> None:
        connect_args: dict[str, Any] = {}
        if url.startswith("sqlite"):
            # SQLite serialises writers; a short wait beats a spurious "database
            # is locked" under the concurrent-run tests.
            connect_args["timeout"] = 15
        self._engine = create_async_engine(url, echo=echo, future=True, connect_args=connect_args)
        self._on_commit = on_commit
        self._outbox = SqlOutboxReader(self._engine)

    @property
    def engine(self) -> AsyncEngine:
        return self._engine

    def set_commit_hook(self, hook: CommitHook) -> None:
        self._on_commit = hook

    @asynccontextmanager
    async def unit_of_work(self) -> AsyncIterator[SqlUnitOfWork]:
        async with self._engine.connect() as conn:
            uow = SqlUnitOfWork(conn, on_commit=self._on_commit)
            try:
                yield uow
            except BaseException:
                await uow.rollback()
                raise

    def outbox(self) -> SqlOutboxReader:
        return self._outbox

    async def create_schema(self) -> None:
        async with self._engine.begin() as conn:
            await conn.run_sync(metadata.create_all)

    async def dispose(self) -> None:
        await self._engine.dispose()

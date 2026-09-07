"""SQLAlchemy Core repositories.

Each takes the connection of the enclosing unit of work, so everything a
workflow touches — including the events it emits — lands in one transaction.
None of these classes opens a transaction of its own.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Sequence

from sqlalchemy import and_, delete, func, insert, literal, select, update
from sqlalchemy.ext.asyncio import AsyncConnection

from ...domain.analysis import AnalysisSnapshot
from ...domain.answer import Claim, GroundedAnswer
from ...domain.attachment import Attachment
from ...domain.errors import Conflict
from ...domain.evidence import EvidenceRecord, EvidenceStatus
from ...domain.message import Message
from ...domain.observation import Observation
from ...domain.run import Run
from ...domain.runtime import RuntimeBinding
from ...domain.usage import RuntimeUsageEvent
from ...domain.session import Session
from ..schema import (
    analysis_snapshots,
    answers,
    attachments,
    capability_tokens,
    claim_sources,
    claims,
    evidence_records,
    message_parts,
    messages,
    observations,
    runs,
    runtime_bindings,
    runtime_usage_events,
    sessions,
    tool_calls,
)
from . import mapping as m


class SqlSessionStore:
    def __init__(self, conn: AsyncConnection) -> None:
        self._conn = conn

    async def add(self, session: Session, *, client_session_id: str | None = None) -> None:
        await self._conn.execute(insert(sessions).values(m.session_to_row(session, client_session_id)))

    async def get(self, session_id: str, *, owner_id: str) -> Session | None:
        row = (
            await self._conn.execute(
                select(sessions).where(
                    and_(sessions.c.id == session_id, sessions.c.owner_id == owner_id)
                )
            )
        ).mappings().first()
        return m.row_to_session(row) if row else None

    async def get_unscoped(self, session_id: str) -> Session | None:
        row = (
            await self._conn.execute(select(sessions).where(sessions.c.id == session_id))
        ).mappings().first()
        return m.row_to_session(row) if row else None

    async def get_for_admission(
        self, session_id: str, *, owner_id: str, lock_timeout_ms: int
    ) -> Session | None:
        """Read an owned session while serializing admissions for it.

        Every message admission holds this parent row through its
        message/run/outbox commit. A second control-plane process can therefore
        re-check idempotency and the active-run cap only after the first has
        made its decision durable. ``NO KEY UPDATE`` is enough because an
        admission never changes the session primary key, and it stays
        compatible with foreign-key checks while child rows are inserted.
        """
        query = select(sessions).where(
            and_(sessions.c.id == session_id, sessions.c.owner_id == owner_id)
        )
        if self._conn.dialect.name == "postgresql":
            # ``set_config(..., true)`` is parameterised and transaction-local
            # (unlike interpolating a ``SET LOCAL`` command), so a pooled
            # connection cannot leak this admission timeout into another
            # request once the UoW commits or rolls back.
            await self._conn.execute(
                select(func.set_config("lock_timeout", f"{max(1, lock_timeout_ms)}ms", True))
            )
            query = query.with_for_update(key_share=True)
        row = (await self._conn.execute(query)).mappings().first()
        return m.row_to_session(row) if row else None

    async def find_by_client_id(self, owner_id: str, client_session_id: str) -> Session | None:
        row = (
            await self._conn.execute(
                select(sessions).where(
                    and_(
                        sessions.c.owner_id == owner_id,
                        sessions.c.client_session_id == client_session_id,
                    )
                )
            )
        ).mappings().first()
        return m.row_to_session(row) if row else None

    async def update(self, session: Session, *, expected_version: int) -> None:
        row = m.session_to_row(session)
        row.pop("client_session_id")
        row.pop("id")
        # event_sequence belongs to the unit of work's allocator, not to a
        # caller holding a stale copy of the aggregate.
        row.pop("event_sequence")
        result = await self._conn.execute(
            update(sessions)
            .where(and_(sessions.c.id == session.id, sessions.c.version == expected_version))
            .values(**row)
        )
        if result.rowcount == 0:
            raise Conflict(
                "session changed underneath this write",
                session_id=session.id,
                expected_version=expected_version,
            )

    async def list_for_owner(self, owner_id: str, *, limit: int, offset: int) -> Sequence[Session]:
        rows = (
            await self._conn.execute(
                select(sessions)
                .where(sessions.c.owner_id == owner_id)
                .order_by(sessions.c.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
        ).mappings().all()
        return [m.row_to_session(r) for r in rows]


class SqlMessageStore:
    def __init__(self, conn: AsyncConnection) -> None:
        self._conn = conn

    async def add(self, message: Message) -> None:
        await self._conn.execute(insert(messages).values(m.message_to_row(message)))
        if message.parts:
            await self._conn.execute(
                insert(message_parts), [m.part_to_row(p) for p in message.parts]
            )

    async def get(self, message_id: str) -> Message | None:
        row = (
            await self._conn.execute(select(messages).where(messages.c.id == message_id))
        ).mappings().first()
        if row is None:
            return None
        parts = (
            await self._conn.execute(
                select(message_parts).where(message_parts.c.message_id == message_id)
            )
        ).mappings().all()
        return m.row_to_message(row, list(parts))

    async def find_by_client_id(self, session_id: str, client_message_id: str) -> Message | None:
        row = (
            await self._conn.execute(
                select(messages).where(
                    and_(
                        messages.c.session_id == session_id,
                        messages.c.client_message_id == client_message_id,
                    )
                )
            )
        ).mappings().first()
        return await self.get(row["id"]) if row else None

    async def list_for_session(
        self, session_id: str, *, after_sequence: int = 0, limit: int = 100
    ) -> Sequence[Message]:
        rows = (
            await self._conn.execute(
                select(messages)
                .where(
                    and_(messages.c.session_id == session_id, messages.c.sequence > after_sequence)
                )
                .order_by(messages.c.sequence)
                .limit(limit)
            )
        ).mappings().all()
        if not rows:
            return []
        ids = [r["id"] for r in rows]
        part_rows = (
            await self._conn.execute(
                select(message_parts).where(message_parts.c.message_id.in_(ids))
            )
        ).mappings().all()
        by_message: dict[str, list] = {i: [] for i in ids}
        for p in part_rows:
            by_message[p["message_id"]].append(p)
        return [m.row_to_message(r, by_message[r["id"]]) for r in rows]

    async def next_sequence(self, session_id: str) -> int:
        current = (
            await self._conn.execute(
                select(func.max(messages.c.sequence)).where(messages.c.session_id == session_id)
            )
        ).scalar()
        return (current or 0) + 1

    async def append_part(
        self, message_id: str, index: int, part_type: str, content: dict[str, Any]
    ) -> str:
        from ...domain.ids import PART, new_id

        part_id = new_id(PART)
        await self._conn.execute(
            insert(message_parts).values(
                id=part_id, message_id=message_id, index=index,
                type=part_type, content=content, version=1,
            )
        )
        return part_id


class SqlRunStore:
    def __init__(self, conn: AsyncConnection) -> None:
        self._conn = conn

    async def add(self, run: Run) -> None:
        await self._conn.execute(insert(runs).values(m.run_to_row(run)))

    async def get(self, run_id: str) -> Run | None:
        row = (await self._conn.execute(select(runs).where(runs.c.id == run_id))).mappings().first()
        return m.row_to_run(row) if row else None

    async def update(self, run: Run, *, expected_version: int) -> None:
        row = m.run_to_row(run)
        row.pop("id")
        result = await self._conn.execute(
            update(runs)
            .where(and_(runs.c.id == run.id, runs.c.version == expected_version))
            .values(**row)
        )
        if result.rowcount == 0:
            raise Conflict("run changed underneath this write", run_id=run.id)

    async def list_for_session(self, session_id: str, *, limit: int = 50) -> Sequence[Run]:
        rows = (
            await self._conn.execute(
                select(runs)
                .where(runs.c.session_id == session_id)
                .order_by(runs.c.created_at.desc())
                .limit(limit)
            )
        ).mappings().all()
        return [m.row_to_run(r) for r in rows]

    async def list_non_terminal(self, *, limit: int = 1000) -> Sequence[Run]:
        """Every run still ``queued``/``running``/``validating``, across every
        session — used only by startup reconciliation, where "unowned by any
        in-process task" and "not terminal" are the same set (plan section
        6.5's cancellation policy assumes exactly one process owns a run)."""
        rows = (
            await self._conn.execute(
                select(runs)
                .where(runs.c.status.in_(("queued", "running", "validating")))
                .order_by(runs.c.created_at)
                .limit(limit)
            )
        ).mappings().all()
        return [m.row_to_run(r) for r in rows]

    async def request_cancel(self, run_id: str) -> bool:
        """Flag a cancellation request. Whether it is honoured, and how, is the
        gateway's business; this only records that it was asked for."""
        result = await self._conn.execute(
            update(runs)
            .where(and_(runs.c.id == run_id, runs.c.status.in_(("queued", "running", "validating"))))
            .values(cancel_requested=True)
        )
        return result.rowcount > 0

    async def cancel_requested(self, run_id: str) -> bool:
        return bool(
            (
                await self._conn.execute(
                    select(runs.c.cancel_requested).where(runs.c.id == run_id)
                )
            ).scalar()
        )


class SqlAnalysisStore:
    def __init__(self, conn: AsyncConnection) -> None:
        self._conn = conn

    async def add(self, snapshot: AnalysisSnapshot) -> None:
        await self._conn.execute(insert(analysis_snapshots).values(m.analysis_to_row(snapshot)))

    async def get(self, analysis_id: str, *, session_id: str) -> AnalysisSnapshot | None:
        row = (
            await self._conn.execute(
                select(analysis_snapshots).where(
                    and_(
                        analysis_snapshots.c.id == analysis_id,
                        analysis_snapshots.c.session_id == session_id,
                    )
                )
            )
        ).mappings().first()
        return m.row_to_analysis(row) if row else None

    async def find_by_idempotency_key(
        self, session_id: str, idempotency_key: str
    ) -> AnalysisSnapshot | None:
        row = (
            await self._conn.execute(
                select(analysis_snapshots).where(
                    and_(
                        analysis_snapshots.c.session_id == session_id,
                        analysis_snapshots.c.idempotency_key == idempotency_key,
                    )
                )
            )
        ).mappings().first()
        return m.row_to_analysis(row) if row else None

    async def list_for_session(
        self, session_id: str, *, limit: int = 50
    ) -> Sequence[AnalysisSnapshot]:
        rows = (
            await self._conn.execute(
                select(analysis_snapshots)
                .where(analysis_snapshots.c.session_id == session_id)
                .order_by(analysis_snapshots.c.created_at.desc())
                .limit(limit)
            )
        ).mappings().all()
        return [m.row_to_analysis(r) for r in rows]


class SqlObservationStore:
    def __init__(self, conn: AsyncConnection) -> None:
        self._conn = conn

    async def add(self, observation: Observation, *, analysis_id: str | None = None) -> None:
        await self._conn.execute(
            insert(observations).values(m.observation_to_row(observation, analysis_id))
        )

    async def get(self, observation_id: str, *, session_id: str) -> Observation | None:
        row = (
            await self._conn.execute(
                select(observations).where(
                    and_(
                        observations.c.id == observation_id,
                        observations.c.session_id == session_id,
                    )
                )
            )
        ).mappings().first()
        return m.row_to_observation(row) if row else None

    async def list_for_run(self, run_id: str) -> Sequence[Observation]:
        rows = (
            await self._conn.execute(
                select(observations)
                .where(observations.c.run_id == run_id)
                .order_by(observations.c.created_at)
            )
        ).mappings().all()
        return [m.row_to_observation(r) for r in rows]

    async def list_for_analysis(self, analysis_id: str) -> Sequence[Observation]:
        rows = (
            await self._conn.execute(
                select(observations)
                .where(observations.c.analysis_id == analysis_id)
                .order_by(observations.c.created_at)
            )
        ).mappings().all()
        return [m.row_to_observation(r) for r in rows]


class SqlEvidenceStore:
    def __init__(self, conn: AsyncConnection) -> None:
        self._conn = conn

    async def add(self, record: EvidenceRecord) -> None:
        await self._conn.execute(insert(evidence_records).values(m.evidence_to_row(record)))

    async def get(self, evidence_id: str, *, session_id: str) -> EvidenceRecord | None:
        row = (
            await self._conn.execute(
                select(evidence_records).where(
                    and_(
                        evidence_records.c.id == evidence_id,
                        evidence_records.c.session_id == session_id,
                    )
                )
            )
        ).mappings().first()
        return m.row_to_evidence(row) if row else None

    async def find_by_dedupe_key(self, session_id: str, dedupe_key: str) -> EvidenceRecord | None:
        row = (
            await self._conn.execute(
                select(evidence_records).where(
                    and_(
                        evidence_records.c.session_id == session_id,
                        evidence_records.c.dedupe_key == dedupe_key,
                    )
                )
            )
        ).mappings().first()
        return m.row_to_evidence(row) if row else None

    async def set_status(
        self, evidence_id: str, status: EvidenceStatus, *, reason: str | None = None
    ) -> None:
        await self._conn.execute(
            update(evidence_records)
            .where(evidence_records.c.id == evidence_id)
            .values(status=status.value, rejection_reason=reason)
        )

    async def list_for_session(
        self,
        session_id: str,
        *,
        status: EvidenceStatus | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Sequence[EvidenceRecord]:
        query = select(evidence_records).where(evidence_records.c.session_id == session_id)
        if status is not None:
            query = query.where(evidence_records.c.status == status.value)
        rows = (
            await self._conn.execute(
                query.order_by(evidence_records.c.retrieved_at.desc()).limit(limit).offset(offset)
            )
        ).mappings().all()
        return [m.row_to_evidence(r) for r in rows]


class SqlAnswerStore:
    def __init__(self, conn: AsyncConnection) -> None:
        self._conn = conn

    async def add(self, answer: GroundedAnswer) -> None:
        await self._conn.execute(insert(answers).values(m.answer_to_row(answer)))
        if not answer.claims:
            return
        await self._conn.execute(
            insert(claims),
            [m.claim_to_row(c, answer.id, i) for i, c in enumerate(answer.claims)],
        )
        citations = [
            {"claim_id": c.claim_id, "evidence_id": e}
            for c in answer.claims
            for e in c.citation_ids
        ]
        if citations:
            await self._conn.execute(insert(claim_sources), citations)

    async def _load(self, row) -> GroundedAnswer:
        claim_rows = (
            await self._conn.execute(
                select(claims).where(claims.c.answer_id == row["id"]).order_by(claims.c.position)
            )
        ).mappings().all()
        citation_rows = (
            await self._conn.execute(
                select(claim_sources).where(
                    claim_sources.c.claim_id.in_([c["id"] for c in claim_rows] or [""])
                )
            )
        ).mappings().all()
        by_claim: dict[str, list[str]] = {}
        for c in citation_rows:
            by_claim.setdefault(c["claim_id"], []).append(c["evidence_id"])
        return m.row_to_answer(
            row,
            tuple(m.row_to_claim(c, tuple(sorted(by_claim.get(c["id"], [])))) for c in claim_rows),
        )

    async def get(self, answer_id: str, *, session_id: str) -> GroundedAnswer | None:
        row = (
            await self._conn.execute(
                select(answers).where(
                    and_(answers.c.id == answer_id, answers.c.session_id == session_id)
                )
            )
        ).mappings().first()
        return await self._load(row) if row else None

    async def get_for_run(self, run_id: str) -> GroundedAnswer | None:
        row = (
            await self._conn.execute(
                select(answers)
                .where(answers.c.run_id == run_id)
                .order_by(answers.c.candidate_generation.desc())
                .limit(1)
            )
        ).mappings().first()
        return await self._load(row) if row else None

    async def candidate_generations(self, run_id: str) -> int:
        return int(
            (
                await self._conn.execute(
                    select(func.count()).select_from(answers).where(answers.c.run_id == run_id)
                )
            ).scalar()
            or 0
        )

    async def claims_for(self, answer_id: str) -> Sequence[Claim]:
        rows = (
            await self._conn.execute(
                select(claims).where(claims.c.answer_id == answer_id).order_by(claims.c.position)
            )
        ).mappings().all()
        return [m.row_to_claim(r) for r in rows]

    async def claim_id_exists(self, claim_id: str) -> bool:
        row = (
            await self._conn.execute(
                select(literal(1)).select_from(claims).where(claims.c.id == claim_id).limit(1)
            )
        ).first()
        return row is not None


class SqlRuntimeBindingStore:
    def __init__(self, conn: AsyncConnection) -> None:
        self._conn = conn

    async def add(self, binding: RuntimeBinding) -> None:
        await self._conn.execute(insert(runtime_bindings).values(m.binding_to_row(binding)))

    async def get(self, binding_id: str) -> RuntimeBinding | None:
        row = (
            await self._conn.execute(
                select(runtime_bindings).where(runtime_bindings.c.id == binding_id)
            )
        ).mappings().first()
        return m.row_to_binding(row) if row else None

    async def active_for_session(self, session_id: str) -> RuntimeBinding | None:
        row = (
            await self._conn.execute(
                select(runtime_bindings)
                .where(
                    and_(
                        runtime_bindings.c.session_id == session_id,
                        runtime_bindings.c.status == "active",
                    )
                )
                .order_by(runtime_bindings.c.created_at.desc())
                .limit(1)
            )
        ).mappings().first()
        return m.row_to_binding(row) if row else None

    async def set_status(self, binding_id: str, status: str, *, now: datetime) -> None:
        await self._conn.execute(
            update(runtime_bindings)
            .where(runtime_bindings.c.id == binding_id)
            .values(status=status, closed_at=now)
        )


class SqlRuntimeUsageStore:
    def __init__(self, conn: AsyncConnection) -> None:
        self._conn = conn

    async def add(self, event: RuntimeUsageEvent) -> None:
        await self._conn.execute(insert(runtime_usage_events).values(m.usage_to_row(event)))

    async def list_for_run(self, run_id: str) -> Sequence[RuntimeUsageEvent]:
        rows = (
            await self._conn.execute(
                select(runtime_usage_events)
                .where(runtime_usage_events.c.run_id == run_id)
                .order_by(runtime_usage_events.c.reported_at, runtime_usage_events.c.id)
            )
        ).mappings().all()
        return [m.row_to_usage(row) for row in rows]


class SqlToolCallStore:
    def __init__(self, conn: AsyncConnection) -> None:
        self._conn = conn

    async def try_reserve(
        self, *, call_id: str, session_id: str, run_id: str, tool_name: str,
        arguments_sha256: str, now: datetime, max_calls: int | None, max_identical: int,
    ) -> bool:
        """Atomically check the per-run budget and duplicate-call cap and, if
        both allow it, reserve the call by inserting its ``running`` row — all
        as one ``INSERT ... SELECT ... WHERE`` statement, so the count and the
        reservation cannot be observed and acted on separately by two
        concurrent calls (the race a plain "SELECT count, then INSERT" allows:
        several callers can each see room under the budget before any of them
        has actually taken a slot). ``status='denied'`` rows are excluded from
        both counts — a denied attempt is kept for audit but must not itself
        shrink the budget for the next real attempt. ``max_calls=None`` (used
        for the final-answer tool) skips the budget check entirely; the
        duplicate-call cap still applies.

        Returns whether the reservation succeeded.
        """
        not_denied = tool_calls.c.status != "denied"
        conditions = [
            (
                select(func.count())
                .select_from(tool_calls)
                .where(and_(tool_calls.c.run_id == run_id, not_denied))
                .scalar_subquery()
                < max_calls
            )
        ] if max_calls is not None else []
        conditions.append(
            select(func.count())
            .select_from(tool_calls)
            .where(
                and_(
                    tool_calls.c.run_id == run_id,
                    tool_calls.c.tool_name == tool_name,
                    tool_calls.c.arguments_sha256 == arguments_sha256,
                    not_denied,
                )
            )
            .scalar_subquery()
            < max_identical
        )

        source = select(
            literal(call_id), literal(session_id), literal(run_id), literal(tool_name),
            literal(arguments_sha256), literal("running"),
            literal([], type_=tool_calls.c.observation_ids.type), literal(now),
        ).where(and_(*conditions))
        stmt = insert(tool_calls).from_select(
            ["id", "session_id", "run_id", "tool_name", "arguments_sha256", "status",
             "observation_ids", "started_at"],
            source,
        )
        result = await self._conn.execute(stmt)
        return result.rowcount == 1

    async def record_denied(
        self, *, call_id: str, session_id: str, run_id: str, tool_name: str,
        arguments_sha256: str, error_code: str, now: datetime,
    ) -> None:
        """A denied call still leaves an audit trail, distinct from ``running``/
        ``completed``/``error`` (which describe a call that was actually
        admitted) so ``count_for_run``-style budget accounting can exclude it
        while `get_run`'s tool-call listing still shows every attempt."""
        await self._conn.execute(
            insert(tool_calls).values(
                id=call_id, session_id=session_id, run_id=run_id, tool_name=tool_name,
                arguments_sha256=arguments_sha256, status="denied", error_code=error_code,
                observation_ids=[], started_at=now, ended_at=now, duration_ms=0,
            )
        )

    async def finish(
        self, call_id: str, *, status: str, error_code: str | None,
        observation_ids: list[str], duration_ms: int, now: datetime,
    ) -> None:
        await self._conn.execute(
            update(tool_calls)
            .where(tool_calls.c.id == call_id)
            .values(
                status=status, error_code=error_code, observation_ids=observation_ids,
                duration_ms=duration_ms, ended_at=now,
            )
        )

    async def count_for_run(self, run_id: str) -> int:
        """Admitted calls only — same accounting ``try_reserve`` enforces, so
        a denied attempt never itself counts against the budget it was
        denied under."""
        return int(
            (
                await self._conn.execute(
                    select(func.count())
                    .select_from(tool_calls)
                    .where(and_(tool_calls.c.run_id == run_id, tool_calls.c.status != "denied"))
                )
            ).scalar()
            or 0
        )

    async def duplicate_count(self, run_id: str, tool_name: str, arguments_sha256: str) -> int:
        return int(
            (
                await self._conn.execute(
                    select(func.count())
                    .select_from(tool_calls)
                    .where(
                        and_(
                            tool_calls.c.run_id == run_id,
                            tool_calls.c.tool_name == tool_name,
                            tool_calls.c.arguments_sha256 == arguments_sha256,
                            tool_calls.c.status != "denied",
                        )
                    )
                )
            ).scalar()
            or 0
        )

    async def list_for_run(self, run_id: str) -> Sequence[dict[str, Any]]:
        rows = (
            await self._conn.execute(
                select(tool_calls)
                .where(tool_calls.c.run_id == run_id)
                .order_by(tool_calls.c.started_at)
            )
        ).mappings().all()
        # Unlike every other repository, these rows go straight out as raw
        # dicts rather than through a row_to_* mapper — so they were the one
        # place SQLite's naive (no-tzinfo) datetimes reached a client
        # unnormalized, rendering as local time in whatever timezone the
        # browser happened to be in instead of UTC.
        results = [dict(r) for r in rows]
        for row in results:
            row["started_at"] = m.utc(row["started_at"])
            row["ended_at"] = m.utc(row["ended_at"])
        return results


class SqlCapabilityTokenStore:
    def __init__(self, conn: AsyncConnection) -> None:
        self._conn = conn

    async def issue(
        self, *, jti: str, session_id: str, run_id: str, runtime_binding_id: str | None,
        allowed_tools: list[str], issued_at: datetime, expires_at: datetime,
    ) -> None:
        await self._conn.execute(
            insert(capability_tokens).values(
                jti=jti, session_id=session_id, run_id=run_id,
                runtime_binding_id=runtime_binding_id, allowed_tools=allowed_tools,
                issued_at=issued_at, expires_at=expires_at,
            )
        )

    async def is_valid(self, jti: str, *, now: datetime) -> bool:
        row = (
            await self._conn.execute(
                select(capability_tokens).where(capability_tokens.c.jti == jti)
            )
        ).mappings().first()
        if row is None or row["revoked_at"] is not None:
            return False
        return m.utc(row["expires_at"]) > now

    async def revoke(self, jti: str, *, now: datetime) -> None:
        await self._conn.execute(
            update(capability_tokens)
            .where(capability_tokens.c.jti == jti)
            .values(revoked_at=now)
        )


class SqlAttachmentStore:
    def __init__(self, conn: AsyncConnection) -> None:
        self._conn = conn

    async def add(self, attachment: Attachment) -> None:
        await self._conn.execute(insert(attachments).values(m.attachment_to_row(attachment)))

    async def get(self, attachment_id: str, *, owner_id: str) -> Attachment | None:
        row = (
            await self._conn.execute(
                select(attachments).where(
                    and_(attachments.c.id == attachment_id, attachments.c.owner_id == owner_id)
                )
            )
        ).mappings().first()
        return m.row_to_attachment(row) if row else None

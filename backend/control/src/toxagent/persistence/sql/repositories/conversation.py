"""Sessions, their settings, messages, attachments and grounded answers."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Sequence

from sqlalchemy import and_, func, insert, literal, select, update
from sqlalchemy.ext.asyncio import AsyncConnection

from ....domain.answer import Claim, GroundedAnswer
from ....domain.attachment import Attachment
from ....domain.errors import Conflict
from ....domain.message import Message
from ....domain.session import Session
from ...schema import (
    answers,
    attachments,
    claim_sources,
    claims,
    message_parts,
    messages,
    session_settings,
    sessions,
)
from .. import mapping as m


class SqlSessionSettingsStore:
    def __init__(self, conn: AsyncConnection) -> None:
        self._conn = conn

    async def get(self, session_id: str) -> dict[str, Any]:
        row = (await self._conn.execute(select(session_settings).where(
            session_settings.c.session_id == session_id
        ))).mappings().first()
        if row is None:
            return {"ai_profile_id": None, "predictor_bindings": {}}
        return {"ai_profile_id": row["ai_profile_id"], "predictor_bindings": dict(row["predictor_bindings"] or {})}

    async def put(self, session_id: str, *, ai_profile_id: str | None,
                  predictor_bindings: dict[str, str], now: datetime) -> None:
        values = {"session_id": session_id, "ai_profile_id": ai_profile_id,
                  "predictor_bindings": dict(predictor_bindings), "updated_at": now}
        existing = await self._conn.execute(select(session_settings.c.session_id).where(
            session_settings.c.session_id == session_id
        ))
        if existing.scalar() is None:
            await self._conn.execute(insert(session_settings).values(**values))
        else:
            await self._conn.execute(update(session_settings).where(
                session_settings.c.session_id == session_id
            ).values(**{k: v for k, v in values.items() if k != "session_id"}))


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

    async def latest_previews(
        self, session_ids: Sequence[str], *, max_length: int = 160
    ) -> dict[str, str]:
        """The newest message's text, per session, in two queries for the page.

        I20: the session list used to read the first 50 messages of each
        session ordered by ascending sequence and take the last of them, which
        is the newest message only while a session has 50 or fewer. Past that
        the preview froze on message 50 — and it cost one message query plus
        one parts query per session in the page.
        """
        if not session_ids:
            return {}
        newest = (
            select(messages.c.session_id, func.max(messages.c.sequence).label("sequence"))
            .where(messages.c.session_id.in_(session_ids))
            .group_by(messages.c.session_id)
            .subquery()
        )
        rows = (
            await self._conn.execute(
                select(messages.c.id, messages.c.session_id).join(
                    newest,
                    and_(
                        messages.c.session_id == newest.c.session_id,
                        messages.c.sequence == newest.c.sequence,
                    ),
                )
            )
        ).mappings().all()
        if not rows:
            return {}
        session_by_message = {row["id"]: row["session_id"] for row in rows}
        part_rows = (
            await self._conn.execute(
                select(message_parts)
                .where(and_(
                    message_parts.c.message_id.in_(list(session_by_message)),
                    message_parts.c.type == "text",
                ))
                .order_by(message_parts.c.index)
            )
        ).mappings().all()
        previews: dict[str, str] = {}
        for part in part_rows:
            session_id = session_by_message[part["message_id"]]
            if session_id in previews:
                continue  # first text part of that message, matching the old read
            text = str((part["content"] or {}).get("text", "")).strip()
            if text:
                previews[session_id] = text if len(text) <= max_length else f"{text[:max_length]}…"
        return previews

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
        from ....domain.ids import PART, new_id

        part_id = new_id(PART)
        await self._conn.execute(
            insert(message_parts).values(
                id=part_id, message_id=message_id, index=index,
                type=part_type, content=content, version=1,
            )
        )
        return part_id


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

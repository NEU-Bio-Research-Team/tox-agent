"""Frozen activity screenings attached to sessions (#44)."""
from __future__ import annotations

from typing import Sequence

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncConnection

from ....domain.screening_attachment import ScreeningAttachment
from ...schema import session_screenings
from .. import mapping as m


def _row_to_attachment(row) -> ScreeningAttachment:
    return ScreeningAttachment(
        session_id=row["session_id"],
        screening_id=row["screening_id"],
        record=dict(row["record"]),
        record_sha256=row["record_sha256"],
        attached_at=m.utc(row["attached_at"]),
        source=row["source"],
    )


class SqlScreeningAttachmentStore:
    def __init__(self, conn: AsyncConnection) -> None:
        self._conn = conn

    async def add_if_absent(self, attachment: ScreeningAttachment) -> ScreeningAttachment:
        """Attach once. A second attach of the same screening returns the
        first, unchanged: what a run already read must not be swapped."""
        if self._conn.dialect.name == "postgresql":
            from sqlalchemy.dialects.postgresql import insert as dialect_insert
        else:
            from sqlalchemy.dialects.sqlite import insert as dialect_insert
        inserted = await self._conn.execute(
            dialect_insert(session_screenings)
            .values(
                session_id=attachment.session_id,
                screening_id=attachment.screening_id,
                record=dict(attachment.record),
                record_sha256=attachment.record_sha256,
                source=attachment.source,
                attached_at=attachment.attached_at,
            )
            .on_conflict_do_nothing(index_elements=["session_id", "screening_id"])
        )
        if inserted.rowcount > 0:
            return attachment
        existing = await self.get(attachment.screening_id, session_id=attachment.session_id)
        return existing if existing is not None else attachment

    async def get(self, screening_id: str, *, session_id: str) -> ScreeningAttachment | None:
        row = (
            await self._conn.execute(
                select(session_screenings).where(
                    and_(
                        session_screenings.c.session_id == session_id,
                        session_screenings.c.screening_id == screening_id,
                    )
                )
            )
        ).mappings().first()
        return _row_to_attachment(row) if row else None

    async def list_for_session(self, session_id: str) -> Sequence[ScreeningAttachment]:
        rows = (
            await self._conn.execute(
                select(session_screenings)
                .where(session_screenings.c.session_id == session_id)
                .order_by(session_screenings.c.attached_at, session_screenings.c.screening_id)
            )
        ).mappings().all()
        return [_row_to_attachment(row) for row in rows]

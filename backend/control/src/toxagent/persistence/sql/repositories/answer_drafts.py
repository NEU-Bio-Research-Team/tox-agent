"""Every draft a run submitted, as the model sent it (#47)."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Sequence

from sqlalchemy import insert, select
from sqlalchemy.ext.asyncio import AsyncConnection

from ...schema import answer_drafts
from .. import mapping as m


class SqlAnswerDraftStore:
    def __init__(self, conn: AsyncConnection) -> None:
        self._conn = conn

    async def add(
        self, *, run_id: str, session_id: str, generation: int, schema_version: str,
        payload: dict[str, Any], outcome: str, answer_id: str | None,
        violations: list[dict[str, Any]], now: datetime, phase: str | None = None,
    ) -> None:
        await self._conn.execute(
            insert(answer_drafts).values(
                run_id=run_id, generation=generation, session_id=session_id,
                schema_version=schema_version, payload=payload, outcome=outcome,
                answer_id=answer_id, violations=violations, created_at=now, phase=phase,
            )
        )

    async def list_for_run(self, run_id: str) -> Sequence[dict[str, Any]]:
        rows = (
            await self._conn.execute(
                select(answer_drafts)
                .where(answer_drafts.c.run_id == run_id)
                .order_by(answer_drafts.c.generation)
            )
        ).mappings().all()
        return [
            {**dict(row), "created_at": m.utc(row["created_at"]).isoformat()} for row in rows
        ]

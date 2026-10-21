"""How an answer was released (#47)."""
from __future__ import annotations

from sqlalchemy import insert, select
from sqlalchemy.ext.asyncio import AsyncConnection

from ....domain.answer_release import AnswerRelease, ReleaseOutcome
from ...schema import answer_releases
from .. import mapping as m


def _row_to_release(row) -> AnswerRelease:
    detail = dict(row["detail"] or {})
    return AnswerRelease(
        answer_id=row["answer_id"], run_id=row["run_id"], session_id=row["session_id"],
        outcome=ReleaseOutcome(row["outcome"]), released_at=m.utc(row["released_at"]),
        reviews=tuple(detail.get("reviews") or ()), repair=detail.get("repair"),
        unresolved_claim_ids=tuple(detail.get("unresolved_claim_ids") or ()),
        ungated_reason=detail.get("ungated_reason"),
        replaced_answer_id=detail.get("replaced_answer_id"),
    )


class SqlAnswerReleaseStore:
    def __init__(self, conn: AsyncConnection) -> None:
        self._conn = conn

    async def add(self, release: AnswerRelease) -> None:
        detail = release.to_dict()
        await self._conn.execute(
            insert(answer_releases).values(
                answer_id=release.answer_id, run_id=release.run_id,
                session_id=release.session_id, outcome=release.outcome.value,
                detail={k: detail[k] for k in (
                    "reviews", "repair", "unresolved_claim_ids", "ungated_reason",
                    "replaced_answer_id",
                )},
                released_at=release.released_at,
            )
        )

    async def get(self, answer_id: str) -> AnswerRelease | None:
        row = (
            await self._conn.execute(
                select(answer_releases).where(answer_releases.c.answer_id == answer_id)
            )
        ).mappings().first()
        return _row_to_release(row) if row else None

    async def get_for_run(self, run_id: str) -> AnswerRelease | None:
        row = (
            await self._conn.execute(
                select(answer_releases).where(answer_releases.c.run_id == run_id)
            )
        ).mappings().first()
        return _row_to_release(row) if row else None

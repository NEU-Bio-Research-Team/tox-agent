"""What a decision-support run ended with (#47, run-result-v1)."""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncConnection

from ....domain.run_result import (
    Disposition,
    ExecutionOutcome,
    KnowledgeOutcome,
    ReleasePolicy,
    ReviewStatus,
    RunResultV1,
)
from ...schema import run_results
from .. import mapping as m


def _row_to_result(row) -> RunResultV1:
    return RunResultV1(
        run_id=row["run_id"], session_id=row["session_id"],
        disposition=Disposition(row["disposition"]),
        review_status=ReviewStatus(row["review_status"]),
        execution_outcome=ExecutionOutcome(row["execution_outcome"]),
        knowledge_outcome=KnowledgeOutcome(row["knowledge_outcome"]),
        release_policy=ReleasePolicy(row["release_policy"]),
        public_answer_id=row["public_answer_id"],
        candidate_answer_id=row["candidate_answer_id"],
        detail=dict(row["detail"] or {}), created_at=m.utc(row["created_at"]),
    )


class SqlRunResultStore:
    def __init__(self, conn: AsyncConnection) -> None:
        self._conn = conn

    async def add_if_absent(self, result: RunResultV1) -> RunResultV1:
        """One terminal result per run. A retried commit, or a second worker
        racing the first, gets back the result that was written."""
        if self._conn.dialect.name == "postgresql":
            from sqlalchemy.dialects.postgresql import insert as dialect_insert
        else:
            from sqlalchemy.dialects.sqlite import insert as dialect_insert
        await self._conn.execute(
            dialect_insert(run_results)
            .values(
                run_id=result.run_id, session_id=result.session_id,
                disposition=result.disposition.value, review_status=result.review_status.value,
                execution_outcome=result.execution_outcome.value,
                knowledge_outcome=result.knowledge_outcome.value,
                release_policy=result.release_policy.value,
                release_policy_sha256=result.policy_sha256,
                public_answer_id=result.public_answer_id,
                candidate_answer_id=result.candidate_answer_id,
                detail=dict(result.detail), created_at=result.created_at,
            )
            .on_conflict_do_nothing(index_elements=["run_id"])
        )
        stored = await self.get(result.run_id)
        assert stored is not None
        return stored

    async def get(self, run_id: str) -> RunResultV1 | None:
        row = (
            await self._conn.execute(select(run_results).where(run_results.c.run_id == run_id))
        ).mappings().first()
        return _row_to_result(row) if row else None

    async def get_for_public_answer(self, answer_id: str) -> RunResultV1 | None:
        row = (
            await self._conn.execute(
                select(run_results).where(run_results.c.public_answer_id == answer_id)
            )
        ).mappings().first()
        return _row_to_result(row) if row else None

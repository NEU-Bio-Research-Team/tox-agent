"""Analyses, their observations and explanation checkpoints."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Sequence

from sqlalchemy import and_, insert, select
from sqlalchemy.ext.asyncio import AsyncConnection

from ....domain.analysis import AnalysisSnapshot
from ....domain.observation import Observation
from ...schema import (
    analysis_snapshots,
    explanation_checkpoints,
    observations,
)
from .. import mapping as m


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


class SqlExplanationCheckpointStore:
    """Explanations that have already been computed and committed.

    I19: a bundle ran predict and every requested explanation before writing
    anything, so a crash after the fifth of eight targets discarded all five,
    and the retry paid for them again. Each target is committed as it lands
    here instead, addressed by what it is an explanation *of* — the canonical
    molecule, the endpoint, the task and the model that produced the
    probability. A retry looks the completed ones up and does only what is
    left.

    Written once and read back; there is no update path, because an
    explanation of a fixed input by a fixed model does not change.
    """

    def __init__(self, conn: AsyncConnection) -> None:
        self._conn = conn

    async def get_many(self, keys: Sequence[str]) -> dict[str, dict[str, Any]]:
        if not keys:
            return {}
        rows = (
            await self._conn.execute(
                select(explanation_checkpoints.c.key, explanation_checkpoints.c.payload)
                .where(explanation_checkpoints.c.key.in_(list(keys)))
            )
        ).mappings().all()
        return {row["key"]: row["payload"] for row in rows}

    async def put(
        self,
        key: str,
        *,
        session_id: str,
        endpoint: str,
        task: str | None,
        model_id: str | None,
        canonical_smiles: str,
        payload: dict[str, Any],
        now: datetime,
    ) -> None:
        """Record one completed explanation.

        A duplicate key means another attempt committed the same artifact
        first; that is the checkpoint working, not a conflict, so the insert
        is skipped rather than raised.
        """
        existing = (
            await self._conn.execute(
                select(explanation_checkpoints.c.key)
                .where(explanation_checkpoints.c.key == key)
            )
        ).scalar()
        if existing is not None:
            return
        await self._conn.execute(
            insert(explanation_checkpoints).values(
                key=key, session_id=session_id, endpoint=endpoint, task=task,
                model_id=model_id, canonical_smiles=canonical_smiles, payload=payload,
                created_at=now,
            )
        )

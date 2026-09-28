"""Evidence records and evidence-relation assessments."""
from __future__ import annotations

from typing import Sequence

from sqlalchemy import and_, insert, select, update
from sqlalchemy.ext.asyncio import AsyncConnection

from ....domain.errors import Conflict
from ....domain.evidence import EvidenceRecord, EvidenceStatus
from ....domain.evidence_relation import EvidenceRelationAssessment
from ...schema import (
    evidence_records,
    evidence_relation_assessments,
)
from .. import mapping as m


class SqlEvidenceStore:
    def __init__(self, conn: AsyncConnection) -> None:
        self._conn = conn

    async def add(self, record: EvidenceRecord) -> None:
        await self._conn.execute(insert(evidence_records).values(m.evidence_to_row(record)))

    async def add_if_absent(self, record: EvidenceRecord) -> EvidenceRecord:
        """Insert ``record`` unless the session already holds its dedupe key,
        and return whichever record the session now has. Two searches a model
        issues in parallel can both miss ``find_by_dedupe_key`` for the same
        paper; a plain insert then failed the second tool call on
        ``uq_evidence_dedupe`` (live, 2026-09-26)."""
        if self._conn.dialect.name == "postgresql":
            from sqlalchemy.dialects.postgresql import insert as dialect_insert
        else:
            from sqlalchemy.dialects.sqlite import insert as dialect_insert
        inserted = await self._conn.execute(
            dialect_insert(evidence_records)
            .values(m.evidence_to_row(record))
            .on_conflict_do_nothing(index_elements=["session_id", "dedupe_key"])
        )
        if inserted.rowcount > 0:
            return record
        existing = await self.find_by_dedupe_key(record.session_id, record.dedupe_key)
        if existing is None:  # the conflict was on something other than the dedupe key
            raise Conflict("evidence record could not be stored", evidence_id=record.id)
        return existing

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


class SqlEvidenceRelationStore:
    """One source-vs-proposition assessment per row (ADS plan section 9.3,
    ADR 0010). Append-only: a relation is never edited in place, a later
    reassessment is a new row for the same proposition."""

    def __init__(self, conn: AsyncConnection) -> None:
        self._conn = conn

    async def add(self, assessment: EvidenceRelationAssessment) -> None:
        await self._conn.execute(
            insert(evidence_relation_assessments).values(
                m.evidence_relation_to_row(assessment)
            )
        )

    async def list_for_run(self, run_id: str) -> Sequence[EvidenceRelationAssessment]:
        rows = (
            await self._conn.execute(
                select(evidence_relation_assessments)
                .where(evidence_relation_assessments.c.run_id == run_id)
                .order_by(evidence_relation_assessments.c.created_at)
            )
        ).mappings().all()
        return [m.row_to_evidence_relation(r) for r in rows]

    async def list_for_proposition(
        self, session_id: str, proposition_id: str
    ) -> Sequence[EvidenceRelationAssessment]:
        rows = (
            await self._conn.execute(
                select(evidence_relation_assessments).where(
                    and_(
                        evidence_relation_assessments.c.session_id == session_id,
                        evidence_relation_assessments.c.proposition_id == proposition_id,
                    )
                ).order_by(evidence_relation_assessments.c.created_at)
            )
        ).mappings().all()
        return [m.row_to_evidence_relation(r) for r in rows]

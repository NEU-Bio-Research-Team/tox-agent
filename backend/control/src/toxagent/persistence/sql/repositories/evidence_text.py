"""What was read of an evidence record's full text (#43)."""
from __future__ import annotations

from sqlalchemy import and_, insert, select
from sqlalchemy.ext.asyncio import AsyncConnection

from ....domain.evidence_text import EvidenceFullText, EvidencePassage
from ...schema import evidence_full_texts, evidence_passages


class SqlEvidenceTextStore:
    def __init__(self, conn: AsyncConnection) -> None:
        self._conn = conn

    async def get(self, evidence_id: str, *, session_id: str) -> EvidenceFullText | None:
        row = (
            await self._conn.execute(
                select(evidence_full_texts).where(
                    and_(
                        evidence_full_texts.c.evidence_id == evidence_id,
                        evidence_full_texts.c.session_id == session_id,
                    )
                )
            )
        ).mappings().first()
        if row is None:
            return None
        passages = (
            await self._conn.execute(
                select(evidence_passages)
                .where(evidence_passages.c.evidence_id == evidence_id)
                .order_by(evidence_passages.c.position)
            )
        ).mappings().all()
        return EvidenceFullText(
            evidence_id=row["evidence_id"],
            session_id=row["session_id"],
            run_id=row["run_id"],
            state=row["state"],
            fetched_at=row["fetched_at"],
            full_text_id=row["full_text_id"],
            reason=row["reason"],
            passages=tuple(
                EvidencePassage(
                    locator=p["locator"], section=p["section"], kind=p["kind"],
                    text=p["text"], truncated=bool(p["truncated"]),
                    cells=tuple(p["cells"] or ()),
                )
                for p in passages
            ),
            section_titles=tuple(row["section_titles"] or ()),
            omitted=dict(row["omitted"] or {}),
            truncated=bool(row["truncated"]),
            document_sha256=row["document_sha256"],
            provenance=dict(row.get("provenance") or {}),
        )

    async def add_if_absent(self, full_text: EvidenceFullText) -> EvidenceFullText:
        """Store ``full_text`` unless the record's text was already stored, and
        return whichever the session now has. Two reads of one record in
        parallel both miss ``get``; the second must not fail on the key, and
        must not replace a text someone may already have quoted."""
        if self._conn.dialect.name == "postgresql":
            from sqlalchemy.dialects.postgresql import insert as dialect_insert
        else:
            from sqlalchemy.dialects.sqlite import insert as dialect_insert
        inserted = await self._conn.execute(
            dialect_insert(evidence_full_texts)
            .values(
                evidence_id=full_text.evidence_id,
                session_id=full_text.session_id,
                run_id=full_text.run_id,
                state=full_text.state,
                full_text_id=full_text.full_text_id,
                reason=full_text.reason,
                section_titles=list(full_text.section_titles),
                omitted=dict(full_text.omitted),
                truncated=full_text.truncated,
                document_sha256=full_text.document_sha256,
                fetched_at=full_text.fetched_at,
                provenance=dict(full_text.provenance) or None,
            )
            .on_conflict_do_nothing(index_elements=["evidence_id"])
        )
        if inserted.rowcount == 0:
            existing = await self.get(full_text.evidence_id, session_id=full_text.session_id)
            if existing is not None:
                return existing
        if full_text.passages:
            await self._conn.execute(
                insert(evidence_passages),
                [
                    {
                        "evidence_id": full_text.evidence_id,
                        "locator": passage.locator,
                        "position": position,
                        "section": passage.section,
                        "kind": passage.kind,
                        "text": passage.text,
                        "truncated": passage.truncated,
                        "cells": list(passage.cells),
                    }
                    for position, passage in enumerate(full_text.passages)
                ],
            )
        return full_text

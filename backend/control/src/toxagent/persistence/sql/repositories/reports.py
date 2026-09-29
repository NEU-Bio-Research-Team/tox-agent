"""Reports, report builds and their renderings."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping, Sequence

from sqlalchemy import and_, func, insert, select, update
from sqlalchemy.ext.asyncio import AsyncConnection

from ...schema import (
    report_artifacts,
    report_builds,
    report_claim_links,
    report_evidence_links,
    report_figures,
    report_renderings,
)
from .. import mapping as m
from ._common import _parse_ts


class SqlReportStore:
    """Report builds, artifacts, figures, renderings and their link rows.

    One store rather than six, because everything here is written inside the
    same transaction as the thing it describes: an artifact and its claim links
    committed separately could leave a report whose citations nobody can audit.

    Artifacts are stored as the canonical document they already are, and read
    back verbatim. Rebuilding a ``ReportArtifact`` dataclass graph from JSON on
    every read would be a second implementation of what the report says, and
    the two would eventually disagree; the content hash is verified on read
    instead, which is a stronger guarantee than a reconstruction that "looks
    right" (spec section 12.1, workstream R2's exit gate).
    """

    def __init__(self, conn: AsyncConnection) -> None:
        self._conn = conn

    # --- builds ------------------------------------------------------------

    async def add_build(self, build) -> None:
        await self._conn.execute(insert(report_builds).values(m.report_build_to_row(build)))

    async def get_build(self, build_id: str, *, session_id: str):
        row = (
            await self._conn.execute(
                select(report_builds).where(
                    and_(
                        report_builds.c.id == build_id,
                        report_builds.c.session_id == session_id,
                    )
                )
            )
        ).mappings().first()
        return m.row_to_report_build(row) if row else None

    async def save_build(self, build) -> None:
        """Upsert the mutable build state. The build aggregate is the one thing
        in the report path that legitimately changes: it is a state machine."""
        row = m.report_build_to_row(build)
        result = await self._conn.execute(
            update(report_builds)
            .where(report_builds.c.id == build.id)
            .values({k: v for k, v in row.items() if k != "id"})
        )
        if result.rowcount == 0:
            await self._conn.execute(insert(report_builds).values(row))

    async def list_builds_for_session(self, session_id: str, *, limit: int = 50):
        rows = (
            await self._conn.execute(
                select(report_builds)
                .where(report_builds.c.session_id == session_id)
                .order_by(report_builds.c.created_at.desc())
                .limit(limit)
            )
        ).mappings().all()
        return [m.row_to_report_build(row) for row in rows]

    # --- artifacts ---------------------------------------------------------

    async def add_artifact(
        self,
        document: dict[str, Any],
        *,
        claim_links: Sequence[Mapping[str, Any]] = (),
        evidence_links: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        await self._conn.execute(
            insert(report_artifacts).values(
                id=document["report_id"],
                report_build_id=document["report_build_id"],
                session_id=document["session_id"],
                analysis_id=document["analysis_id"],
                schema_version=document["schema_version"],
                title=document["title"],
                status=document["status"],
                report_language=document.get("report_language", "en"),
                document=document,
                content_sha256=document["content_sha256"],
                supersedes_report_id=document.get("supersedes_report_id"),
                version=document.get("version", 1),
                created_at=_parse_ts(document["created_at"]),
            )
        )
        # Deduplicated here rather than trusted from the caller: the link rows
        # are a primary key pair, and one claim cited in two sections would
        # otherwise abort the whole artifact write.
        seen: set[tuple[str, str]] = set()
        rows = []
        for link in claim_links:
            key = (document["report_id"], link["claim_id"])
            if key in seen:
                continue
            seen.add(key)
            rows.append({"report_id": document["report_id"], **dict(link)})
        if rows:
            await self._conn.execute(insert(report_claim_links).values(rows))

        seen.clear()
        rows = []
        for link in evidence_links:
            key = (document["report_id"], link["evidence_id"])
            if key in seen:
                continue
            seen.add(key)
            rows.append({"report_id": document["report_id"], **dict(link)})
        if rows:
            await self._conn.execute(insert(report_evidence_links).values(rows))

    async def get_artifact(self, report_id: str, *, session_id: str) -> dict[str, Any] | None:
        row = (
            await self._conn.execute(
                select(report_artifacts).where(
                    and_(
                        report_artifacts.c.id == report_id,
                        report_artifacts.c.session_id == session_id,
                    )
                )
            )
        ).mappings().first()
        if row is None:
            return None
        document = dict(row["document"])
        stored = row["content_sha256"]
        if document.get("content_sha256") != stored:
            # The row and the document it holds disagree about what this report
            # is. Refusing is the only safe answer: a report served from a
            # document whose hash nobody can vouch for is not auditable, which
            # is the whole reason it is stored hashed.
            raise ValueError(
                f"report {report_id} has a document hash that does not match its row; "
                "refusing to serve it"
            )
        document["renderings"] = await self.list_renderings(report_id)
        return document

    async def list_artifacts_for_session(
        self, session_id: str, *, limit: int = 50
    ) -> list[dict[str, Any]]:
        """Metadata only: a listing that returned every full document would
        grow with the size of the reports rather than their number."""
        rows = (
            await self._conn.execute(
                select(
                    report_artifacts.c.id, report_artifacts.c.report_build_id,
                    report_artifacts.c.analysis_id, report_artifacts.c.title,
                    report_artifacts.c.status, report_artifacts.c.report_language,
                    report_artifacts.c.content_sha256, report_artifacts.c.version,
                    report_artifacts.c.supersedes_report_id, report_artifacts.c.created_at,
                )
                .where(report_artifacts.c.session_id == session_id)
                .order_by(report_artifacts.c.created_at.desc())
                .limit(limit)
            )
        ).mappings().all()
        return [
            {
                "report_id": row["id"],
                "report_build_id": row["report_build_id"],
                "analysis_id": row["analysis_id"],
                "title": row["title"],
                "status": row["status"],
                "report_language": row["report_language"],
                "content_sha256": row["content_sha256"],
                "version": row["version"],
                "supersedes_report_id": row["supersedes_report_id"],
                "created_at": m.utc(row["created_at"]).isoformat(),
            }
            for row in rows
        ]

    async def get_latest_artifact_for_analysis(
        self, analysis_id: str, *, session_id: str
    ) -> dict[str, Any] | None:
        """The most recent version's full document, or ``None`` if this
        analysis has no report yet. "Latest" is ``version`` DESC, not
        ``created_at`` DESC (ADS plan section 7.3/W3-03): a rebuild's
        ``supersedes_report_id`` chain is what makes a version newer, and
        ``version`` is the field that encodes that ordering directly."""
        report_id = (
            await self._conn.execute(
                select(report_artifacts.c.id)
                .where(
                    and_(
                        report_artifacts.c.analysis_id == analysis_id,
                        report_artifacts.c.session_id == session_id,
                    )
                )
                .order_by(report_artifacts.c.version.desc())
                .limit(1)
            )
        ).scalar()
        if report_id is None:
            return None
        return await self.get_artifact(report_id, session_id=session_id)

    async def latest_version_for_analysis(self, analysis_id: str, *, session_id: str) -> int:
        row = (
            await self._conn.execute(
                select(func.max(report_artifacts.c.version)).where(
                    and_(
                        report_artifacts.c.analysis_id == analysis_id,
                        report_artifacts.c.session_id == session_id,
                    )
                )
            )
        ).scalar()
        return int(row or 0)

    # --- figures -----------------------------------------------------------

    async def add_figure(self, figure, *, session_id: str, created_at: datetime) -> None:
        await self._conn.execute(
            insert(report_figures).values(
                figure_id=figure.figure_id,
                session_id=session_id,
                attachment_id=figure.attachment_id,
                observation_id=figure.observation_id,
                endpoint=figure.endpoint,
                task=figure.task,
                media_type=figure.media_type,
                caption=figure.caption,
                alt_text=figure.alt_text,
                content_sha256=figure.content_sha256,
                renderer_version=figure.renderer_version,
                created_at=created_at,
            )
        )

    async def get_figure(self, figure_id: str, *, session_id: str) -> dict[str, Any] | None:
        row = (
            await self._conn.execute(
                select(report_figures).where(
                    and_(
                        report_figures.c.figure_id == figure_id,
                        report_figures.c.session_id == session_id,
                    )
                )
            )
        ).mappings().first()
        return dict(row) if row else None

    # --- renderings --------------------------------------------------------

    async def add_rendering(self, report_id: str, rendering) -> None:
        await self._conn.execute(
            insert(report_renderings).values(
                id=rendering.rendering_id,
                report_id=report_id,
                format=rendering.format,
                media_type=rendering.media_type,
                object_uri=rendering.object_uri,
                content_sha256=rendering.content_sha256,
                size_bytes=rendering.size_bytes,
                renderer_version=rendering.renderer_version,
                created_at=rendering.created_at,
            )
        )

    async def list_renderings(self, report_id: str) -> list[dict[str, Any]]:
        rows = (
            await self._conn.execute(
                select(report_renderings)
                .where(report_renderings.c.report_id == report_id)
                .order_by(report_renderings.c.format)
            )
        ).mappings().all()
        return [
            {
                "rendering_id": row["id"],
                "format": row["format"],
                "media_type": row["media_type"],
                "object_uri": row["object_uri"],
                "content_sha256": row["content_sha256"],
                "size_bytes": row["size_bytes"],
                "renderer_version": row["renderer_version"],
                "created_at": m.utc(row["created_at"]).isoformat(),
            }
            for row in rows
        ]

    async def get_rendering(
        self, report_id: str, fmt: str, *, session_id: str
    ) -> dict[str, Any] | None:
        row = (
            await self._conn.execute(
                select(report_renderings)
                .select_from(
                    report_renderings.join(
                        report_artifacts,
                        report_renderings.c.report_id == report_artifacts.c.id,
                    )
                )
                .where(
                    and_(
                        report_renderings.c.report_id == report_id,
                        report_renderings.c.format == fmt,
                        # Scoped through the artifact, so a rendering id from
                        # another session resolves to nothing rather than to
                        # someone else's file.
                        report_artifacts.c.session_id == session_id,
                    )
                )
            )
        ).mappings().first()
        if row is None:
            return None
        return {
            "rendering_id": row["id"],
            "report_id": row["report_id"],
            "format": row["format"],
            "media_type": row["media_type"],
            "object_uri": row["object_uri"],
            "content_sha256": row["content_sha256"],
            "size_bytes": row["size_bytes"],
            "renderer_version": row["renderer_version"],
        }

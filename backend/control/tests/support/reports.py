"""Seed the rows a report artifact references.

`report_artifacts` and `report_builds` carry real foreign keys to sessions,
runs, analyses and builds. SQLite leaves them unenforced by default, so a test
could insert a report whose build never existed and pass; the PostgreSQL lane
enforces them. Seeding through here keeps a test honest on both.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from toxagent.domain.report import ReportBuild, ReportBuildRequest


async def seed_report_build(
    db, *, session_id: str, run_id: str, analysis_id: str, now: datetime,
    selected_endpoints: tuple[str, ...] = ("herg",),
) -> ReportBuild:
    """Persist a queued build for an existing session, run and analysis."""
    request = ReportBuildRequest(
        session_id=session_id, analysis_id=analysis_id,
        selected_endpoints=selected_endpoints,
        include_explanations=False, include_external_evidence=False,
    )
    build = ReportBuild.start(
        session_id=session_id, run_id=run_id, request=request,
        now=now, deadline_at=now + timedelta(minutes=15),
    )
    async with db.unit_of_work() as uow:
        await uow.reports.add_build(build)
        await uow.commit()
    return build

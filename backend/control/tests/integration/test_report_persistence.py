from datetime import datetime, timedelta, timezone

import pytest

from toxagent.domain.ids import new_id
from toxagent.domain.report import BuildStage, ReportBuild, ReportBuildRequest


pytestmark = pytest.mark.anyio
NOW = datetime(2026, 9, 9, 5, 0, tzinfo=timezone.utc)


async def test_report_build_round_trips_with_request_and_stage_state(db):
    request = ReportBuildRequest(
        session_id=new_id("ses"), analysis_id=new_id("ana"),
        selected_endpoints=("herg", "tox21"),
        selected_tox21_tasks=("NR-AR",),
        output_formats=("markdown", "html", "pdf"),
    )
    build = ReportBuild.start(
        session_id=request.session_id, run_id=new_id("run"), request=request,
        now=NOW, deadline_at=NOW + timedelta(minutes=15),
    )
    # The report tables deliberately reference product entities. This test
    # exercises the mapping with FK checks disabled by SQLite's default; the
    # migrated PostgreSQL lane supplies the cross-table FK gate.
    async with db.unit_of_work() as uow:
        await uow.reports.add_build(build)
        await uow.commit()

    loaded = None
    async with db.unit_of_work() as uow:
        loaded = await uow.reports.get_build(build.id, session_id=build.session_id)

    assert loaded == build
    advanced = loaded.advance(BuildStage.PREPARING_ANALYSIS, now=NOW)
    async with db.unit_of_work() as uow:
        await uow.reports.save_build(advanced)
        await uow.commit()
    async with db.unit_of_work() as uow:
        assert await uow.reports.get_build(build.id, session_id=build.session_id) == advanced

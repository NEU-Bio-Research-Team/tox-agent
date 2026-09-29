from datetime import datetime, timedelta, timezone

import pytest

from toxagent.application.policy import Actor
from toxagent.application.prediction.create_analysis import CreateAnalysis
from toxagent.platform.config import PolicySettings
from toxagent.domain.message import Message, Role
from toxagent.domain.report import BuildStage, ReportBuild, ReportBuildRequest
from toxagent.domain.run import Intent, Lane, Run
from toxagent.domain.session import Session
from tests.support.predictor import StubPredictor


pytestmark = pytest.mark.anyio
NOW = datetime(2026, 9, 9, 5, 0, tzinfo=timezone.utc)
ACTOR = Actor(subject_id="user-1")


async def test_report_build_round_trips_with_request_and_stage_state(db):
    session = Session.create(ACTOR.subject_id, now=NOW)
    message = Message.create(session.id, Role.USER, 1, now=NOW)
    run = Run.create(session.id, message.id, Lane.MIXED, Intent.BUILD_REPORT, now=NOW)
    async with db.unit_of_work() as uow:
        await uow.sessions.add(session)
        await uow.messages.add(message)
        await uow.runs.add(run)
        await uow.commit()
    analysis = await CreateAnalysis(db, StubPredictor().client(), PolicySettings()).execute(
        actor=ACTOR, session_id=session.id, run_id=run.id, smiles="CCO",
        endpoints=("herg",), owns_run=False,
    )
    request = ReportBuildRequest(
        session_id=session.id, analysis_id=analysis.snapshot.id,
        selected_endpoints=("herg", "tox21"),
        selected_tox21_tasks=("NR-AR",),
        output_formats=("markdown", "html", "pdf"),
    )
    build = ReportBuild.start(
        session_id=request.session_id, run_id=run.id, request=request,
        now=NOW, deadline_at=NOW + timedelta(minutes=15),
    )
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

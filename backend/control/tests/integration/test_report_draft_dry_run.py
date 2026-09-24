"""`check_report_draft`: ask the validator without spending the answer.

A live build (run_947bfb9b, 2026-09-09) assembled eleven correct sections and
forgot one *derived* limitation code. That single omission came back as its only
violation — and by then the run had spent its forty tool calls gathering, so
every read it attempted while fixing the draft was budget-denied and it never
submitted again. A complete report was lost to a one-line bookkeeping slip.

The correction attempt exists to bound how many times a model may *disagree*
with the validator. It was also, accidentally, bounding how many times a model
could *ask* — because submitting was the only way to run a check that is
deterministic, provider-free, and reads only rows the run already produced.

These tests pin the two properties that make the dry run trustworthy: it judges
by exactly the same rules as the real submission, and it changes nothing.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import text

from toxagent.application.policy import Actor
from toxagent.application.create_analysis import CreateAnalysis
from toxagent.application.submit_report_draft import (
    ReportValidationFailed,
    SubmitReportDraft,
)
from toxagent.config import PolicySettings
from toxagent.domain.errors import Conflict
from toxagent.domain.ids import new_id
from toxagent.domain.message import Message, Role
from toxagent.domain.report import (
    REQUIRED_SECTION_IDS,
    BuildStage,
    ReportBuild,
    ReportBuildRequest,
)
from toxagent.domain.run import Intent, Lane, Run
from toxagent.domain.session import Session
from toxagent.validation.report_wire import ReportDraftCandidate
from tests.support.predictor import StubPredictor

pytestmark = pytest.mark.anyio

NOW = datetime(2026, 9, 9, tzinfo=timezone.utc)
ACTOR = Actor(subject_id="user-1")


async def _seed(db):
    """A session with a real analysis and a report build in `synthesizing`."""
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
        selected_endpoints=("herg",),
        include_explanations=False, include_external_evidence=False,
    )
    build = ReportBuild.start(
        session_id=session.id, run_id=run.id, request=request, now=NOW,
        deadline_at=NOW + timedelta(minutes=15),
    )
    for stage in (
        BuildStage.PREPARING_ANALYSIS, BuildStage.ASSEMBLING_SUBSTANCE,
        BuildStage.ASSEMBLING_PREDICTIONS, BuildStage.SYNTHESIZING,
    ):
        build = build.advance(stage, now=NOW)
    async with db.unit_of_work() as uow:
        await uow.reports.add_build(build)
        await uow.commit()
    return session.id, run.id, build


def _draft(build_id: str, *, sections=None) -> ReportDraftCandidate:
    """A structurally valid draft that will fail on content, not on shape."""
    return ReportDraftCandidate.model_validate({
        "report_build_id": build_id,
        "title": "Toxicity Screening Report",
        "sections": sections or [
            {
                "section_id": section_id,
                "heading": section_id.replace("_", " ").title(),
                "body_markdown": "Recorded content.",
                "source_classes": ["agent_synthesis"],
            }
            for section_id in REQUIRED_SECTION_IDS
        ],
    })


async def _build_stage(db, session_id, build_id):
    async with db.unit_of_work() as uow:
        return await uow.reports.get_build(build_id, session_id=session_id)


# --- it changes nothing ------------------------------------------------------

async def test_a_failing_check_consumes_no_correction_attempt(db):
    """The property the whole tool rests on. If asking cost an attempt, a model
    would be right not to ask."""
    session_id, run_id, build = await _seed(db)
    service = SubmitReportDraft(db)

    for _ in range(5):
        violations = await service.check(
            session_id=session_id, run_id=run_id,
            owner_id=ACTOR.subject_id, draft=_draft(build.id),
        )
        assert violations, "this draft is incomplete and must not pass"

    after = await _build_stage(db, session_id, build.id)
    assert after.correction_attempts == build.correction_attempts == 0
    assert after.stage is BuildStage.SYNTHESIZING
    assert after.report_id is None
    assert after.failure_code is None


async def _row_counts(db) -> dict[str, int]:
    tables = ("event_outbox", "report_artifacts", "report_renderings", "report_claim_links")
    async with db.engine.connect() as connection:
        return {
            table: await connection.scalar(text(f"SELECT count(*) FROM {table}"))
            for table in tables
        }


async def test_a_check_writes_no_row_at_all(db):
    session_id, run_id, build = await _seed(db)
    before = await _row_counts(db)

    await SubmitReportDraft(db).check(
        session_id=session_id, run_id=run_id,
        owner_id=ACTOR.subject_id, draft=_draft(build.id),
    )

    # No event either: a rejected *submission* is an auditable decision and
    # emits one, but a question nobody acted on is not a decision.
    assert await _row_counts(db) == before


async def test_a_real_submission_still_consumes_the_attempt(db):
    """The dry run must not have made the real gate cheaper by accident."""
    session_id, run_id, build = await _seed(db)
    service = SubmitReportDraft(db)

    await service.check(
        session_id=session_id, run_id=run_id,
        owner_id=ACTOR.subject_id, draft=_draft(build.id),
    )
    with pytest.raises(ReportValidationFailed):
        await service.execute(
            session_id=session_id, run_id=run_id,
            owner_id=ACTOR.subject_id, draft=_draft(build.id),
        )

    after = await _build_stage(db, session_id, build.id)
    assert after.correction_attempts == 1


# --- it judges by the same rules ---------------------------------------------

async def test_the_check_reports_exactly_what_the_submission_reports(db):
    """Drift here would be worse than no dry run: a pass the real submission
    then refuses is a trap rather than a tool."""
    session_id, run_id, build = await _seed(db)
    service = SubmitReportDraft(db)
    draft = _draft(build.id)

    checked = await service.check(
        session_id=session_id, run_id=run_id, owner_id=ACTOR.subject_id, draft=draft,
    )
    with pytest.raises(ReportValidationFailed) as raised:
        await service.execute(
            session_id=session_id, run_id=run_id, owner_id=ACTOR.subject_id, draft=draft,
        )

    assert [(v.code, v.path) for v in checked] == [
        (v.code, v.path) for v in raised.value.violations
    ]


async def test_the_check_refuses_a_build_from_another_session(db):
    """Same scoping as the submission: a foreign build id resolves to nothing
    rather than to somebody else's report."""
    session_id, run_id, build = await _seed(db)
    other_session_id, _, _ = await _seed(db)

    with pytest.raises(Conflict):
        await SubmitReportDraft(db).check(
            session_id=other_session_id, run_id=run_id,
            owner_id=ACTOR.subject_id, draft=_draft(build.id),
        )


async def test_the_check_refuses_a_build_that_already_produced_a_report(db):
    session_id, run_id, build = await _seed(db)
    async with db.unit_of_work() as uow:
        await uow.reports.save_build(
            build.advance(BuildStage.VALIDATING, now=NOW)
            .advance(BuildStage.RENDERING, now=NOW)
            .advance(BuildStage.COMPLETED, now=NOW, report_id=new_id("rpt"))
        )
        await uow.commit()

    with pytest.raises(Conflict):
        await SubmitReportDraft(db).check(
            session_id=session_id, run_id=run_id,
            owner_id=ACTOR.subject_id, draft=_draft(build.id),
        )


# --- the tool surface --------------------------------------------------------

async def test_the_tool_reports_zero_attempts_consumed_and_stores_nothing(db):
    """The model has to believe the tool is free, or it will not use it."""
    from toxagent.tools.definitions import report as report_tools
    from toxagent.tools.registry import ToolContext

    session_id, run_id, build = await _seed(db)
    definitions = {d.name: d for d in report_tools.build(db)}
    assert "check_report_draft" in definitions

    context = ToolContext(
        session_id=session_id, run_id=run_id, actor=ACTOR, profile="report_build",
        deadline_at=NOW + timedelta(minutes=10), intent=Intent.BUILD_REPORT.value,
    )
    output = await definitions["check_report_draft"].handler(context, _draft(build.id))

    assert output.model_view["ok"] is False
    assert output.model_view["violation_count"] > 0
    assert output.model_view["correction_attempts_consumed"] == 0
    assert "Nothing was stored" in output.model_view["note"]
    assert output.provenance["dry_run"] is True

    after = await _build_stage(db, session_id, build.id)
    assert after.correction_attempts == 0
    assert after.stage is BuildStage.SYNTHESIZING


# --- durable working draft -------------------------------------------------

async def test_a_saved_draft_is_repaired_by_small_versioned_patches(db):
    session_id, run_id, build = await _seed(db)
    service = SubmitReportDraft(db)

    saved = await service.save_checkpoint(
        session_id=session_id, run_id=run_id, owner_id=ACTOR.subject_id,
        draft=_draft(build.id),
    )
    assert saved.version == 1
    assert saved.violations

    patched = await service.patch_checkpoint(
        session_id=session_id, run_id=run_id, owner_id=ACTOR.subject_id,
        report_build_id=build.id, expected_version=saved.version,
        operations=[{"op": "replace", "path": "/title", "value": "Repaired report"}],
    )
    assert patched.version == 2
    assert patched.draft.title == "Repaired report"
    assert patched.content_sha256 != saved.content_sha256

    reread = await service.check_checkpoint(
        session_id=session_id, run_id=run_id, owner_id=ACTOR.subject_id,
        report_build_id=build.id,
    )
    assert reread.version == 2
    assert reread.draft.title == "Repaired report"

    after = await _build_stage(db, session_id, build.id)
    assert after.correction_attempts == 0
    assert after.stage is BuildStage.SYNTHESIZING


async def test_a_stale_patch_cannot_overwrite_a_newer_saved_draft(db):
    session_id, run_id, build = await _seed(db)
    service = SubmitReportDraft(db)
    saved = await service.save_checkpoint(
        session_id=session_id, run_id=run_id, owner_id=ACTOR.subject_id,
        draft=_draft(build.id),
    )
    await service.patch_checkpoint(
        session_id=session_id, run_id=run_id, owner_id=ACTOR.subject_id,
        report_build_id=build.id, expected_version=saved.version,
        operations=[{"op": "replace", "path": "/title", "value": "Version two"}],
    )

    with pytest.raises(Conflict, match="changed"):
        await service.patch_checkpoint(
            session_id=session_id, run_id=run_id, owner_id=ACTOR.subject_id,
            report_build_id=build.id, expected_version=saved.version,
            operations=[{"op": "replace", "path": "/title", "value": "Stale write"}],
        )


async def test_saved_submit_revalidates_the_checkpoint(db):
    session_id, run_id, build = await _seed(db)
    service = SubmitReportDraft(db)
    saved = await service.save_checkpoint(
        session_id=session_id, run_id=run_id, owner_id=ACTOR.subject_id,
        draft=_draft(build.id),
    )

    with pytest.raises(ReportValidationFailed):
        await service.execute_checkpoint(
            session_id=session_id, run_id=run_id, owner_id=ACTOR.subject_id,
            report_build_id=build.id, expected_version=saved.version,
        )

    after = await _build_stage(db, session_id, build.id)
    assert after.correction_attempts == 1

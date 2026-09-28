"""get_artifact_inventory and get_report_summary (ADS plan section 8.1, W2-03/04).

Exercises both tools through the real registry/runner/database, the same
pattern as ``tests/integration/test_evidence_tools.py``.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from toxagent.application.prediction.create_analysis import CreateAnalysis
from toxagent.application.policy import Actor
from toxagent.platform.config import PolicySettings
from toxagent.domain.events import EventType
from toxagent.domain.message import Message, Role
from toxagent.domain.ids import CLAIM, GAP, REPORT_BUILD, new_id
from toxagent.domain.report import (
    REQUIRED_SECTION_IDS,
    BuildStage,
    GapReason,
    ReportArtifact,
    ReportGap,
    ReportRecommendation,
    ReportSection,
    SubstanceProfile,
)
from toxagent.domain.run import Intent, Lane, Run
from toxagent.domain.session import Session
from toxagent.tools.bootstrap import build_registry
from toxagent.tools.registry import ToolContext
from toxagent.tools.runner import ToolRunner
from tests.support.predictor import ASPIRIN, StubPredictor

pytestmark = pytest.mark.anyio

NOW = datetime(2026, 9, 16, tzinfo=timezone.utc)
ACTOR = Actor(subject_id="user-1")


async def scenario(db, *, with_report=False):
    actor = ACTOR
    session = Session.create(actor.subject_id, now=NOW)
    message = Message.create(session.id, Role.USER, 1, now=NOW)
    run = Run.create(session.id, message.id, Lane.AGENTIC, Intent.DECISION_SUPPORT, now=NOW)
    async with db.unit_of_work() as uow:
        await uow.sessions.add(session)
        await uow.messages.add(message)
        await uow.runs.add(run)
        uow.emit(
            session_id=session.id, type=EventType.SESSION_CREATED,
            entity_type="session", entity_id=session.id,
        )
        await uow.commit()
    predictor = StubPredictor().client()
    analysis_service = CreateAnalysis(db, predictor, PolicySettings())
    result = await analysis_service.execute(
        actor=actor, session_id=session.id, run_id=run.id, smiles=ASPIRIN, owns_run=False,
    )
    analysis_id = result.snapshot.id

    report_id = None
    if with_report:
        report = ReportArtifact.create(
            report_build_id=new_id(REPORT_BUILD),
            session_id=session.id,
            analysis_id=analysis_id,
            title="Aspirin screening report",
            subject=SubstanceProfile(canonical_smiles=ASPIRIN),
            sections=[
                ReportSection(section_id=sid, heading=sid, body_markdown="n/a")
                for sid in REQUIRED_SECTION_IDS
            ],
            gaps=(
                ReportGap(
                    gap_id=new_id(GAP),
                    reason=GapReason.NO_RELEVANT_EVIDENCE,
                    detail="no hERG literature found",
                    section_id="evidence",
                ),
            ),
            recommendations=(
                ReportRecommendation(
                    recommendation_id="rec_placeholder",
                    text="Confirm hERG signal with an orthogonal assay",
                    basis_claim_ids=(new_id(CLAIM),),
                    action_category="verification",
                    priority="medium",
                    rationale="predictor score alone is not confirmatory",
                ),
            ),
            now=NOW,
        )
        async with db.unit_of_work() as uow:
            await uow.reports.add_artifact(_to_document(report))
            await uow.commit()
        report_id = report.id

    registry = build_registry(db, predictor, analysis_service, PolicySettings())
    runner = ToolRunner(registry, db, max_calls_per_run=20)
    context = ToolContext(
        session_id=session.id, run_id=run.id, actor=actor, profile="decision_support",
        deadline_at=datetime.now(timezone.utc) + timedelta(seconds=60),
    )
    return runner, context, analysis_id, report_id


def _to_document(report: ReportArtifact) -> dict:
    document = {
        "report_id": report.id,
        "report_build_id": report.report_build_id,
        "session_id": report.session_id,
        "analysis_id": report.analysis_id,
        "schema_version": report.schema_version,
        "title": report.title,
        "status": report.status.value,
        "report_language": report.report_language,
        "content_sha256": report.content_sha256,
        "supersedes_report_id": report.supersedes_report_id,
        "version": report.version,
        "created_at": report.created_at.isoformat(),
        "gaps": [g.to_dict() for g in report.gaps],
        "recommendations": [r.to_dict() for r in report.recommendations],
        "evidence_synthesis": [],
    }
    return document


async def test_inventory_reports_the_active_analysis_and_no_report_yet(db):
    runner, context, analysis_id, _ = await scenario(db)
    result = await runner.call(context, "get_artifact_inventory", {})
    assert result["status"] == "completed"
    view = result["model_view"]
    assert view["schema_version"] == "artifact-inventory-v1"
    assert view["analysis"]["analysis_id"] == analysis_id
    assert view["latest_report"] is None
    assert view["available_tools"]["evidence_search"] is False


async def test_inventory_surfaces_the_latest_reports_gaps_and_recommendations(db):
    runner, context, analysis_id, report_id = await scenario(db, with_report=True)
    result = await runner.call(context, "get_artifact_inventory", {})
    view = result["model_view"]
    assert view["latest_report"]["report_id"] == report_id
    assert view["latest_report"]["gap_summaries"] == ["no hERG literature found"]
    assert view["latest_report"]["recommendation_summaries"] == [
        "Confirm hERG signal with an orthogonal assay"
    ]


async def test_report_summary_defaults_to_latest_for_the_active_analysis(db):
    runner, context, _, report_id = await scenario(db, with_report=True)
    result = await runner.call(context, "get_report_summary", {})
    view = result["model_view"]
    assert view["report_id"] == report_id
    assert view["status"] == BuildStage.COMPLETED_WITH_GAPS.value
    assert view["gaps"][0]["reason"] == GapReason.NO_RELEVANT_EVIDENCE.value
    assert view["recommendations"][0]["action_category"] == "verification"


async def test_report_summary_with_no_report_is_a_typed_error(db):
    runner, context, _, _ = await scenario(db, with_report=False)
    result = await runner.call(context, "get_report_summary", {})
    assert result["status"] == "error"

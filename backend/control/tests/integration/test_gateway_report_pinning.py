"""A decision_support turn's prompt pins the latest report for the active
analysis (ADS plan section 8.1, W3-03) — the gap the code itself used to
document as deferred (see the W2-04/get_report_summary comment this
supersedes in harness/gateway.py).
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from toxagent.application.prediction.create_analysis import CreateAnalysis
from toxagent.application.policy import Actor
from toxagent.application.runs.scheduler import RunContext
from toxagent.platform.config import PolicySettings, RuntimeSettings
from toxagent.domain.events import EventType
from toxagent.domain.ids import CLAIM, GAP, REPORT_BUILD, new_id
from toxagent.domain.message import Message, Role
from toxagent.domain.report import (
    REQUIRED_SECTION_IDS,
    GapReason,
    ReportArtifact,
    ReportGap,
    ReportRecommendation,
    ReportSection,
    SubstanceProfile,
)
from toxagent.domain.run import Intent, Lane, Run
from toxagent.domain.session import Session
from toxagent.harness.gateway import AgentRuntimeGateway
from toxagent.tools.registry import ToolRegistry
from tests.support.predictor import ASPIRIN, StubPredictor

pytestmark = pytest.mark.anyio

# Relative, not a calendar date: _prepare_context refuses a run whose deadline
# (created_at + run_deadline_s) has already passed, so a fixed date rots.
NOW = datetime.now(timezone.utc)


@pytest.fixture(autouse=True)
def _fresh_now():
    """Re-read the clock per test. Read once at collection, NOW went stale
    whenever the tests before these took longer than run_deadline_s, and the
    run's deadline had passed before the test began."""
    global NOW
    NOW = datetime.now(timezone.utc)
ACTOR = Actor(subject_id="user-1")


def _to_document(report: ReportArtifact) -> dict:
    return {
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


async def test_decision_support_turn_pins_the_latest_report(db):
    session = Session.create(ACTOR.subject_id, now=NOW)
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
        actor=ACTOR, session_id=session.id, run_id=run.id, smiles=ASPIRIN, owns_run=False,
    )
    analysis_id = result.snapshot.id

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

    gateway = AgentRuntimeGateway(
        db, registry=ToolRegistry(), capability_tokens=None, provider=None, settings=RuntimeSettings(),
    )
    context = RunContext(
        actor=ACTOR, session_id=session.id, run_id=run.id, intent=Intent.DECISION_SUPPORT,
        analysis_id=analysis_id,
    )
    system_prompt, _profile, _deadline, _hash = await gateway._prepare_context(context)

    assert f"- report {report.id}:" in system_prompt
    assert "1 open gap(s), 1 recommendation(s)" in system_prompt
    assert "get_report_summary" in system_prompt


async def test_decision_support_turn_with_no_report_yet_pins_nothing_report_shaped(db):
    session = Session.create(ACTOR.subject_id, now=NOW)
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
        actor=ACTOR, session_id=session.id, run_id=run.id, smiles=ASPIRIN, owns_run=False,
    )

    gateway = AgentRuntimeGateway(
        db, registry=ToolRegistry(), capability_tokens=None, provider=None, settings=RuntimeSettings(),
    )
    context = RunContext(
        actor=ACTOR, session_id=session.id, run_id=run.id, intent=Intent.DECISION_SUPPORT,
        analysis_id=result.snapshot.id,
    )
    system_prompt, _profile, _deadline, _hash = await gateway._prepare_context(context)

    assert "- report " not in system_prompt

"""W5-06: an accepted decision-support answer's proposed evidence_relations
are actually persisted — the check that would have caught
domain/evidence_relation.py's table being wired up but never called from
anywhere (confirmed orphaned before this PR: zero call sites in application/,
tools/, harness/).
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from toxagent.application.prediction.create_analysis import CreateAnalysis
from toxagent.application.policy import Actor
from toxagent.application.conversation.submit_answer import SubmitAnswer
from toxagent.platform.config import PolicySettings
from toxagent.domain.errors import AnswerValidationFailed
from toxagent.domain.events import EventType
from toxagent.domain.message import Message, Role
from toxagent.domain.run import Intent, Lane, Run
from toxagent.domain.session import Session
from toxagent.validation.answer.draft_wire import (
    ClaimCandidateV2,
    EvidenceRelationInputV2,
    GroundedAnswerDraftV2,
)
from toxagent.validation.answer.candidate_wire import LimitationCandidate
from tests.support.predictor import ASPIRIN, StubPredictor

pytestmark = pytest.mark.anyio

NOW = datetime(2026, 9, 16, tzinfo=timezone.utc)
ACTOR = Actor(subject_id="user-1")


async def rig(db):
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
    return session, run, result.observation


def _draft(observation, **overrides) -> GroundedAnswerDraftV2:
    payload = dict(
        answer_markdown="Predicted hERG blocker probability is 0.731.",
        claims=[
            ClaimCandidateV2(
                local_ref="herg_p", kind="numeric",
                text="Predicted hERG blocker probability is 0.731.",
                observation_id=observation.id,
                field_path="predictions.herg.probability_blocker",
                transform="round:3",
            )
        ],
        limitations=[LimitationCandidate(code="uncalibrated_probability", text="")],
    )
    payload.update(overrides)
    return GroundedAnswerDraftV2(**payload)


async def test_an_accepted_answers_evidence_relations_are_persisted_not_orphaned(db):
    session, run, observation = await rig(db)
    draft = _draft(
        observation,
        evidence_relations=[
            EvidenceRelationInputV2(
                proposition="does the hERG signal support deprioritizing this candidate",
                source_class="predictor_fact",
                source_id=observation.id,
                relation="supports",
                reason_codes=["endpoint_match"],
                endpoint="herg",
            )
        ],
    )
    outcome = await SubmitAnswer(db, PolicySettings()).execute(
        session_id=session.id, run_id=run.id, candidate=draft, language="en",
    )
    assert outcome.is_fallback is False

    async with db.unit_of_work() as uow:
        persisted = await uow.evidence_relations.list_for_run(run.id)
    assert len(persisted) == 1
    assert persisted[0].source_ref.source_id == observation.id
    assert persisted[0].relation.value == "supports"
    assert persisted[0].scope.endpoint == "herg"


async def test_an_evidence_relation_with_an_unresolvable_source_is_rejected(db):
    session, run, observation = await rig(db)
    draft = _draft(
        observation,
        evidence_relations=[
            EvidenceRelationInputV2(
                proposition="does this evidence apply",
                source_class="external_experimental",
                source_id="evd_" + "0" * 32,
                relation="supports",
                reason_codes=["endpoint_match"],
            )
        ],
    )
    with pytest.raises(AnswerValidationFailed) as exc_info:
        await SubmitAnswer(db, PolicySettings(max_answer_candidates_per_run=2)).execute(
            session_id=session.id, run_id=run.id, candidate=draft, language="en",
        )
    codes = {
        v["code"] if isinstance(v, dict) else v.code
        for v in exc_info.value.detail["violations"]
    }
    assert "evidence_relation_source_not_found" in codes

    async with db.unit_of_work() as uow:
        persisted = await uow.evidence_relations.list_for_run(run.id)
    assert persisted == []

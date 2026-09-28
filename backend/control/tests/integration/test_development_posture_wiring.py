"""W6: a decision-support answer's proposed development_posture is validated
for safety/posture conflation and persisted — not left as an unreferenced
domain dataclass with no tool, no migration and no wording gate (confirmed
that way before this PR: zero call sites in application/, tools/, harness/,
and no posture-aware check in validation/prohibited_claims.py).
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from datetime import timedelta

from toxagent.application.prediction.create_analysis import CreateAnalysis
from toxagent.application.policy import Actor
from toxagent.application.conversation.submit_answer import SubmitAnswer
from toxagent.config import PolicySettings
from toxagent.domain.errors import AnswerValidationFailed
from toxagent.domain.events import EventType
from toxagent.domain.message import Message, Role
from toxagent.domain.run import Intent, Lane, Run
from toxagent.domain.session import Session
from toxagent.tools.bootstrap import build_registry
from toxagent.tools.registry import ToolContext
from toxagent.tools.runner import ToolRunner
from toxagent.validation.answer.candidate_wire import LimitationCandidate
from toxagent.validation.answer.draft_wire import (
    ClaimCandidateV2,
    DevelopmentPostureInputV2,
    GroundedAnswerDraftV2,
)
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


def _draft(observation, *, posture: DevelopmentPostureInputV2) -> GroundedAnswerDraftV2:
    return GroundedAnswerDraftV2(
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
        development_posture=posture,
    )


async def test_a_valid_posture_is_persisted_with_the_answer(db):
    session, run, observation = await rig(db)
    draft = _draft(
        observation,
        posture=DevelopmentPostureInputV2(
            value="hold",
            scope="drug_candidate",
            confidence_band="moderate",
            basis_local_refs=["herg_p"],
            rationale="hERG signal alone is not enough; confirm with an orthogonal assay.",
            recommended_next_steps=["Confirm hERG signal with an orthogonal assay"],
        ),
    )
    outcome = await SubmitAnswer(db, PolicySettings()).execute(
        session_id=session.id, run_id=run.id, candidate=draft, language="en",
    )
    assert outcome.is_fallback is False

    async with db.unit_of_work() as uow:
        persisted = await uow.development_postures.get_for_answer(outcome.answer.id)
    assert persisted is not None
    assert persisted.value.value == "hold"
    assert len(persisted.basis_claim_ids) == 1
    assert persisted.basis_claim_ids[0] != "herg_p"  # a real server-issued claim id, not the local ref


async def test_a_posture_that_conflates_safety_with_proceeding_is_rejected(db):
    session, run, observation = await rig(db)
    draft = _draft(
        observation,
        posture=DevelopmentPostureInputV2(
            value="proceed",
            scope="drug_candidate",
            confidence_band="moderate",
            basis_local_refs=["herg_p"],
            rationale="This compound appears safe to proceed with further development.",
        ),
    )
    with pytest.raises(AnswerValidationFailed) as exc_info:
        await SubmitAnswer(db, PolicySettings(max_answer_candidates_per_run=2)).execute(
            session_id=session.id, run_id=run.id, candidate=draft, language="en",
        )
    codes = {
        v["code"] if isinstance(v, dict) else v.code
        for v in exc_info.value.detail["violations"]
    }
    assert "safety_posture_conflation" in codes

    async with db.unit_of_work() as uow:
        assert await uow.development_postures.list_for_run(run.id) == []


async def test_the_submit_tool_renders_posture_as_its_own_labeled_block(db, monkeypatch):
    monkeypatch.setenv("TOXAGENT_FLAG_ANSWER_DRAFT_V2", "1")
    session, run, observation = await rig(db)
    predictor = StubPredictor().client()
    analysis_service = CreateAnalysis(db, predictor, PolicySettings())
    registry = build_registry(db, predictor, analysis_service, PolicySettings())
    runner = ToolRunner(registry, db)
    context = ToolContext(
        session_id=session.id, run_id=run.id, actor=ACTOR, profile="decision_support",
        deadline_at=datetime.now(timezone.utc) + timedelta(seconds=60),
    )
    draft = _draft(
        observation,
        posture=DevelopmentPostureInputV2(
            value="hold",
            scope="drug_candidate",
            confidence_band="moderate",
            basis_local_refs=["herg_p"],
            rationale="hERG signal alone is not enough; confirm with an orthogonal assay.",
            recommended_next_steps=["Confirm hERG signal with an orthogonal assay"],
        ),
    )
    result = await runner.call(context, "submit_grounded_answer", draft.model_dump())
    assert result["status"] == "completed", result
    assert result["model_view"]["development_posture"]["value"] == "hold"
    assert result["ui_view"]["development_posture"]["value"] == "hold"
    assert "development_posture" not in result["ui_view"].get("answer_markdown", "")


async def test_an_insufficient_posture_with_no_next_steps_is_rejected_not_silently_dropped(db):
    """DevelopmentPosture.__post_init__ requires recommended_next_steps for
    'insufficient' — this must surface as a correctable violation, not vanish
    the posture while still accepting the answer underneath it."""
    session, run, observation = await rig(db)
    draft = _draft(
        observation,
        posture=DevelopmentPostureInputV2(
            value="insufficient",
            scope="drug_candidate",
            confidence_band="not_assessed",
            rationale="Not enough evidence to choose a posture yet.",
        ),
    )
    with pytest.raises(AnswerValidationFailed) as exc_info:
        await SubmitAnswer(db, PolicySettings(max_answer_candidates_per_run=2)).execute(
            session_id=session.id, run_id=run.id, candidate=draft, language="en",
        )
    codes = {
        v["code"] if isinstance(v, dict) else v.code
        for v in exc_info.value.detail["violations"]
    }
    assert "development_posture_invalid" in codes


async def test_a_committing_posture_cannot_rest_on_synthesis_alone(db):
    """P1-03 (TAB-Suite Wave 2): agent_synthesis is a transformation of real
    sources, never a source. A moderate/strong proceed or deprioritize whose
    only bearing relations are syntheses is refused; the same synthesis with
    resolvable lineage beside a direct source is accepted, lineage persisted."""
    from toxagent.validation.answer.draft_wire import EvidenceRelationInputV2

    session, run, observation = await rig(db)
    synthesis = EvidenceRelationInputV2(
        proposition="hERG liability is manageable", source_class="agent_synthesis",
        source_id=observation.id, relation="supports", reason_codes=["synthesis"],
        input_source_refs=[f"observation:{observation.id}"],
    )
    posture = DevelopmentPostureInputV2(
        value="proceed", scope="drug_candidate", confidence_band="moderate",
        basis_local_refs=["herg_p"],
        rationale="The predicted hERG signal is moderate and can be monitored.",
    )
    draft = _draft(observation, posture=posture).model_copy(update={"evidence_relations": [synthesis]})
    with pytest.raises(AnswerValidationFailed) as exc_info:
        await SubmitAnswer(db, PolicySettings(max_answer_candidates_per_run=2)).execute(
            session_id=session.id, run_id=run.id, candidate=draft, language="en",
        )
    codes = {
        v["code"] if isinstance(v, dict) else v.code
        for v in exc_info.value.detail["violations"]
    }
    assert "posture_rests_on_synthesis_only" in codes

    direct = EvidenceRelationInputV2(
        proposition="hERG liability is manageable", source_class="predictor_fact",
        source_id=observation.id, relation="supports", reason_codes=["predicted_probability"],
    )
    accepted = draft.model_copy(update={"evidence_relations": [direct, synthesis]})
    outcome = await SubmitAnswer(db, PolicySettings(max_answer_candidates_per_run=2)).execute(
        session_id=session.id, run_id=run.id, candidate=accepted, language="en",
    )
    assert outcome.is_fallback is False
    async with db.unit_of_work() as uow:
        relations = await uow.evidence_relations.list_for_run(run.id)
    lineage = {r.source_ref.source_class.value: r.input_refs for r in relations}
    assert lineage["agent_synthesis"] == (f"observation:{observation.id}",)
    assert lineage["predictor_fact"] == ()

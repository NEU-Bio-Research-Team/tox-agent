"""The contract, the route and the case view reach the turn (#69/#68/#63).

The unit tests prove the three build correctly. This proves the turn actually
reads them: with ``investigator_contract_v1`` on, a decision-support run's
prompt carries the goal contract and the six-group view instead of the case
checkpoint, the route decision is taken from the run's own bound artifacts,
and the contract is recorded on the decision state where a grader reads it.

Off, the prompt is what it was. That is the flag's whole promise, and it is
what lets this ship before the pilot that decides whether it stays.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from toxagent.application.investigation import router
from toxagent.application.policy import Actor
from toxagent.application.prediction.create_analysis import CreateAnalysis
from toxagent.application.runs.scheduler import RunContext
from toxagent.domain.events import EventType
from toxagent.domain.message import Message, Role
from toxagent.domain.run import Intent, Lane, Run
from toxagent.domain.session import Session
from toxagent.harness.gateway import AgentRuntimeGateway
from toxagent.platform.config import PolicySettings, RuntimeSettings
from toxagent.tools.bootstrap import build_registry
from tests.support.predictor import ASPIRIN, StubPredictor

pytestmark = pytest.mark.anyio

ACTOR = Actor(subject_id="user-1")


@pytest.fixture
def now():
    return datetime.now(timezone.utc)


async def _seed(db, now, text: str):
    session = Session.create(ACTOR.subject_id, now=now)
    message = Message.create(session.id, Role.USER, 1, now=now)
    run = Run.create(session.id, message.id, Lane.AGENTIC, Intent.DECISION_SUPPORT, now=now)
    async with db.unit_of_work() as uow:
        await uow.sessions.add(session)
        await uow.messages.add(message)
        await uow.runs.add(run)
        uow.emit(session_id=session.id, type=EventType.SESSION_CREATED,
                 entity_type="session", entity_id=session.id)
        await uow.commit()
    predictor = StubPredictor().client()
    create_analysis = CreateAnalysis(db, predictor, PolicySettings())
    analysis = await create_analysis.execute(
        actor=ACTOR, session_id=session.id, run_id=run.id, smiles=ASPIRIN, owns_run=False,
    )
    gateway = AgentRuntimeGateway(
        db, registry=build_registry(db, predictor, create_analysis, PolicySettings()),
        capability_tokens=None, provider=None, settings=RuntimeSettings(),
    )
    context = RunContext(
        actor=ACTOR, session_id=session.id, run_id=run.id, intent=Intent.DECISION_SUPPORT,
        analysis_id=analysis.snapshot.id, text=text,
    )
    return gateway, context


async def test_the_turn_reads_its_contract_and_the_view(db, now, monkeypatch):
    monkeypatch.setenv("TOXAGENT_FLAG_INVESTIGATOR_CONTRACT_V1", "1")
    gateway, context = await _seed(
        db, now, "Does the measured IC50 contradict the prediction?")

    prompt, _profile, _deadline, _hash = await gateway._prepare_context(context)

    # the contract: route, what must be answered, the actions, the stops
    assert "Goal contract (investigate route)" in prompt
    assert "Does the measured IC50 contradict the prediction?" in prompt
    assert "Actions available to this turn:" in prompt
    assert "No other tool exists for you" in prompt
    # the typed stops, including the one #70 separated out
    assert "provider_failure" in prompt
    assert "needs_user_input" in prompt
    # and the budget it is spending
    assert "Budget:" in prompt


async def test_the_route_comes_from_what_the_run_is_bound_to(db, now, monkeypatch):
    """A value question with the analysis pinned is a read, not an
    investigation -- and the turn is told so, so it does not search."""
    monkeypatch.setenv("TOXAGENT_FLAG_INVESTIGATOR_CONTRACT_V1", "1")
    gateway, context = await _seed(db, now, "What is the hERG probability?")

    prompt, _profile, _deadline, _hash = await gateway._prepare_context(context)

    assert "Goal contract (read_explain route)" in prompt
    assert gateway._routes[context.run_id].route == router.READ_EXPLAIN
    assert gateway._routes[context.run_id].confidence == "high"


async def test_an_assessment_question_is_not_routed_to_a_lookup(db, now, monkeypatch):
    monkeypatch.setenv("TOXAGENT_FLAG_INVESTIGATOR_CONTRACT_V1", "1")
    gateway, context = await _seed(db, now, "Is this compound safe to take forward?")

    await gateway._prepare_context(context)

    decision = gateway._routes[context.run_id]
    assert decision.route == router.INVESTIGATE
    assert decision.confidence == "low"  # wording decided it, and the record says so


async def test_the_contract_is_recorded_where_a_grader_reads_it(db, now, monkeypatch):
    monkeypatch.setenv("TOXAGENT_FLAG_INVESTIGATOR_CONTRACT_V1", "1")
    gateway, context = await _seed(db, now, "Does the measured IC50 contradict it?")

    await gateway._prepare_context(context)
    await gateway._begin_decision_state(context)

    async with db.unit_of_work() as uow:
        state = await uow.decision_states.get(context.run_id)
    contract = state.goal_contract
    assert contract["route"] == router.INVESTIGATE
    assert [q["question"] for q in contract["must_answer"]] == [
        "Does the measured IC50 contradict it?",
    ]
    assert "submit_grounded_answer" in contract["admissible_actions"]
    assert contract["route_decision"]["reason_codes"]
    assert contract["stop_conditions"]["provider_failure"]


async def test_with_the_flag_off_the_prompt_is_what_it_was(db, now, monkeypatch):
    monkeypatch.delenv("TOXAGENT_FLAG_INVESTIGATOR_CONTRACT_V1", raising=False)
    gateway, context = await _seed(db, now, "Does the measured IC50 contradict it?")

    prompt, _profile, _deadline, _hash = await gateway._prepare_context(context)

    assert "Goal contract" not in prompt
    assert gateway._routes[context.run_id] is None
    assert gateway._contracts == {}

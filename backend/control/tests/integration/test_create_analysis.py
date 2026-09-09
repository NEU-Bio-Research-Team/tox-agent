"""The deterministic analysis workflow (plan section 7.1).

The exit gate for Phase 1 is that this path calls no model at all, so the stub
predictor here is the only outbound dependency in the test.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from toxagent.application.create_analysis import CreateAnalysis
from toxagent.application.policy import Actor
from toxagent.config import PolicySettings
from toxagent.domain.errors import EndpointUnavailable, Forbidden, InvalidSmiles, SessionNotFound
from toxagent.domain.events import EventType
from toxagent.domain.ids import new_id
from toxagent.domain.message import Message, Role
from toxagent.domain.run import Intent, Lane, Run, RunStatus
from toxagent.domain.session import Session
from tests.support.predictor import ASPIRIN, StubPredictor

pytestmark = pytest.mark.anyio

NOW = datetime(2026, 9, 4, tzinfo=timezone.utc)
ACTOR = Actor(subject_id="user-1")


async def seed(db, owner="user-1"):
    session = Session.create(owner, now=NOW)
    message = Message.create(session.id, Role.USER, 1, now=NOW)
    run = Run.create(session.id, message.id, Lane.DETERMINISTIC, Intent.ANALYSIS, now=NOW)
    async with db.unit_of_work() as uow:
        await uow.sessions.add(session)
        await uow.messages.add(message)
        await uow.runs.add(run)
        uow.emit(
            session_id=session.id, type=EventType.SESSION_CREATED,
            entity_type="session", entity_id=session.id,
        )
        await uow.commit()
    return session, run


def service(db, stub: StubPredictor, **policy):
    return CreateAnalysis(db, stub.client(), PolicySettings(**policy))


async def test_a_valid_molecule_produces_an_immutable_snapshot(db):
    session, run = await seed(db)
    stub = StubPredictor()
    result = await service(db, stub).execute(
        actor=ACTOR, session_id=session.id, run_id=run.id, smiles=ASPIRIN
    )

    assert result.snapshot.canonical_smiles == ASPIRIN
    assert result.snapshot.served_endpoints == ("herg", "tox21")
    assert result.observation.value_at("predictions.herg.probability_blocker") == 0.73064
    # The response is stored losslessly, not reshaped.
    assert result.snapshot.predictor_response["provenance"]["git_commit"].startswith("562b988")

    async with db.unit_of_work() as uow:
        stored = await uow.analyses.get(result.snapshot.id, session_id=session.id)
        assert stored.content_sha256 == result.snapshot.content_sha256
        assert (await uow.runs.get(run.id)).status is RunStatus.COMPLETED
        assert (await uow.sessions.get_unscoped(session.id)).active_analysis_id == result.snapshot.id


async def test_the_run_and_its_events_tell_the_same_story(db):
    session, run = await seed(db)
    await service(db, StubPredictor()).execute(
        actor=ACTOR, session_id=session.id, run_id=run.id, smiles=ASPIRIN
    )
    events = [e.type for e in await db.outbox().read_after(session.id, 0)]
    assert EventType.ANALYSIS_CREATED in events
    assert EventType.OBSERVATION_CREATED in events
    assert EventType.RUN_COMPLETED in events


async def test_the_same_molecule_twice_makes_one_snapshot(db):
    session, run = await seed(db)
    stub = StubPredictor()
    first = await service(db, stub).execute(
        actor=ACTOR, session_id=session.id, run_id=run.id, smiles=ASPIRIN
    )
    _, second_run = await seed(db)  # a second run in a new session would differ; reuse this one
    async with db.unit_of_work() as uow:
        message = Message.create(session.id, Role.USER, 2, now=NOW)
        another = Run.create(session.id, message.id, Lane.DETERMINISTIC, Intent.ANALYSIS, now=NOW)
        await uow.messages.add(message)
        await uow.runs.add(another)
        await uow.commit()
    second = await service(db, stub).execute(
        actor=ACTOR, session_id=session.id, run_id=another.id, smiles=ASPIRIN
    )
    assert second.reused
    assert second.snapshot.id == first.snapshot.id


async def test_an_invalid_smiles_creates_no_snapshot(db):
    session, run = await seed(db)
    with pytest.raises(InvalidSmiles):
        await service(db, StubPredictor()).execute(
            actor=ACTOR, session_id=session.id, run_id=run.id, smiles="not-a-molecule"
        )
    async with db.unit_of_work() as uow:
        assert await uow.analyses.list_for_session(session.id) == []


async def test_an_unserved_endpoint_creates_no_snapshot(db):
    session, run = await seed(db)
    stub = StubPredictor(served=("herg",))
    with pytest.raises(EndpointUnavailable):
        await service(db, stub).execute(
            actor=ACTOR, session_id=session.id, run_id=run.id, smiles=ASPIRIN,
            endpoints=("herg", "clintox"),
        )
    async with db.unit_of_work() as uow:
        assert await uow.analyses.list_for_session(session.id) == []


async def test_another_users_session_is_not_found(db):
    session, run = await seed(db, owner="alice")
    with pytest.raises(SessionNotFound):
        await service(db, StubPredictor()).execute(
            actor=Actor(subject_id="mallory"), session_id=session.id, run_id=run.id, smiles=ASPIRIN
        )


async def test_threshold_overrides_are_refused_by_default(db):
    session, run = await seed(db)
    with pytest.raises(Forbidden, match="disabled"):
        await service(db, StubPredictor()).execute(
            actor=ACTOR, session_id=session.id, run_id=run.id, smiles=ASPIRIN,
            threshold_overrides={"herg": 0.3},
        )


async def test_threshold_overrides_need_the_expert_role(db):
    session, run = await seed(db)
    svc = service(db, StubPredictor(), allow_threshold_overrides=True)
    with pytest.raises(Forbidden, match="expert role"):
        await svc.execute(
            actor=ACTOR, session_id=session.id, run_id=run.id, smiles=ASPIRIN,
            threshold_overrides={"herg": 0.3},
        )


async def test_an_authorised_override_is_recorded_in_the_snapshot(db):
    session, run = await seed(db)
    stub = StubPredictor()
    svc = service(db, stub, allow_threshold_overrides=True)
    result = await svc.execute(
        actor=Actor(subject_id="user-1", roles=frozenset({"expert"})),
        session_id=session.id, run_id=run.id, smiles=ASPIRIN,
        threshold_overrides={"herg": 0.3},
    )
    assert result.snapshot.policy_snapshot["threshold_override_source"] == "request_override"
    assert stub.requests[-1]["body"]["threshold_overrides"] == {"herg": 0.3}


async def test_no_attribution_or_research_happens_on_an_analysis(db):
    """Plan section 7.1: an analysis does exactly one thing."""
    session, run = await seed(db)
    stub = StubPredictor()
    await service(db, stub).execute(
        actor=ACTOR, session_id=session.id, run_id=run.id, smiles=ASPIRIN
    )
    assert [r["path"] for r in stub.requests] == ["/v1/predictions"]

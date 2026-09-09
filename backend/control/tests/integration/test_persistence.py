"""Store behaviour that the workflows depend on (plan sections 13.2, 13.3, 14.1).

These run against the same SQLAlchemy schema production uses, on SQLite. What
is being tested is the mapping and the transaction boundary, not the dialect.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from toxagent.domain.analysis import AnalysisSnapshot, PredictorProvenance
from toxagent.domain.answer import Claim, ClaimKind, GroundedAnswer, Limitation, LimitationCode
from toxagent.domain.errors import Conflict
from toxagent.domain.events import EventType
from toxagent.domain.ids import new_id
from toxagent.domain.message import Message, PartType, Role
from toxagent.domain.observation import Observation, ObservationKind, Producer
from toxagent.domain.run import Intent, Lane, Run, RunStatus
from toxagent.domain.session import Language, Session

pytestmark = pytest.mark.anyio

NOW = datetime(2026, 9, 4, tzinfo=timezone.utc)
PREDICTION = {
    "input_smiles": "CC(=O)Oc1ccccc1C(=O)O",
    "canonical_smiles": "CC(=O)Oc1ccccc1C(=O)O",
    "predictions": {
        "herg": {
            "probability_blocker": 0.73064, "label": "blocker", "threshold": 0.5,
            "threshold_source": "model_default", "model_id": "herg-chemberta-v1",
        }
    },
    "applicability": {"status": "ok", "method": "element_rules_v1", "reasons": []},
    "provenance": {"git_commit": "562b988"},
}


async def make_session(db, owner="user-1") -> Session:
    session = Session.create(owner, now=NOW, preferred_language=Language.VI)
    async with db.unit_of_work() as uow:
        await uow.sessions.add(session)
        uow.emit(
            session_id=session.id, type=EventType.SESSION_CREATED,
            entity_type="session", entity_id=session.id,
        )
        await uow.commit()
    return session


async def make_run(db, session: Session, lane=Lane.DETERMINISTIC, intent=Intent.ANALYSIS) -> Run:
    message = Message.create(
        session.id, Role.USER, 1, now=NOW,
        parts=((PartType.TEXT, {"text": "analyse this"}),),
    )
    run = Run.create(session.id, message.id, lane, intent, now=NOW)
    async with db.unit_of_work() as uow:
        await uow.messages.add(message)
        await uow.runs.add(run)
        await uow.commit()
    return run


# --- ownership -------------------------------------------------------------

async def test_a_foreign_session_reads_as_absent(db):
    session = await make_session(db, owner="alice")
    async with db.unit_of_work() as uow:
        assert await uow.sessions.get(session.id, owner_id="alice") is not None
        assert await uow.sessions.get(session.id, owner_id="mallory") is None


async def test_an_analysis_is_scoped_to_its_session(db):
    session = await make_session(db)
    other = await make_session(db)
    run = await make_run(db, session)
    snapshot = AnalysisSnapshot.create(
        session_id=session.id, run_id=run.id, input_smiles=PREDICTION["input_smiles"],
        requested_endpoints=("herg",), predictor_response=PREDICTION,
        provenance=PredictorProvenance(base_url_id="local"), policy_snapshot={}, now=NOW,
    )
    async with db.unit_of_work() as uow:
        await uow.analyses.add(snapshot)
        await uow.commit()
    async with db.unit_of_work() as uow:
        assert await uow.analyses.get(snapshot.id, session_id=session.id) is not None
        assert await uow.analyses.get(snapshot.id, session_id=other.id) is None


# --- outbox atomicity ------------------------------------------------------

async def test_events_and_state_commit_together(db):
    session = await make_session(db)
    run = await make_run(db, session)
    async with db.unit_of_work() as uow:
        running = run.transition(RunStatus.RUNNING, now=NOW)
        await uow.runs.update(running, expected_version=run.version)
        uow.emit(
            session_id=session.id, type=EventType.RUN_STARTED, entity_type="run",
            entity_id=run.id, run_id=run.id,
        )
        await uow.commit()

    events = await db.outbox().read_after(session.id, 0)
    assert [e.type for e in events] == [EventType.SESSION_CREATED, EventType.RUN_STARTED]
    assert [e.sequence for e in events] == [1, 2]


async def test_a_rolled_back_transaction_emits_nothing(db):
    session = await make_session(db)
    before = await db.outbox().latest_sequence(session.id)
    with pytest.raises(RuntimeError):
        async with db.unit_of_work() as uow:
            uow.emit(
                session_id=session.id, type=EventType.RUN_QUEUED,
                entity_type="run", entity_id=new_id("run"),
            )
            raise RuntimeError("workflow blew up after emitting")
    assert await db.outbox().latest_sequence(session.id) == before
    assert len(await db.outbox().read_after(session.id, 0)) == 1  # only session.created


async def test_sequences_are_contiguous_across_units_of_work(db):
    session = await make_session(db)
    for _ in range(5):
        async with db.unit_of_work() as uow:
            uow.emit(
                session_id=session.id, type=EventType.PART_CREATED,
                entity_type="part", entity_id=new_id("prt"),
            )
            await uow.commit()
    events = await db.outbox().read_after(session.id, 0)
    # 1 is session.created; the five parts continue from there without a gap.
    assert [e.sequence for e in events] == [1, 2, 3, 4, 5, 6]


async def test_a_client_resumes_from_its_last_sequence(db):
    session = await make_session(db)
    async with db.unit_of_work() as uow:
        for _ in range(3):
            uow.emit(
                session_id=session.id, type=EventType.PART_CREATED,
                entity_type="part", entity_id=new_id("prt"),
            )
        await uow.commit()
    tail = await db.outbox().read_after(session.id, 2)
    assert [e.sequence for e in tail] == [3, 4]


# --- optimistic concurrency ------------------------------------------------

async def test_a_stale_session_write_conflicts(db):
    session = await make_session(db)
    async with db.unit_of_work() as uow:
        await uow.sessions.update(
            session.with_active_analysis(new_id("ana"), now=NOW), expected_version=session.version
        )
        await uow.commit()
    async with db.unit_of_work() as uow:
        with pytest.raises(Conflict):
            await uow.sessions.update(
                session.with_active_analysis(new_id("ana"), now=NOW),
                expected_version=session.version,
            )


async def test_updating_a_session_does_not_rewind_the_event_counter(db):
    """A caller holding a stale aggregate must not roll the feed backwards."""
    session = await make_session(db)  # event_sequence is now 1 in the database
    async with db.unit_of_work() as uow:
        await uow.sessions.update(
            session.with_active_analysis(new_id("ana"), now=NOW), expected_version=session.version
        )
        uow.emit(
            session_id=session.id, type=EventType.ANALYSIS_CREATED,
            entity_type="analysis", entity_id=new_id("ana"),
        )
        await uow.commit()
    assert await db.outbox().latest_sequence(session.id) == 2


# --- idempotency -----------------------------------------------------------

async def test_the_same_molecule_and_policy_finds_the_existing_snapshot(db):
    session = await make_session(db)
    run = await make_run(db, session)
    snapshot = AnalysisSnapshot.create(
        session_id=session.id, run_id=run.id, input_smiles="CCO",
        requested_endpoints=("herg",), predictor_response=PREDICTION,
        provenance=PredictorProvenance(base_url_id="local", artifact_hashes=("sha-a",)),
        policy_snapshot={"herg": 0.5}, now=NOW,
    )
    async with db.unit_of_work() as uow:
        await uow.analyses.add(snapshot)
        await uow.commit()
    async with db.unit_of_work() as uow:
        found = await uow.analyses.find_by_idempotency_key(session.id, snapshot.idempotency_key)
        assert found is not None and found.id == snapshot.id


async def test_a_duplicate_client_message_id_is_rejected_by_the_database(db):
    session = await make_session(db)
    first = Message.create(session.id, Role.USER, 1, now=NOW, client_message_id="web-1")
    second = Message.create(session.id, Role.USER, 2, now=NOW, client_message_id="web-1")
    async with db.unit_of_work() as uow:
        await uow.messages.add(first)
        await uow.commit()
    with pytest.raises(Exception):
        async with db.unit_of_work() as uow:
            await uow.messages.add(second)
            await uow.commit()


# --- the claim source graph ------------------------------------------------

async def test_an_accepted_answer_round_trips_with_its_source_graph(db):
    session = await make_session(db)
    run = await make_run(db, session, lane=Lane.AGENTIC, intent=Intent.REPORT_QA)
    observation = Observation.create(
        session_id=session.id, run_id=run.id, producer=Producer.PREDICTOR,
        kind=ObservationKind.PREDICTION, schema_version="prediction-v1",
        canonical_payload=PREDICTION, model_projection={"herg": {"probability_blocker": 0.731}},
        provenance={"git_commit": "562b988"}, now=NOW,
        required_limitations=("uncalibrated_probability",),
    )
    claim = Claim(
        claim_id=new_id("clm"), kind=ClaimKind.NUMERIC,
        text="Predicted hERG blocker probability is 0.731.",
        observation_id=observation.id, field_path="predictions.herg.probability_blocker",
        source_value=0.73064, rendered_value="0.731", transform="round:3",
    )
    answer = GroundedAnswer.create(
        session_id=session.id, run_id=run.id, answer_markdown="…", claims=(claim,),
        limitations=(Limitation(LimitationCode.UNCALIBRATED_PROBABILITY, "not calibrated"),),
        now=NOW,
    )
    async with db.unit_of_work() as uow:
        await uow.observations.add(observation)
        await uow.answers.add(answer)
        await uow.commit()

    async with db.unit_of_work() as uow:
        loaded = await uow.answers.get(answer.id, session_id=session.id)
        assert loaded is not None
        (restored,) = loaded.claims
        assert restored.source_value == 0.73064
        assert restored.field_path == "predictions.herg.probability_blocker"
        source = await uow.observations.get(restored.observation_id, session_id=session.id)
        assert source is not None
        assert source.value_at(restored.field_path) == restored.source_value


async def test_two_candidate_generations_cannot_collide(db):
    session = await make_session(db)
    run = await make_run(db, session, lane=Lane.AGENTIC, intent=Intent.REPORT_QA)
    first = GroundedAnswer.create(
        session_id=session.id, run_id=run.id, answer_markdown="a", claims=(),
        candidate_generation=1, now=NOW,
    )
    clash = GroundedAnswer.create(
        session_id=session.id, run_id=run.id, answer_markdown="b", claims=(),
        candidate_generation=1, now=NOW,
    )
    async with db.unit_of_work() as uow:
        await uow.answers.add(first)
        await uow.commit()
    with pytest.raises(Exception):
        async with db.unit_of_work() as uow:
            await uow.answers.add(clash)
            await uow.commit()


# --- run cancellation flag -------------------------------------------------

async def test_cancel_is_recorded_as_a_request_not_as_an_outcome(db):
    session = await make_session(db)
    run = await make_run(db, session)
    async with db.unit_of_work() as uow:
        assert await uow.runs.request_cancel(run.id) is True
        await uow.commit()
    async with db.unit_of_work() as uow:
        assert await uow.runs.cancel_requested(run.id) is True
        assert (await uow.runs.get(run.id)).status is RunStatus.QUEUED


async def test_a_terminal_run_cannot_be_asked_to_cancel(db):
    session = await make_session(db)
    run = await make_run(db, session)
    async with db.unit_of_work() as uow:
        done = run.transition(RunStatus.RUNNING, now=NOW).transition(RunStatus.COMPLETED, now=NOW)
        await uow.runs.update(done, expected_version=run.version)
        await uow.commit()
    async with db.unit_of_work() as uow:
        assert await uow.runs.request_cancel(run.id) is False

"""A record's full text, read once and kept (#43), against a real database."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from tests.support.research import ACCEPTED_HIT, POTENCY_JATS, FullTextStubProvider
from toxagent.application.investigation import evidence_text
from toxagent.domain.answer import Claim, ClaimKind, GroundedAnswer
from toxagent.domain.events import EventType
from toxagent.domain.evidence import EvidenceRecord, EvidenceStatus, SourceType
from toxagent.domain.ids import CLAIM, new_id
from toxagent.domain.message import Message, Role
from toxagent.domain.run import Intent, Lane, Run
from toxagent.domain.session import Session
from toxagent.research.text_access import TextAccess

pytestmark = pytest.mark.anyio

NOW = datetime(2026, 10, 21, tzinfo=timezone.utc)


async def _rig(db, *, source_type: SourceType = SourceType.ARTICLE, provider: str = "stub"):
    session = Session.create("user-1", now=NOW)
    message = Message.create(session.id, Role.USER, 1, now=NOW)
    run = Run.create(session.id, message.id, Lane.AGENTIC, Intent.DECISION_SUPPORT, now=NOW)
    record = EvidenceRecord.create(
        session_id=session.id, provider=provider,
        provider_record_id=ACCEPTED_HIT.provider_record_id, source_type=source_type,
        title=ACCEPTED_HIT.title, retrieved_at=NOW, identifier=ACCEPTED_HIT.identifier,
        abstract_or_excerpt=ACCEPTED_HIT.abstract_or_excerpt,
        normalized_facts={"text_access": "abstract_only"},
    ).to_status(EvidenceStatus.NORMALIZED).to_status(EvidenceStatus.ACCEPTED)
    async with db.unit_of_work() as uow:
        await uow.sessions.add(session)
        await uow.messages.add(message)
        await uow.runs.add(run)
        await uow.evidence.add(record)
        uow.emit(session_id=session.id, type=EventType.SESSION_CREATED,
                 entity_type="session", entity_id=session.id)
        await uow.commit()
    return session, run, record


async def _read(db, provider, session, run, record):
    return await evidence_text.read_full_text(
        db, provider, session_id=session.id, run_id=run.id, record=record, now=NOW,
    )


async def test_a_text_is_fetched_once_and_then_read_from_the_store(db):
    session, run, record = await _rig(db)
    provider = FullTextStubProvider(documents={record.provider_record_id: POTENCY_JATS})

    first = await _read(db, provider, session, run, record)
    second = await _read(db, provider, session, run, record)

    assert first.state == TextAccess.FULL_TEXT_RETRIEVED.value and first.fetched_now
    assert second.state == first.state and not second.fetched_now
    assert provider.fetches == [record.provider_record_id]
    # what was stored is what was parsed, cells and all
    row = second.full_text.passage("table-wrap:1/row:3")
    assert row.text == "Compound: compound 7 | hERG IC50 (uM): 0.38"
    assert row.cells == ("compound 7", "0.38")
    assert second.full_text.document_sha256 == first.full_text.document_sha256
    assert second.full_text_id == "PMC1111111"


async def test_a_definitive_refusal_is_kept_with_its_reason(db):
    session, run, record = await _rig(db)
    provider = FullTextStubProvider(documents={})

    first = await _read(db, provider, session, run, record)
    second = await _read(db, provider, session, run, record)

    assert first.state == second.state == TextAccess.ACCESS_FAILED.value
    assert "PMCID" in second.reason
    assert provider.fetches == [record.provider_record_id]


async def test_an_outage_is_reported_and_not_kept(db):
    """It says nothing about the document, so the next call asks again."""
    session, run, record = await _rig(db)
    provider = FullTextStubProvider(documents={record.provider_record_id: POTENCY_JATS})
    provider.outage = True

    failed = await _read(db, provider, session, run, record)
    provider.outage = False
    recovered = await _read(db, provider, session, run, record)

    assert failed.state == TextAccess.ACCESS_FAILED.value and failed.full_text is None
    assert recovered.state == TextAccess.FULL_TEXT_RETRIEVED.value
    assert len(provider.fetches) == 2


@pytest.mark.parametrize(
    ("source_type", "provider_name"),
    [(SourceType.DATABASE, "stub"), (SourceType.ARTICLE, "chembl")],
)
async def test_a_record_this_provider_cannot_serve_is_not_asked_for(db, source_type, provider_name):
    session, run, record = await _rig(db, source_type=source_type, provider=provider_name)
    provider = FullTextStubProvider(documents={record.provider_record_id: POTENCY_JATS})

    read = await _read(db, provider, session, run, record)

    assert provider.fetches == []
    assert read.full_text is None
    # what the search knew is reported, not an invented failure
    assert read.state == "abstract_only"


async def test_two_reads_of_one_record_keep_the_first_text(db):
    """Two parallel reads both miss the store; the second must neither fail
    on the key nor replace a text that may already be quoted."""
    session, run, record = await _rig(db)
    provider = FullTextStubProvider(documents={record.provider_record_id: POTENCY_JATS})
    first = await _read(db, provider, session, run, record)
    duplicate = evidence_text.EvidenceFullText.failed(
        evidence_id=record.id, session_id=session.id, run_id=run.id,
        reason="late duplicate", full_text_id=None, now=NOW,
    )
    async with db.unit_of_work() as uow:
        kept = await uow.evidence_text.add_if_absent(duplicate)
        await uow.commit()
    assert kept.state == first.state
    assert kept.document_sha256 == first.full_text.document_sha256


async def test_a_claims_spans_survive_the_answer_store(db):
    session, run, record = await _rig(db)
    claim = Claim(
        claim_id=new_id(CLAIM), kind=ClaimKind.SCIENTIFIC, text="Compound 7 is most potent.",
        citation_ids=(record.id,),
        evidence_spans=((record.id, "table-wrap:1/row:3"), (record.id, "table-wrap:1")),
    )
    answer = GroundedAnswer.create(
        session_id=session.id, run_id=run.id, answer_markdown="Compound 7 is most potent.",
        claims=(claim,), now=NOW,
    )
    async with db.unit_of_work() as uow:
        await uow.answers.add(answer)
        await uow.commit()
    async with db.unit_of_work() as uow:
        loaded = await uow.answers.get(answer.id, session_id=session.id)
    assert loaded.claims[0].evidence_spans == claim.evidence_spans
    assert loaded.content_sha256 == answer.content_sha256

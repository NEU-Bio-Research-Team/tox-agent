"""#47 groundwork: what a run submitted, how claims carry across a repair, and
what the reviewer is shown -- against a real database.

Three things the release gate needs before it can be wired: the raw draft is
kept beside the accepted one, a claim keeps its draft label so a repair can be
compared claim by claim, and the reviewer judges a claim against the passage
it quotes and the scope it declared, not only an abstract's first 1,500
characters.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from tests.support.research import POTENCY_JATS
from toxagent.application.conversation.submit_answer import SubmitAnswer
from toxagent.application.investigation import claim_review, evidence_text
from toxagent.domain.errors import AnswerValidationFailed
from toxagent.domain.evidence import EvidenceRecord, EvidenceStatus, SourceType
from toxagent.domain.evidence_text import EvidenceFullText
from toxagent.domain.ids import TOOL_CALL, new_id
from toxagent.platform.config import PolicySettings
from toxagent.research.fulltext import parse_jats
from toxagent.validation.answer.candidate_wire import LimitationCandidate
from toxagent.validation.answer.draft_wire import (
    ClaimCandidateV2Spans,
    EvidenceRelationInputV2,
    GroundedAnswerDraftV2Spans,
)
from tests.integration.test_evidence_relations_wiring import rig

pytestmark = pytest.mark.anyio

NOW = datetime(2026, 10, 21, tzinfo=timezone.utc)


async def _article(db, session, run, *, read: bool) -> EvidenceRecord:
    record = EvidenceRecord.create(
        session_id=session.id, provider="europepmc", provider_record_id="MED:1",
        source_type=SourceType.ARTICLE, title="hERG potency of a series", retrieved_at=NOW,
        abstract_or_excerpt="We screened a series for hERG blockade.",
    ).to_status(EvidenceStatus.NORMALIZED).to_status(EvidenceStatus.ACCEPTED)
    async with db.unit_of_work() as uow:
        await uow.evidence.add(record)
        if read:
            document = parse_jats(POTENCY_JATS)
            await uow.evidence_text.add_if_absent(EvidenceFullText.retrieved(
                evidence_id=record.id, session_id=session.id, run_id=run.id,
                passages=tuple(evidence_text._passage(p) for p in document.passages),
                section_titles=document.section_titles, omitted={}, truncated=False,
                full_text_id="PMC1", now=NOW,
            ))
        await uow.commit()
    return record


def _draft(observation, record, *, limitations, spans=True) -> GroundedAnswerDraftV2Spans:
    return GroundedAnswerDraftV2Spans(
        answer_markdown="Predicted hERG blocker probability is {{herg_p}}; compound 7 was "
                        "the most potent in the paper's table.",
        claims=[
            {"local_ref": "herg_p", "kind": "numeric", "text": "Predicted probability.",
             "observation_id": observation.id,
             "field_path": "predictions.herg.probability_blocker", "transform": "round:3"},
            ClaimCandidateV2Spans(
                local_ref="table_row", kind="scientific",
                text="Compound 7 was the most potent hERG blocker in the retrieved table.",
                citation_ids=[record.id],
                evidence_spans=[{"evidence_id": record.id, "locator": "table-wrap:1/row:3"}]
                if spans else [],
            ),
        ],
        limitations=[LimitationCandidate(code=c, text="") for c in limitations],
        evidence_relations=[EvidenceRelationInputV2(
            proposition="compound 7 blocks hERG", source_class="external_experimental",
            source_id=record.id, relation="supports", reason_codes=["measured_value"],
            endpoint="herg", ligand_id="compound 7", assay_type="patch clamp",
        )],
    )


async def test_every_draft_is_kept_with_what_the_server_made_of_it(db, monkeypatch):
    monkeypatch.setenv("TOXAGENT_FLAG_EVIDENCE_FULL_TEXT_V1", "1")
    session, run, observation = await rig(db)
    record = await _article(db, session, run, read=True)
    submit = SubmitAnswer(db, PolicySettings(max_answer_candidates_per_run=2))

    with pytest.raises(AnswerValidationFailed):
        # missing the probability caveat
        await submit.execute(session_id=session.id, run_id=run.id, language="en",
                             candidate=_draft(observation, record,
                                              limitations=["evidence_scope_limited"]))
    async with db.unit_of_work() as uow:
        rejected = await uow.answer_drafts.list_for_run(run.id)
    assert [d["outcome"] for d in rejected] == ["rejected"]
    assert rejected[0]["payload"]["claims"][1]["local_ref"] == "table_row"
    assert {"missing_required_limitation", "citation_not_read"} <= {
        v["code"] for v in rejected[0]["violations"]
    }


async def test_a_claim_keeps_its_draft_label_and_the_reviewer_sees_what_it_quotes(
    db, monkeypatch,
):
    monkeypatch.setenv("TOXAGENT_FLAG_EVIDENCE_FULL_TEXT_V1", "1")
    session, run, observation = await rig(db)
    record = await _article(db, session, run, read=True)
    outcome = await _submit_reading(db, session, run, observation, record)

    claims = {c.local_ref: c for c in outcome.answer.claims}
    assert set(claims) == {"herg_p", "table_row"}
    async with db.unit_of_work() as uow:
        stored = await uow.answers.get(outcome.answer.id, session_id=session.id)
        bundle = await claim_review.review_bundle(uow, session_id=session.id, answer=stored)
    assert {c.local_ref for c in stored.claims} == {"herg_p", "table_row"}

    by_text = {c["text"]: c for c in bundle["claims"]}
    (evidence_source,) = by_text[
        "Compound 7 was the most potent hERG blocker in the retrieved table."
    ]["sources"]
    assert evidence_source["text_access"] == "full_text_retrieved"
    (passage,) = evidence_source["quoted_passages"]
    assert passage["locator"] == "table-wrap:1/row:3"
    assert "0.38" in passage["text"]
    (relation,) = evidence_source["relations"]
    assert relation["scope"] == {"endpoint": "herg", "ligand_id": "compound 7",
                                 "assay_type": "patch clamp"}
    (observation_source,) = by_text["Predicted probability."]["sources"]
    assert observation_source["required_limitations"]


async def _submit_reading(db, session, run, observation, record):
    """Submit as a run that read the record's text: the read is what makes the
    span acceptable, so it is recorded the way the tool runner records it."""
    call_id = new_id(TOOL_CALL)
    async with db.unit_of_work() as uow:
        assert await uow.tool_calls.try_reserve(
            call_id=call_id, session_id=session.id, run_id=run.id,
            tool_name="read_evidence_text", arguments_sha256="0" * 64, now=NOW,
            max_calls=None, max_identical=3,
        )
        await uow.tool_calls.finish(
            call_id, status="completed", error_code=None, observation_ids=[record.id],
            duration_ms=1, now=NOW,
        )
        await uow.commit()
    return await SubmitAnswer(db, PolicySettings()).execute(
        session_id=session.id, run_id=run.id, language="en",
        candidate=_draft(observation, record, limitations=[
            "uncalibrated_probability", "evidence_scope_limited",
        ]),
    )

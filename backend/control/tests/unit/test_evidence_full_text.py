"""Full text as something read, selected and quoted (#43), without a database.

The integration tests cover storing it; these cover what is pure: what a
stored text must hold, how a page of it is chosen, which numbers a quoted
passage grounds, and when an answer owes the limitation that its paper was
read from the abstract only.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from tests.support.research import POTENCY_JATS
from toxagent.application.investigation import evidence_text
from toxagent.domain.evidence import EvidenceRecord, SourceType
from toxagent.domain.evidence_text import EvidenceFullText
from toxagent.domain.ids import EVIDENCE, RUN, SESSION, new_id
from toxagent.research.fulltext import parse_jats
from toxagent.validation.answer.coverage import passage_values
from toxagent.validation.limitations import required_for_answer, text_for
from toxagent.domain.answer import LimitationCode

NOW = datetime(2026, 10, 21, tzinfo=timezone.utc)
SESSION_ID, RUN_ID = new_id(SESSION), new_id(RUN)


def _full_text(xml: str = POTENCY_JATS) -> EvidenceFullText:
    document = parse_jats(xml)
    return EvidenceFullText.retrieved(
        evidence_id=new_id(EVIDENCE), session_id=SESSION_ID, run_id=RUN_ID,
        passages=tuple(evidence_text._passage(p) for p in document.passages),
        section_titles=document.section_titles, omitted=document.omitted,
        truncated=document.truncated, full_text_id="PMC1", now=NOW,
    )


def _record(source_type: SourceType = SourceType.ARTICLE) -> EvidenceRecord:
    return EvidenceRecord.create(
        session_id=SESSION_ID, provider="europepmc", provider_record_id="MED:1",
        source_type=source_type, title="t", retrieved_at=NOW,
    )


# ── the stored text ───────────────────────────────────────────────────────

def test_a_failed_access_must_say_why():
    with pytest.raises(ValueError, match="record why"):
        EvidenceFullText(
            evidence_id=new_id(EVIDENCE), session_id=SESSION_ID, run_id=RUN_ID,
            state="access_failed", fetched_at=NOW,
        )


def test_a_retrieved_text_must_hold_passages():
    with pytest.raises(ValueError, match="at least one passage"):
        EvidenceFullText.retrieved(
            evidence_id=new_id(EVIDENCE), session_id=SESSION_ID, run_id=RUN_ID,
            passages=(), section_titles=(), omitted={}, truncated=False,
            full_text_id=None, now=NOW,
        )


def test_the_same_passages_hash_the_same():
    assert _full_text().document_sha256 == _full_text().document_sha256


def test_a_cell_of_a_stored_row_resolves_by_locator():
    full_text = _full_text()
    cell = full_text.passage("table-wrap:1/row:3/cell:2")
    assert cell is not None and cell.text == "0.38"
    assert full_text.passage("table-wrap:1/row:3/cell:3") is None


# ── choosing a page ───────────────────────────────────────────────────────

def test_a_page_of_table_rows_skips_the_prose():
    selection = evidence_text.select(_full_text(), kinds=["table_row"])
    assert [p.locator for p in selection.passages] == [
        "table-wrap:1/row:2", "table-wrap:1/row:3",
    ]
    assert selection.next_offset is None


def test_a_section_filter_matches_tables_by_their_label():
    selection = evidence_text.select(_full_text(), section="table 1")
    assert {p.kind for p in selection.passages} == {"table", "table_header", "table_row"}


def test_named_locators_come_back_in_the_order_asked():
    selection = evidence_text.select(
        _full_text(), locators=["table-wrap:1/row:3", "sec:1/p:1", "nowhere:1"],
    )
    assert [p.locator for p in selection.passages] == ["table-wrap:1/row:3", "sec:1/p:1"]


def test_pages_continue_from_next_offset():
    first = evidence_text.select(_full_text(), limit=2)
    second = evidence_text.select(_full_text(), offset=first.next_offset, limit=2)
    assert first.next_offset == 2
    assert {p.locator for p in first.passages}.isdisjoint(p.locator for p in second.passages)
    assert first.total == second.total


# ── whether an article was read ───────────────────────────────────────────

def test_an_article_with_no_text_read_is_unread():
    assert evidence_text.text_not_read(_record(), None)


def test_an_article_whose_text_was_refused_is_unread():
    refused = EvidenceFullText.failed(
        evidence_id=new_id(EVIDENCE), session_id=SESSION_ID, run_id=RUN_ID,
        reason="no PMCID", full_text_id=None, now=NOW,
    )
    assert evidence_text.text_not_read(_record(), refused)


def test_an_article_read_in_full_is_read():
    assert not evidence_text.text_not_read(_record(), _full_text())


def test_a_database_record_has_no_text_to_have_left_unread():
    """A ChEMBL activity is a structured record; "the paper was not read in
    full" means nothing for it."""
    assert not evidence_text.text_not_read(_record(SourceType.DATABASE), None)


# ── the numbers a span grounds ────────────────────────────────────────────

@pytest.mark.parametrize(
    ("text", "values"),
    [
        # "IC50" is a name, not a value: a digit run glued to letters is skipped.
        ("Compound: compound 7 | hERG IC50 (uM): 0.38", (7.0, 0.38)),
        ("Number | 14 | 14 | 30", (14.0, 14.0, 30.0)),
        ("IC50 0,38 µM", (0.38,)),
        ("inhibition 65%", (0.65,)),
    ],
)
def test_a_passage_grounds_the_numbers_it_contains(text, values):
    assert passage_values(text) == values


# ── the limitation ────────────────────────────────────────────────────────

def test_citing_an_unread_article_owes_the_limitation():
    required = required_for_answer([], cited_evidence=True, cited_article_unread=True)
    assert required == {"evidence_scope_limited", "evidence_full_text_not_read"}
    assert "evidence_full_text_not_read" not in required_for_answer([], cited_evidence=True)


def test_the_scope_wording_stops_claiming_no_full_text_was_read(monkeypatch):
    """Without the tool, "full texts were not read" is true of every answer.
    With it, that half belongs to evidence_full_text_not_read."""
    default = text_for(LimitationCode.EVIDENCE_SCOPE_LIMITED)
    assert "full texts were not read" in default
    monkeypatch.setenv("TOXAGENT_FLAG_EVIDENCE_FULL_TEXT_V1", "1")
    with_tool = text_for(LimitationCode.EVIDENCE_SCOPE_LIMITED)
    assert "full texts" not in with_tool
    assert "abstract only" in text_for(LimitationCode.EVIDENCE_FULL_TEXT_NOT_READ)


# ── the surface a model is shown ──────────────────────────────────────────

def _answer_schema() -> dict:
    from toxagent.platform.config import PolicySettings
    from toxagent.tools.definitions import answer

    (definition,) = answer.build(None, PolicySettings())
    return definition.input_model.model_json_schema()


def test_without_the_flag_claims_cannot_name_spans(monkeypatch):
    monkeypatch.setenv("TOXAGENT_FLAG_ANSWER_DRAFT_V2", "1")
    assert "evidence_spans" not in str(_answer_schema())


def test_with_the_flag_a_claim_can_name_spans(monkeypatch):
    monkeypatch.setenv("TOXAGENT_FLAG_ANSWER_DRAFT_V2", "1")
    monkeypatch.setenv("TOXAGENT_FLAG_EVIDENCE_FULL_TEXT_V1", "1")
    assert "evidence_spans" in str(_answer_schema())


def test_a_span_must_belong_to_a_cited_record():
    from toxagent.validation.answer.draft_wire import ClaimCandidateV2Spans

    evidence_id = new_id(EVIDENCE)
    with pytest.raises(ValueError, match="does not cite"):
        ClaimCandidateV2Spans(
            local_ref="row", kind="scientific", text="t", citation_ids=[],
            evidence_spans=[{"evidence_id": evidence_id, "locator": "table-wrap:1/row:2"}],
        )
    with pytest.raises(ValueError):
        ClaimCandidateV2Spans(
            local_ref="row", kind="scientific", text="t", citation_ids=[evidence_id],
            evidence_spans=[{"evidence_id": evidence_id, "locator": "somewhere in results"}],
        )


def test_the_prompt_explains_the_tool_only_where_it_exists(monkeypatch):
    from toxagent.harness.context import FULL_TEXT_POLICY, build_system_prompt
    from toxagent.harness.context import SessionCheckpoint

    def prompt() -> str:
        return build_system_prompt(
            capability_profile="decision_support", checkpoint=SessionCheckpoint(),
            pinned=(), recent_messages=(), answer_schema="grounded-answer-v2",
        )

    assert FULL_TEXT_POLICY not in prompt()
    monkeypatch.setenv("TOXAGENT_FLAG_EVIDENCE_FULL_TEXT_V1", "1")
    assert FULL_TEXT_POLICY in prompt()


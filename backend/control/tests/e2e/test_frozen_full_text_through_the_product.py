"""#43 on a frozen deployment: a paper's table read with the network off.

The frozen evidence provider serves search from a pinned snapshot and, given a
pinned full-text store, serves read_evidence_text from what was captured --
never from Europe PMC. On the scripted runtime:

* the text tool exists on the frozen deployment, and a value only the table
  holds is quoted through its row and validated against the captured text;
* a captured access failure reaches the answer as an unread source;
* a locator the store does not hold cannot be cited;
* what was read records that it was a replay, and of which capture.
"""
from __future__ import annotations

import json

import pytest

from tests.e2e.test_full_text_through_the_product import ROW, _draft, _run, _Script
from tests.support.research import POTENCY_JATS
from toxagent.domain.provenance import content_sha256
from toxagent.platform.config import ResearchSettings
from toxagent.research.fulltext import parse_jats
from toxagent.research.fulltext_store import freeze_document, write_store
from toxagent.research.interfaces import FullTextFetch
from toxagent.research.providers import build_provider

pytestmark = pytest.mark.anyio

RECORD = "MED:11111111"


@pytest.fixture
def flags_on(monkeypatch):
    monkeypatch.setenv("TOXAGENT_FLAG_ANSWER_DRAFT_V2", "1")
    monkeypatch.setenv("TOXAGENT_FLAG_EVIDENCE_FULL_TEXT_V1", "1")


def _frozen(tmp_path, *, readable: bool):
    evidence = {
        "search": {"herg": [RECORD]},
        "records": {RECORD: {
            "provider_record_id": RECORD, "source_type": "article",
            "title": "hERG blockade screening of a marine natural product series",
            "authors": ["Doe J"], "published_at": "2025-06-01",
            "canonical_url": "https://europepmc.org/article/MED/11111111",
            "identifier": {"pmid": "11111111", "pmcid": "PMC1111111"},
            "abstract_or_excerpt": "This study screened a series of compounds for hERG "
                                   "channel blockade.",
            "normalized_facts": {"journal": "J Med Chem"},
        }},
    }
    body = {"fixture_version": "eval-fixture-v1", "name": "frozen-potency", "evidence": evidence}
    body["content_sha256"] = content_sha256(body)
    fixture = tmp_path / "fixture.json"
    fixture.write_text(json.dumps(body))
    if readable:
        document = parse_jats(POTENCY_JATS)
        fetch = FullTextFetch(retrieved=True, passages=document.passages,
                              section_titles=document.section_titles,
                              omitted=dict(document.omitted), full_text_id="PMC1111111",
                              provenance={"source_sha256": "b" * 64})
    else:
        fetch = FullTextFetch(retrieved=False, full_text_id="PMC1111111",
                              reason="the provider holds no full text for PMC1111111")
    store = tmp_path / "full-text.json"
    pin = write_store({RECORD: freeze_document(fetch, provider="europepmc",
                                               captured_at="2026-10-21T00:00:00+00:00")}, store)
    return build_provider(ResearchSettings(
        provider="snapshot", snapshot_path=str(fixture),
        full_text_path=str(store), full_text_sha256=pin,
    ))


async def test_a_table_value_is_read_and_quoted_offline(db, flags_on, tmp_path):
    async def steps(turn, evidence_id, seen) -> None:
        seen["read"] = await turn.call_tool("read_evidence_text", {
            "evidence_id": evidence_id, "kinds": ["table_row"],
        })
        markdown = "In the retrieved study's table, compound 7 had a hERG IC50 of 0.38 uM."
        seen["unknown_locator"] = await turn.call_tool("submit_grounded_answer", _draft(
            evidence_id, markdown=markdown, spans=["table-wrap:1/row:9"],
            limitations=["evidence_scope_limited"],
        ))
        seen["with_span"] = await turn.call_tool("submit_grounded_answer", _draft(
            evidence_id, markdown=markdown, spans=[ROW], limitations=["evidence_scope_limited"],
        ))

    provider = _frozen(tmp_path, readable=True)
    script = _Script(steps)
    run, evidence, answer = await _run(db, provider, script)

    view = script.seen["read"]["model_view"]
    assert view["text_access"] == "full_text_retrieved"
    assert view["text_provenance"]["replay"] is True
    assert view["text_provenance"]["served_by"] == "snapshot"
    assert view["text_provenance"]["captured_from"] == "europepmc"
    assert "0.38" in view["passages"][-1]["untrusted"]["content"]
    refused = script.seen["unknown_locator"]
    assert refused["status"] == "error"
    assert script.seen["with_span"]["status"] == "completed", script.seen["with_span"]
    assert run["status"] == "completed", run
    (record,) = [e for e in evidence if e["evidence_id"] == script.seen["evidence_id"]]
    assert record["full_text"]["state"] == "full_text_retrieved"
    assert record["full_text"]["provenance"]["replay"] is True
    assert record["full_text"]["provenance"]["source_sha256"] == "b" * 64
    (claim,) = answer["claims"]
    assert claim["evidence_spans"][0]["locator"] == ROW


async def test_a_captured_access_failure_reaches_the_answer(db, flags_on, tmp_path):
    async def steps(turn, evidence_id, seen) -> None:
        seen["read"] = await turn.call_tool("read_evidence_text", {"evidence_id": evidence_id})
        seen["answer"] = await turn.call_tool("submit_grounded_answer", _draft(
            evidence_id, markdown="The retrieved study screened the series for hERG blockade.",
            spans=[], limitations=["evidence_scope_limited"],
        ))

    script = _Script(steps)
    run, evidence, _ = await _run(db, _frozen(tmp_path, readable=False), script)

    assert script.seen["read"]["model_view"]["text_access"] == "access_failed"
    refused = script.seen["answer"]
    assert refused["status"] == "error"
    assert refused["error"]["details"]["violations"][0]["expected"] == [
        "evidence_full_text_not_read"
    ]
    (record,) = [e for e in evidence if e["evidence_id"] == script.seen["evidence_id"]]
    assert record["full_text"]["state"] == "access_failed"

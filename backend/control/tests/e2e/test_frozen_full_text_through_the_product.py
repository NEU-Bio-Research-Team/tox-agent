"""#43 on a frozen deployment: a paper's table read with the network off.

The frozen evidence provider serves search from a pinned snapshot and, given a
pinned full-text store, serves read_evidence_text from what was captured --
never from Europe PMC. On the scripted runtime:

* the text tool exists on the frozen deployment, and a value only the table
  holds is quoted through its row and validated against the captured text;
* a captured access failure reaches the answer as an unread source;
* a locator the store does not hold cannot be cited;
* what was read records that it was a replay, and of which capture;
* the passage viewer's read model (AG-UI) shows the cited row under its
  header, the cited cell, and the provider's reason when nothing was read.
"""
from __future__ import annotations

import json

import pytest

from tests.e2e.test_full_text_through_the_product import ROW, _draft, _run, _Script
from tests.support.api import AUTH
from tests.support.research import POTENCY_JATS
from toxagent.api.responses import EvidencePassagesView
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


async def test_the_passage_viewer_shows_the_cited_row_under_its_header(db, flags_on, tmp_path):
    """AG-UI: what a reader checks a table quote against -- the table's
    caption, the header the row sits under, the row's cells as written and the
    cell a claim names -- from the frozen store, with the capture it replays."""
    async def steps(turn, evidence_id, seen) -> None:
        await turn.call_tool("read_evidence_text", {
            "evidence_id": evidence_id, "kinds": ["table_row"],
        })
        seen["answer"] = await turn.call_tool("submit_grounded_answer", _draft(
            evidence_id, markdown="In the retrieved study's table, compound 7 had a hERG "
                                  "IC50 of 0.38 uM.",
            spans=[ROW], limitations=["evidence_scope_limited"],
        ))

    views: dict = {}

    async def inspect(client, session_id) -> None:
        evidence_id = script.seen["evidence_id"]
        base = f"/v1/sessions/{session_id}/evidence/{evidence_id}/passages"
        for name, params in (("outline", {}), ("row", {"locator": ROW}),
                             ("cell", {"locator": f"{ROW}/cell:2"}),
                             ("missing", {"locator": "table-wrap:1/row:9"}),
                             ("prose", {"locator": "sec:1/p:1"})):
            response = await client.get(base, params=params, headers=AUTH)
            assert response.status_code == 200, response.text
            # The generated frontend types are this model's; hold the route to it.
            EvidencePassagesView.model_validate(response.json())
            views[name] = response.json()
        views["unknown"] = await client.get(
            f"/v1/sessions/{session_id}/evidence/evd_{'0' * 32}/passages", headers=AUTH,
        )

    script = _Script(steps)
    run, _, _ = await _run(db, _frozen(tmp_path, readable=True), script, inspect=inspect)
    assert run["status"] == "completed", run

    outline = views["outline"]
    assert outline["schema_version"] == "evidence-passages-v1"
    assert outline["untrusted_external_content"] is True
    assert outline["access"]["state"] == "full_text_retrieved"
    assert outline["access"]["provenance"]["replay"] is True
    assert outline["access"]["provenance"]["captured_from"] == "europepmc"
    assert outline["focus"] is None
    assert ROW in [p["locator"] for p in outline["passages"]]

    table = views["row"]["focus"]["table"]
    assert views["row"]["focus"]["found"] is True
    assert views["row"]["passages"] == []
    assert table["label"] == "Table 1"
    assert "hERG potency by patch clamp" in table["caption"]
    assert table["header"] == ["Compound", "hERG IC50 (uM)"]
    assert table["cells"] == ["compound 7", "0.38"]
    assert table["cited_cell"] is None
    # The unit stays in the header, as written; nothing parses it.
    assert table["columns"][1] == {"header": "hERG IC50 (uM)", "value": "0.38", "cited": False}

    cell = views["cell"]["focus"]
    assert cell["passage"]["kind"] == "table_cell" and cell["passage"]["text"] == "0.38"
    assert cell["table"]["cited_cell"] == 2
    assert [c["cited"] for c in cell["table"]["columns"]] == [False, True]

    assert views["missing"]["focus"] == {"locator": "table-wrap:1/row:9", "found": False,
                                         "passage": None, "table": None}
    prose = views["prose"]["focus"]
    assert prose["found"] is True and prose["table"] is None
    assert views["unknown"].status_code == 404


async def test_the_passage_viewer_says_why_nothing_was_read(db, flags_on, tmp_path):
    async def steps(turn, evidence_id, seen) -> None:
        await turn.call_tool("read_evidence_text", {"evidence_id": evidence_id})
        await turn.call_tool("submit_grounded_answer", _draft(
            evidence_id, markdown="The retrieved study screened the series for hERG blockade.",
            spans=[], limitations=["evidence_scope_limited", "evidence_full_text_not_read"],
        ))

    views: dict = {}

    async def inspect(client, session_id) -> None:
        response = await client.get(
            f"/v1/sessions/{session_id}/evidence/{script.seen['evidence_id']}/passages",
            params={"locator": ROW}, headers=AUTH,
        )
        assert response.status_code == 200, response.text
        EvidencePassagesView.model_validate(response.json())
        views["failed"] = response.json()

    script = _Script(steps)
    await _run(db, _frozen(tmp_path, readable=False), script, inspect=inspect)

    access = views["failed"]["access"]
    assert access["state"] == "access_failed"
    assert access["reason"] == "the provider holds no full text for PMC1111111"
    assert access["passage_count"] == 0
    # A locator into a text that was never read is not found, never guessed.
    assert views["failed"]["focus"] == {"locator": ROW, "found": False,
                                        "passage": None, "table": None}

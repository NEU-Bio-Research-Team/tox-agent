"""The passage viewer's focus (AG-UI): a cited table row under its header,
values as written, never re-aligned when the row does not line up."""
from __future__ import annotations

from datetime import datetime, timezone

from toxagent.application.investigation.evidence_text import (
    passage_focus,
    passage_outline,
)
from toxagent.domain.evidence_text import EvidenceFullText, EvidencePassage

T = "table-wrap:1"


def _text(*passages):
    return EvidenceFullText.retrieved(
        evidence_id="evd_" + "1" * 32, session_id="ses_" + "2" * 32, run_id="run_" + "3" * 32,
        passages=tuple(passages), section_titles=("Results",), omitted={}, truncated=False,
        full_text_id="PMC1", now=datetime(2026, 10, 22, tzinfo=timezone.utc),
    )


DOC = _text(
    EvidencePassage("sec:1/p:1", "Results", "paragraph", "Potencies are in Table 1."),
    EvidencePassage(T, "Table 1", "table", "Table 1 hERG potency by patch clamp, 37 C"),
    EvidencePassage(f"{T}/row:1", "Table 1", "table_header", "Compound | IC50 (uM)",
                    cells=("Compound", "IC50 (uM)")),
    EvidencePassage(f"{T}/row:2", "Table 1", "table_row", "Compound: c4 | IC50 (uM): > 30",
                    cells=("c4", "> 30")),
    EvidencePassage(f"{T}/row:3", "Table 1", "table_row", "c7 | 0.38 | n=3",
                    cells=("c7", "0.38", "n=3")),
)


def test_a_comparator_in_a_cell_stays_part_of_the_value():
    focus = passage_focus(DOC, f"{T}/row:2/cell:2")
    table = focus["table"]
    assert focus["passage"]["text"] == "> 30"
    assert table["caption"].endswith("37 C")
    assert table["columns"] == [{"header": "Compound", "value": "c4", "cited": False},
                                {"header": "IC50 (uM)", "value": "> 30", "cited": True}]


def test_a_ragged_row_is_shown_as_it_is_not_realigned():
    table = passage_focus(DOC, f"{T}/row:3")["table"]
    assert table["header"] == ["Compound", "IC50 (uM)"]
    assert table["cells"] == ["c7", "0.38", "n=3"]
    assert table["columns"] == []


def test_prose_has_no_table_and_an_unknown_locator_is_not_found():
    assert passage_focus(DOC, "sec:1/p:1")["table"] is None
    assert passage_focus(DOC, f"{T}/row:9") == {
        "locator": f"{T}/row:9", "found": False, "passage": None, "table": None}
    assert passage_focus(DOC, f"{T}/row:2/cell:7")["found"] is False


def test_the_table_passage_itself_has_no_row():
    table = passage_focus(DOC, T)["table"]
    assert table["row_locator"] is None and table["header"] == []


def test_the_outline_lists_every_passage_in_document_order():
    outline = passage_outline(DOC)
    assert [p["locator"] for p in outline] == [
        "sec:1/p:1", T, f"{T}/row:1", f"{T}/row:2", f"{T}/row:3"]
    assert passage_outline(DOC, limit=2)[-1]["locator"] == T

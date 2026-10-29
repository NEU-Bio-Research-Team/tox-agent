"""Retrieval recall is placed before extraction (#43): found, read, read in
full, quoted -- each counted on its own, from the product trace alone."""
from __future__ import annotations

from evals.investigation.retrieval import retrieval_recall, within

CASE = {
    "evidence_spans": [
        {"evidence_ref": "pmid:11111111", "locator": "table-wrap:1"},
        {"evidence_ref": "pmid:11111111", "locator": "table-wrap:1/row:3"},
        {"evidence_ref": "doi:10.1000/OTHER", "locator": "sec:2/p:1"},
        {"evidence_ref": "evi_patch", "locator": "abstract"},
    ]
}


def _trace(*, read_tools=("read_evidence_text",), state="full_text_retrieved", spans=()):
    return {
        "evidence": [
            {"evidence_id": "evd_a", "identifier": {"pmid": "11111111"},
             "full_text": {"state": state} if state else None},
            {"evidence_id": "evd_b", "identifier": {"doi": "10.1000/unrelated"}},
        ],
        "runs": [{
            "run": {"tool_calls": [
                {"tool_name": tool, "status": "completed", "observation_ids": ["evd_a"]}
                for tool in read_tools
            ]},
            "answer": {"claims": [{
                "citation_ids": ["evd_a"],
                "evidence_spans": [{"evidence_id": "evd_a", "locator": loc} for loc in spans],
            }]},
        }],
    }


def test_each_stage_is_counted_on_its_own():
    result = retrieval_recall(CASE, _trace(spans=["table-wrap:1/row:3"]))
    assert result["records"] == {
        "retrieved": {"n": 1, "of": 2}, "read": {"n": 1, "of": 2}, "cited": {"n": 1, "of": 2},
    }
    assert result["spans"] == {
        "retrieved": {"n": 2, "of": 3}, "text_read": {"n": 2, "of": 3},
        "quoted": {"n": 2, "of": 3},
    }


def test_a_case_local_name_is_not_counted_as_a_miss():
    result = retrieval_recall(CASE, _trace())
    assert result["unusable_refs"] == ["evi_patch"]
    assert result["spans"]["retrieved"]["of"] == 3


def test_an_abstract_only_read_is_retrieved_but_not_read_in_full():
    result = retrieval_recall(CASE, _trace(read_tools=("get_evidence_record",), state=None))
    assert result["records"]["read"]["n"] == 1
    assert result["spans"]["text_read"]["n"] == 0


def test_quoting_the_caption_does_not_meet_a_gold_row():
    result = retrieval_recall(CASE, _trace(spans=["table-wrap:1"]))
    rows = {row["locator"]: row["quoted"] for row in result["span_rows"]}
    assert rows["table-wrap:1"] is True
    assert rows["table-wrap:1/row:3"] is False


def test_locators_nest_one_way():
    assert within("table-wrap:1/row:3/cell:2", "table-wrap:1/row:3")
    assert within("table-wrap:1/row:3", "table-wrap:1")
    assert not within("table-wrap:1", "table-wrap:1/row:3")
    assert not within("table-wrap:10/row:1", "table-wrap:1")


def test_a_case_without_spans_is_not_scored():
    assert retrieval_recall({}, _trace())["scored"] is False

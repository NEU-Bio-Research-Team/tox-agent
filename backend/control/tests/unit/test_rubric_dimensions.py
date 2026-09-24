"""A rubric that cannot let three good dimensions cover for the bad one (K11).

`rubric` was one grader name and `rubric_notes` one free-text string. A single
blended judgement is the wrong shape for this product's failure mode: an
answer that is on topic, well written, and cites a URL that does not exist
scores well on three counts, and the one that matters is the fourth.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.graders.rubric import DIMENSIONS, check_judgement, passed, prompt_for

SCHEMA = Path(__file__).resolve().parents[2] / "evals" / "schema" / "task.schema.json"


def judgement(**verdicts) -> dict:
    complete = {dimension.name: {"verdict": "pass"} for dimension in DIMENSIONS}
    complete.update(verdicts)
    return {"dimensions": complete}


def test_the_four_dimensions_are_separate_and_two_of_them_are_blocking():
    assert [d.name for d in DIMENSIONS] == [
        "url_validity", "claim_support", "topic_relevance", "source_quality",
    ]
    blocking = {d.name for d in DIMENSIONS if d.blocking}
    assert blocking == {"url_validity", "claim_support"}


def test_a_fabricated_citation_fails_however_good_the_rest_is():
    """The whole point. Three passes do not outvote this one."""
    verdict = judgement(
        url_validity={"verdict": "fail", "reason": "the DOI does not resolve"}
    )
    assert passed(verdict) is False


def test_a_citation_that_does_not_support_its_sentence_fails_the_same_way():
    verdict = judgement(
        claim_support={"verdict": "fail", "reason": "the paper is about a different assay"}
    )
    assert passed(verdict) is False


def test_a_non_blocking_failure_also_fails_the_task():
    """Blocking is about what cannot be traded away, not about what may be
    ignored: the task still has to pass every dimension."""
    assert passed(judgement(source_quality={"verdict": "fail", "reason": "a preprint"})) is False


def test_a_complete_pass_passes():
    assert passed(judgement()) is True


def test_an_unjudged_dimension_is_not_a_pass():
    """"Nobody judged this" and "this failed" are different facts, and only
    one of them is about the product."""
    partial = {"dimensions": {"url_validity": {"verdict": "pass"}}}
    assert passed(partial) is None
    problems = check_judgement(partial)
    assert any("claim_support" in problem for problem in problems)


def test_a_failure_has_to_say_why():
    verdict = judgement(url_validity={"verdict": "fail"})
    assert any("say why" in problem for problem in check_judgement(verdict))
    assert passed(verdict) is None


def test_a_verdict_that_is_not_pass_or_fail_is_not_a_judgement():
    assert check_judgement(judgement(topic_relevance={"verdict": "mostly"}))


def test_the_prompt_carries_each_dimension_with_its_own_notes():
    """Guidance about citations must not be readable as guidance about tone."""
    prompt = prompt_for({
        "task_id": "evsyn-01",
        "rubric_notes": {"url_validity": "EuropePMC ids only", "general": "Vietnamese answer"},
    })
    by_name = {entry["name"]: entry for entry in prompt["dimensions"]}
    assert by_name["url_validity"]["notes"] == "EuropePMC ids only"
    assert by_name["topic_relevance"]["notes"] == ""
    assert prompt["general_notes"] == "Vietnamese answer"


def test_the_older_single_string_form_still_reads():
    prompt = prompt_for({"task_id": "evsyn-01", "rubric_notes": "be strict about citations"})
    assert prompt["general_notes"] == "be strict about citations"


def test_the_task_schema_accepts_per_dimension_notes():
    schema = json.loads(SCHEMA.read_text())["properties"]["rubric_notes"]
    object_form = next(form for form in schema["oneOf"] if form["type"] == "object")
    assert set(object_form["properties"]) == {
        "general", "url_validity", "claim_support", "topic_relevance", "source_quality",
    }
    assert object_form["additionalProperties"] is False

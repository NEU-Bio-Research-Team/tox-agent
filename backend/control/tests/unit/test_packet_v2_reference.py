"""A grader reference names limitation codes and forbidden inferences; it must
also say what they mean, or two graders grade "L8" two different ways."""
from __future__ import annotations

from types import SimpleNamespace

from evals.investigation import packet_v2
from evals.investigation.task_groups import FORBIDDEN_INFERENCES, LIMITATION_CODES, LIMITATION_GUIDE

GOLD = {
    "status": "synthetic_engineering_gold", "reference_outcome": "review; toxicity not run",
    "acceptable_conclusions": [], "mandatory_facts": [],
    "required_limitations": ["L7", "L8"], "optional_limitations": ["L9"],
    "limitation_scoring": "scored",
    "forbidden_inferences": ["not_run_as_negative", "pic50_as_efficacy"],
}


def _reference() -> str:
    loaded = SimpleNamespace(case={"case_id": "inv-x", "task_group": "T1",
                                   "answerability": "answerable_from_corpus"})
    return packet_v2.reference_markdown(loaded, GOLD)


def test_every_code_the_reference_names_comes_with_its_meaning():
    text = _reference()
    for code in ("L7", "L8", "L9"):
        assert f"- `{code}`: {LIMITATION_GUIDE[code]}" in text
    for code in GOLD["forbidden_inferences"]:
        assert f"- `{code}`: {FORBIDDEN_INFERENCES[code]}" in text


def test_the_guide_covers_the_whole_taxonomy_and_the_readme_carries_it():
    assert set(LIMITATION_GUIDE) == set(LIMITATION_CODES)
    for code, meaning in LIMITATION_GUIDE.items():
        assert f"- `{code}`: {meaning}" in packet_v2.GRADER_README
    assert "{limitation_guide}" not in packet_v2.GRADER_README

"""The rubric, split into dimensions that cannot cover for each other (K11).

`rubric` was one grader name and `rubric_notes` one free-text string handed to
whoever judges. A single blended judgement is the wrong shape for this
product's failure mode: an answer that is on topic, well written and cites a
URL that does not exist scores well on three counts, and the one that matters
is the fourth. Averaging lets the three carry the one.

Four dimensions, judged and reported separately:

* `url_validity` — does every cited URL exist, resolve to the record claimed,
  and was it served by the provider rather than composed by the model? A
  fabricated citation is not a partial answer.
* `topic_relevance` — is the cited work about the substance and the endpoint
  asked about?
* `claim_support` — does the cited passage support the specific sentence that
  cites it, rather than the general area?
* `source_quality` — is the source what the answer implies it is: peer
  reviewed where that is claimed, primary where that is claimed?

`url_validity` and `claim_support` are `blocking`: failing either is failing
the task, whatever the other two say. That is a statement about what this
product may not do, not a weighting to be tuned.

Nothing here scores anything. Judging is a model or SME job (plan 9.3), and a
grader that quietly scored it would be exactly the un-run judgement the
deferral exists to prevent. This module defines the dimensions, checks a task
declares them coherently, and produces the per-dimension record a judge fills
in — so a returned judgement missing a dimension is detectable rather than
silently absent.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Dimension:
    name: str
    question: str
    #: A failure here fails the task regardless of the other dimensions.
    blocking: bool


DIMENSIONS: tuple[Dimension, ...] = (
    Dimension(
        "url_validity",
        "Does every cited URL exist, resolve to the record cited, and come from the "
        "provider rather than from the model? A composed URL is a fabrication.",
        blocking=True,
    ),
    Dimension(
        "claim_support",
        "Does the cited passage support the specific sentence citing it, rather than "
        "the general subject area?",
        blocking=True,
    ),
    Dimension(
        "topic_relevance",
        "Is the cited work about the substance and the endpoint that was asked about?",
        blocking=False,
    ),
    Dimension(
        "source_quality",
        "Is the source what the answer implies — peer reviewed where that is claimed, "
        "primary where that is claimed?",
        blocking=False,
    ),
)

BY_NAME = {dimension.name: dimension for dimension in DIMENSIONS}


def prompt_for(task: dict[str, Any]) -> dict[str, Any]:
    """What a judge is asked, one entry per dimension.

    Task-specific guidance is attached per dimension rather than as one blob,
    so guidance about citations cannot be read as guidance about tone.
    """
    notes = task.get("rubric_notes") or {}
    if isinstance(notes, str):
        # The older single-string form, kept readable rather than dropped.
        notes = {"general": notes}
    return {
        "task_id": task["task_id"],
        "general_notes": notes.get("general", ""),
        "dimensions": [
            {
                "name": dimension.name,
                "question": dimension.question,
                "blocking": dimension.blocking,
                "notes": notes.get(dimension.name, ""),
            }
            for dimension in DIMENSIONS
        ],
    }


def check_judgement(judgement: dict[str, Any]) -> list[str]:
    """Problems with a returned judgement. Empty means it is usable.

    A judgement missing a dimension is not a pass on that dimension, and a
    suite that treated it as one would report a rubric as clean because
    nobody looked at the part that matters.
    """
    problems: list[str] = []
    verdicts = judgement.get("dimensions") or {}
    for dimension in DIMENSIONS:
        verdict = verdicts.get(dimension.name)
        if verdict is None:
            problems.append(f"{dimension.name}: not judged")
            continue
        if verdict.get("verdict") not in {"pass", "fail"}:
            problems.append(f"{dimension.name}: verdict must be pass or fail")
        if verdict.get("verdict") == "fail" and not (verdict.get("reason") or "").strip():
            problems.append(f"{dimension.name}: a failure has to say why")
    return problems


def passed(judgement: dict[str, Any]) -> bool | None:
    """Whether a complete judgement is a pass. None when it is incomplete.

    Not an average, and not a majority: any blocking dimension failing fails
    the task. `None` rather than `False` for an incomplete judgement, because
    "nobody judged this" and "this failed" are different facts and only one of
    them is about the product.
    """
    if check_judgement(judgement):
        return None
    verdicts = judgement["dimensions"]
    return all(
        verdicts[dimension.name]["verdict"] == "pass"
        for dimension in DIMENSIONS
        if dimension.blocking
    ) and all(verdicts[d.name]["verdict"] == "pass" for d in DIMENSIONS)

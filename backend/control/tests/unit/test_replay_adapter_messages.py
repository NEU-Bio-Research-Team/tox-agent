"""What the researcher read for a run that did not publish (protocol v2).

The API's Message has no run_id; a message names its run through its parts.
Matching on a run_id field matched nothing, so every withheld result was
recorded as an empty reply -- graded as "no response" -- when the reader had
been shown the server's confirmed facts and the reason.
"""
from __future__ import annotations

from evals.investigation.adapters.toxagent_replay import messages_for_run, visible_text

RUN = "run_" + "1" * 32
OTHER = "run_" + "2" * 32
ANSWER = "ans_" + "1" * 32


def _message(sequence, *parts, role="assistant"):
    return {"message_id": f"msg_{sequence}", "role": role, "sequence": sequence,
            "created_at": "2026-10-22T00:00:00+00:00", "client_message_id": None,
            "parts": [{"part_id": f"prt_{sequence}_{i}", "index": i, "type": t, "content": c}
                      for i, (t, c) in enumerate(parts)]}


WITHHELD = _message(
    2, ("text", {"text": "The answer could not be released: the reviewer still blocked a claim.\n"
                         "What the server confirmed: the policy decision is review."}),
    ("result_ref", {"run_id": RUN, "disposition": "partial_facts", "review_status": "blocked"}),
)
PUBLISHED_ELSEWHERE = _message(
    4, ("text", {"text": "another run's answer"}), ("answer_ref", {"answer_id": "ans_x"}),
    ("result_ref", {"run_id": OTHER, "disposition": "publish"}),
)
USER = _message(1, ("text", {"text": "why review?"}), role="user")


def test_a_withheld_result_is_read_from_its_result_ref_message():
    found = messages_for_run([USER, WITHHELD, PUBLISHED_ELSEWHERE], RUN)
    assert found == [WITHHELD]
    text = visible_text(None, found, [], published=False)
    assert text.startswith("The answer could not be released")
    assert "the policy decision is review" in text


def test_a_message_without_a_result_ref_is_matched_by_the_run_answer():
    legacy = _message(3, ("text", {"text": "old answer"}), ("answer_ref", {"answer_id": ANSWER}))
    assert messages_for_run([legacy], RUN, ANSWER) == [legacy]
    assert messages_for_run([legacy], RUN) == []


def test_another_runs_messages_and_the_users_are_never_this_runs():
    assert messages_for_run([USER, PUBLISHED_ELSEWHERE], RUN) == []

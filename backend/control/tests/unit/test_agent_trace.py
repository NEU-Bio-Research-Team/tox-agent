"""The chat's live log: reasoning, tool arguments, and what each turn runs on.

Display only. Reasoning is mirrored as its own event type -- never as message
text, which stays unpersisted until an answer is validated -- and it is
throttled so a long thought is a handful of rows, not one per token.
"""
from __future__ import annotations

from toxagent.harness.adapters.opencode_v1 import OpenCodeV1Provider
from toxagent.harness.gateway.completion import REASONING_MAX_CHARS, _ReasoningMirror
from toxagent.harness.provider import RuntimeEventType
from toxagent.tools.runner import ARGUMENTS_PREVIEW_CHARS, _arguments_preview


def _envelope(part: dict, **properties) -> dict:
    return {"payload": {"type": "message.part.updated",
                        "properties": {"part": {"sessionID": "rs_1", **part}, **properties}}}


def _normalize(envelope: dict):
    return OpenCodeV1Provider._normalize_event(object.__new__(OpenCodeV1Provider), envelope, "rs_1")


def test_a_reasoning_part_becomes_a_reasoning_event_with_its_whole_text():
    event = _normalize(_envelope(
        {"type": "reasoning", "id": "prt_1", "text": "First, the hERG", "time": {"start": 1}},
        delta=" hERG",
    ))
    assert event.type is RuntimeEventType.REASONING_DELTA
    assert event.payload == {"part_id": "prt_1", "text": "First, the hERG", "done": False}


def test_a_finished_reasoning_part_says_so():
    event = _normalize(_envelope(
        {"type": "reasoning", "id": "prt_1", "text": "done thinking", "time": {"start": 1, "end": 2}}
    ))
    assert event.payload["done"] is True


def test_reasoning_never_becomes_message_text():
    event = _normalize(_envelope({"type": "reasoning", "id": "prt_1", "text": "IC50 is 3 µM"}))
    assert event.type is not RuntimeEventType.MESSAGE_DELTA


def test_another_sessions_reasoning_is_ignored():
    envelope = _envelope({"type": "reasoning", "id": "prt_1", "text": "x"})
    envelope["payload"]["properties"]["part"]["sessionID"] = "rs_other"
    assert _normalize(envelope) is None


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_the_mirror_throttles_a_streaming_part_and_always_sends_the_last_word():
    clock = _Clock()
    mirror = _ReasoningMirror(clock=clock)
    assert mirror.accept({"part_id": "p", "text": "a"})["text"] == "a"
    clock.now = 0.3
    assert mirror.accept({"part_id": "p", "text": "ab"}) is None
    clock.now = 1.5
    assert mirror.accept({"part_id": "p", "text": "abc"})["text"] == "abc"
    clock.now = 1.6
    final = mirror.accept({"part_id": "p", "text": "abcd", "done": True})
    assert final["text"] == "abcd" and final["done"] is True
    # Nothing after the end of a part.
    assert mirror.accept({"part_id": "p", "text": "abcde"}) is None


def test_held_reasoning_is_flushed_when_the_turn_ends():
    clock = _Clock()
    mirror = _ReasoningMirror(clock=clock)
    mirror.accept({"part_id": "p", "text": "a"})
    mirror.accept({"part_id": "p", "text": "ab"})
    flushed = mirror.flush()
    assert [(item["text"], item["done"]) for item in flushed] == [("ab", True)]
    assert mirror.flush() == []


def test_a_long_thought_keeps_its_tail_and_says_it_was_cut():
    shaped = _ReasoningMirror().accept({"part_id": "p", "text": "x" * (REASONING_MAX_CHARS + 10)})
    assert len(shaped["text"]) == REASONING_MAX_CHARS
    assert shaped["truncated"] is True and shaped["chars"] == REASONING_MAX_CHARS + 10


def test_tool_arguments_preview_is_one_bounded_line():
    assert _arguments_preview({}) is None
    assert _arguments_preview({"skill_id": "assess-conflicting-evidence"}) == \
        '{"skill_id": "assess-conflicting-evidence"}'
    long = _arguments_preview({"query": "q" * 1000})
    assert len(long) == ARGUMENTS_PREVIEW_CHARS and long.endswith("…")

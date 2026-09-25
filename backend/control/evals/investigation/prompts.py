"""The one prompt every general platform receives, so arms differ only in the model.

The preamble is deliberately neutral: it tells a platform who is asking, not
how to answer. Nothing about limitations, citations or hedging is said, because
what a platform does unprompted is what a researcher using it would get. The
same text, hashed, is recorded in every platform record.

The ``with_snapshot`` arms (RETHINK §5.4 system B) add the predictor's own
output for the compound, byte-identical to what the predictor-template arm
renders from, so B differs from the bare platform only by that snapshot.
Multi-turn cases are stitched into one transcript per turn, the same way for
every platform, because the command-line clients are stateless.
"""
from __future__ import annotations

import json
from typing import Any, Sequence

from evals.investigation.cases import sha256

PREAMBLE = (
    "You are assisting a scientist in early-stage drug discovery. Answer their "
    "message as you would for a colleague."
)

SNAPSHOT_HEADER = (
    "For reference, this is the output of our in-house toxicity predictor (ToxPred) "
    "for the compound in the question, as JSON:"
)

PROMPT_VERSION = "platform-prompt-v1"


def preamble_sha256() -> str:
    return sha256(PREAMBLE)


def render(
    turns: Sequence[dict[str, Any]], index: int, previous_responses: Sequence[str], *,
    snapshot: dict[str, Any] | None = None,
) -> str:
    """The full text a stateless platform receives for turn ``index``."""
    parts = [PREAMBLE]
    if snapshot is not None:
        parts.append(
            SNAPSHOT_HEADER + "\n```json\n"
            + json.dumps(snapshot, indent=2, sort_keys=True, ensure_ascii=False) + "\n```"
        )
    if index == 0:
        parts.append(turns[0]["text"])
        return "\n\n".join(parts)
    transcript = []
    for earlier in range(index):
        transcript.append(f"Scientist: {turns[earlier]['text']}")
        transcript.append(f"You: {previous_responses[earlier]}")
    parts.append("Conversation so far:\n\n" + "\n\n".join(transcript))
    parts.append(f"Scientist's new message:\n{turns[index]['text']}")
    return "\n\n".join(parts)

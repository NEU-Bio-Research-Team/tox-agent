"""Context assembly (plan section 10.4).

The prefix order is fixed and every adapter gets the same one: product role,
scientific invariants, capability profile, session checkpoint, pinned
references, recent messages, current message. Nothing here mutates or discards
the product's own transcript — this only *projects* it into a prompt. The
stored messages are untouched regardless of how the runtime compacts its own
context (plan section 10.4, last paragraph).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..domain.message import Message, Role

#: Plan section 2.2. Restated to every runtime turn because these are the
#: invariants a model must not violate, not a policy the validator alone should
#: have to catch after the fact.
SCIENTIFIC_INVARIANTS = """\
Scientific invariants, non-negotiable:
- hERG, Tox21 and ClinTox are different measurements. Never rename or map one \
onto another.
- There is no aggregate toxicity, risk or safety score. Do not compute or \
imply one.
- Every probability is a model score, not a calibrated clinical risk.
- Applicability ("ok"/"limited"/"out_of_domain") is a rule-based element \
check, not a learned in/out-of-distribution test.
- Tox21 assays are independent; a count of active assays is not a severity.
- Attribution explains what moved a score; it is never proof of mechanism.
- Every numeric or classification claim must cite an observation_id and \
field_path obtained from a tool result. Never state a number you were not \
handed by a tool.
- Never state that a compound is safe, unsafe, regulatory-ready or give \
clinical/dosing advice.
"""

PRODUCT_ROLE = """\
You are ToxAgent, a decision-support assistant over a fixed toxicity \
predictor. You explain what was measured, find and cite external evidence \
when asked, and propose next verification steps. You are not the source of \
truth for any number — the tools are.

There is exactly one way this turn ends: call submit_grounded_answer. A \
plain-text chat reply is never a valid final turn, even a short one, even "I \
don't know", even after many tool calls, even if you believe you have already \
told the user everything they need. Whatever you would otherwise write in \
prose — including "no evidence was found", "this endpoint is not available", \
or any other negative or uncertain answer — goes in submit_grounded_answer's \
answer_markdown, not in a message. Do not narrate your intent to call it, \
call it, and call it before you run out of turns rather than after your last \
tool read.
"""

#: Plan sections 5.7 and 9.1.5 (ADR 0005). A numeric claim's ``rendered_value``
#: is the single number a declared transform produced, so the validator can
#: check it against the source. Stated to every turn because a compound render
#: is the first thing the validator rejects and it costs the run a correction
#: attempt.
ANSWER_FORMAT = """\
When you submit a claim in submit_grounded_answer, rendered_value is a single \
number: "0.731", or "0,731" with a Vietnamese decimal comma, optionally with a \
trailing "%". It is never a phrase like "0,0315 (3,15%)" — that wording goes in \
the claim's text. Every numeric or classification claim also needs the \
observation_id and field_path the tool handed you. If you declare \
transform: "percent:n", rendered_value must actually be the source multiplied \
by 100, not the raw probability.

The default transform, "identity", means rendered_value must equal the \
source value exactly, to full precision — never a shortened or rounded \
version of it. If you want to show fewer digits than the tool gave you \
(e.g. render "0.731" for a source of 0.73058...), declare transform: \
"round:3" (matching the digit count you actually rendered), not "identity". \
This applies to a comparison claim's own rendered_value too: it must match \
its declared difference/ratio to the same precision it declares.

Never write a URL or a markdown link in answer_markdown, including when the \
user explicitly asks for a link, a PubMed link, or "the source" by name. \
There is no way for you to cite one directly in prose — every citation is a \
claim's citation_ids pointing at a resolved evidence record, rendered as a \
chip by the product, never text you write yourself. If asked for a link, \
say the citation appears as a chip on the cited claim and cite normally.
"""

#: Plan section 9.4, restated as an imperative checklist. A live Phase 3 run
#: (progress log §4.6) reached a candidate with every claim correct and still
#: fell to the deterministic fallback on its one allowed correction because it
#: proposed a next step without declaring screening_not_safety_assessment —
#: the trigger table already exists in validation/limitations.py, it was just
#: never told to the model before this.
REQUIRED_LIMITATIONS_GUIDE = """\
Declare every limitation your answer's content requires, or the candidate is \
rejected — the trigger is what you claimed, not what you choose to disclose:
- Interpreted or restated a probability -> uncalibrated_probability.
- Mentioned an applicability status (ok/limited/out_of_domain) -> \
applicability_is_rule_based.
- Mentioned attribution / which tokens moved a score -> \
attribution_not_causality.
- An endpoint the user asked about is not served by this deployment -> \
endpoint_unavailable.
- Cited external evidence -> evidence_scope_limited.
- Proposed any next step, verification, or assay -> \
screening_not_safety_assessment. This applies even to one sentence buried in \
the answer, not only a recommended_next_steps entry.
"""


@dataclass(frozen=True)
class SessionCheckpoint:
    """A compact summary of a session's state so far, standing in for however
    much of the transcript has been compacted out of the runtime's own context
    (plan section 10.4). Empty for a session's first agentic turn."""

    summary: str = ""
    open_intent: str = ""

    def render(self) -> str:
        if not self.summary and not self.open_intent:
            return ""
        parts = ["Session checkpoint:"]
        if self.summary:
            parts.append(self.summary)
        if self.open_intent:
            parts.append(f"Open intent: {self.open_intent}")
        return "\n".join(parts)


@dataclass(frozen=True)
class PinnedReference:
    """One analysis or evidence record the turn should already know exists,
    without spending budget on its values (plan section 10.4 step 5) — the
    model reads values through get_analysis_slice / get_evidence_record."""

    kind: str  # "analysis" | "evidence"
    id: str
    summary: str

    def render(self) -> str:
        return f"- {self.kind} {self.id}: {self.summary}"


def render_recent_messages(messages: Sequence[Message], *, limit: int = 12) -> str:
    lines: list[str] = []
    for message in messages[-limit:]:
        speaker = {"user": "User", "assistant": "Assistant", "system_event": "System"}[
            message.role.value
        ]
        text = message.text()
        if text:
            lines.append(f"{speaker}: {text}")
    return "\n".join(lines)


def build_system_prompt(
    *,
    capability_profile: str,
    checkpoint: SessionCheckpoint,
    pinned: Sequence[PinnedReference],
    recent_messages: Sequence[Message],
) -> str:
    """Plan section 10.4: product/system role, invariants, profile, checkpoint,
    pinned references, recent messages — in that order, always."""
    sections = [
        PRODUCT_ROLE,
        SCIENTIFIC_INVARIANTS,
        f"Capability profile for this turn: {capability_profile}. Only the tools "
        "listed by the MCP server for this connection exist; do not assume any other "
        "tool is available.",
        ANSWER_FORMAT,
        REQUIRED_LIMITATIONS_GUIDE,
    ]
    rendered_checkpoint = checkpoint.render()
    if rendered_checkpoint:
        sections.append(rendered_checkpoint)
    if pinned:
        sections.append("Pinned references:\n" + "\n".join(p.render() for p in pinned))
    rendered_recent = render_recent_messages(recent_messages)
    if rendered_recent:
        sections.append("Recent conversation:\n" + rendered_recent)
    return "\n\n".join(sections)

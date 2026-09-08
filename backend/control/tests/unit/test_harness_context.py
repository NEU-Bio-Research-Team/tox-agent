"""Context assembly is a contract, not a prompt-string accident."""
from __future__ import annotations

from datetime import datetime, timezone

from toxagent.domain.message import Message, PartType, Role
from toxagent.domain.session import Session
from toxagent.harness.context import (
    ANSWER_FORMAT,
    PRODUCT_ROLE,
    REQUIRED_LIMITATIONS_GUIDE,
    SCIENTIFIC_INVARIANTS,
    PinnedReference,
    SessionCheckpoint,
    build_system_prompt,
)


def test_context_prefix_has_the_plan_order_and_keeps_messages_as_a_projection():
    now = datetime(2026, 9, 4, tzinfo=timezone.utc)
    session = Session.create("user-1", now=now)
    prior_message = Message.create(
        session.id,
        Role.USER,
        1,
        now=now,
        parts=((PartType.TEXT, {"text": "Earlier report question"}),),
    )
    prompt = build_system_prompt(
        capability_profile="report_qa",
        checkpoint=SessionCheckpoint(summary="The user is reviewing one hERG result."),
        pinned=(
            PinnedReference(
                kind="analysis",
                id="ana_" + "a" * 32,
                summary="canonical SMILES=CCO; sections=herg",
            ),
        ),
        recent_messages=(prior_message,),
    )

    assert prompt.index(PRODUCT_ROLE) < prompt.index(SCIENTIFIC_INVARIANTS)
    assert prompt.index(SCIENTIFIC_INVARIANTS) < prompt.index("Capability profile")
    assert prompt.index("Capability profile") < prompt.index(ANSWER_FORMAT)
    assert prompt.index(ANSWER_FORMAT) < prompt.index(REQUIRED_LIMITATIONS_GUIDE)
    assert prompt.index(REQUIRED_LIMITATIONS_GUIDE) < prompt.index("Session checkpoint")
    assert prompt.index("Session checkpoint") < prompt.index("Pinned references")
    assert prompt.index("Pinned references") < prompt.index("Recent conversation")
    assert "User: Earlier report question" in prompt
    # The current user message is sent in RuntimeTurn, never duplicated into
    # this prefix by the context builder.
    assert "Current user message" not in prompt

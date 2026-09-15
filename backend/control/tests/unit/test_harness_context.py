"""Context assembly is a contract, not a prompt-string accident."""
from __future__ import annotations

from datetime import datetime, timezone

from toxagent.domain.message import Message, PartType, Role
from toxagent.domain.session import Session
from toxagent.harness.context import (
    ANSWER_FORMAT,
    DECISION_SUPPORT_POLICY,
    PRODUCT_ROLE,
    REQUIRED_LIMITATIONS_GUIDE,
    SCIENTIFIC_INVARIANTS,
    PinnedReference,
    SessionCheckpoint,
    build_system_prompt,
    render_recent_messages,
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


def test_decision_support_gets_the_search_policy_other_profiles_do_not():
    """ADR 0010 / W4: the sufficiency/search-trigger policy is only meaningful
    where the model decides whether to search at all; a closed profile like
    report_qa never had that choice, so it should not carry the policy."""
    common = dict(
        checkpoint=SessionCheckpoint(), pinned=(), recent_messages=(),
    )
    decision_support_prompt = build_system_prompt(
        capability_profile="decision_support", **common
    )
    report_qa_prompt = build_system_prompt(capability_profile="report_qa", **common)

    assert DECISION_SUPPORT_POLICY in decision_support_prompt
    assert DECISION_SUPPORT_POLICY not in report_qa_prompt
    assert decision_support_prompt.index(REQUIRED_LIMITATIONS_GUIDE) < (
        decision_support_prompt.index(DECISION_SUPPORT_POLICY)
    )


def test_typed_ref_parts_are_not_dropped_from_recent_history():
    """W3-02: answer_ref/report_ref parts carry the pointer a follow-up turn
    needs; before this, render_recent_messages only ever read PartType.TEXT
    and silently dropped every one of them."""
    now = datetime(2026, 9, 16, tzinfo=timezone.utc)
    session_id = "ses_" + "a" * 32
    answer_message = Message.create(
        session_id, Role.ASSISTANT, 2, now=now,
        parts=(
            (PartType.TEXT, {"text": "Benzene is non_blocker for hERG."}),
            (PartType.ANSWER_REF, {"answer_id": "ans_" + "b" * 32}),
        ),
    )
    report_message = Message.create(
        session_id, Role.ASSISTANT, 3, now=now,
        parts=(
            (PartType.TEXT, {"text": "Report completed."}),
            (PartType.REPORT_REF, {"report_id": "rpt_" + "c" * 32}),
        ),
    )

    rendered = render_recent_messages((answer_message, report_message))

    assert f"[answer_ref=ans_{'b' * 32}]" in rendered
    assert f"[report_ref=rpt_{'c' * 32}]" in rendered
    assert "Benzene is non_blocker for hERG." in rendered


def test_a_stale_analysis_ref_is_not_leaked_into_history_rendering():
    """audit_5_9.md A02: a no-longer-active analysis must not reappear in the
    prompt through any path, including recent-history rendering — only the
    dedicated, currently-scoped pinning in harness/gateway.py may name which
    analysis is active."""
    now = datetime(2026, 9, 16, tzinfo=timezone.utc)
    session_id = "ses_" + "a" * 32
    analysis_request = Message.create(
        session_id, Role.USER, 1, now=now,
        parts=((PartType.ANALYSIS_REF, {"smiles": "c1ccccc1"}),),
    )

    rendered = render_recent_messages((analysis_request,))

    assert "c1ccccc1" not in rendered

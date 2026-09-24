"""The durable copy of a run's request must survive a round trip intact.

If a field is dropped here, an adopted run executes a *different* request from
the one the user submitted — a worse outcome than the failure this replaced,
because it looks like it worked. The round-trip test therefore names every
field rather than checking a handful.
"""
from __future__ import annotations

import pytest

from toxagent.application.policy import Actor
from toxagent.application.run_envelope import (
    ENVELOPE_VERSION, UnreadableEnvelope, from_envelope, to_envelope,
)
from toxagent.application.run_scheduler import RunContext
from toxagent.domain.run import Intent

FULL = RunContext(
    actor=Actor(subject_id="user-7", roles=frozenset({"expert"})),
    session_id="ses_1",
    run_id="run_1",
    intent=Intent.ATTRIBUTION,
    text="phân tích phân tử này",
    smiles="CCO",
    batch_smiles=("CCO", "c1ccccc1"),
    endpoints=("herg", "tox21"),
    model_selection={"herg": "herg-tox21-chemberta-v1"},
    ai_profile_id="conn_9",
    threshold_overrides={"herg": 0.7},
    explanation_mode="required",
    explanation_targets=(("herg", None), ("tox21", "NR-AR")),
    analysis_id="ana_3",
    needs_snapshot_first=True,
    language="vi",
    attachment_id="att_2",
)


def test_every_field_survives_the_round_trip():
    restored = from_envelope(to_envelope(FULL))
    for field in FULL.__dataclass_fields__:
        assert getattr(restored, field) == getattr(FULL, field), field


def test_a_credential_is_never_written_into_the_envelope():
    """The profile is an opaque id; the gateway resolves the key at dispatch.

    A credential here would be a credential in the database, in every backup,
    and in anything that reads the job table for operational reasons.
    """
    envelope = to_envelope(FULL)
    assert envelope["ai_profile_id"] == "conn_9"
    flattened = repr(envelope)
    assert "credential" not in flattened
    assert "api_key" not in flattened


def test_a_missing_optional_field_decodes_to_the_default():
    """A rolling upgrade means an older writer's envelope reaching a newer
    reader. Every field it did not write must decode to what it meant."""
    minimal = {
        "v": ENVELOPE_VERSION,
        "actor": {"subject_id": "user-1"},
        "session_id": "ses_1",
        "run_id": "run_1",
        "intent": Intent.REPORT_QA.value,
    }
    context = from_envelope(minimal)
    assert context.text == ""
    assert context.smiles is None
    assert context.endpoints is None
    assert context.explanation_mode == "on_demand"
    assert context.explanation_targets == ()
    assert context.language == "en"


def test_an_unknown_key_is_ignored_rather_than_fatal():
    """The other direction of a rolling upgrade: a newer writer at the same
    envelope version adds a key this reader has never heard of."""
    envelope = to_envelope(FULL) | {"a_field_from_a_later_release": ["anything"]}
    assert from_envelope(envelope).run_id == FULL.run_id


def test_an_envelope_from_a_newer_version_is_refused_not_guessed_at():
    with pytest.raises(UnreadableEnvelope):
        from_envelope(to_envelope(FULL) | {"v": ENVELOPE_VERSION + 1})


def test_an_envelope_naming_an_intent_this_build_lacks_is_refused():
    with pytest.raises(UnreadableEnvelope):
        from_envelope(to_envelope(FULL) | {"intent": "telepathy"})


def test_an_envelope_with_no_actor_is_refused():
    """Executing as nobody would run someone's request without an owner to
    scope it to — worse than leaving it for a worker that can read it."""
    envelope = to_envelope(FULL)
    envelope["actor"] = {}
    with pytest.raises(UnreadableEnvelope):
        from_envelope(envelope)

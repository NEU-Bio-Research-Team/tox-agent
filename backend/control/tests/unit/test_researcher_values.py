"""Researcher-supplied values as citable versions (#61, §5.1 and §5.3).

The case the proposal turns on: the researcher gives a free Cmax, a margin is
computed from it, and then they correct the Cmax. What the old claim was
computed from has to stay knowable, and the number they typed must never read
as a measurement this product made.
"""
from datetime import datetime, timezone

import pytest

from toxagent.application.investigation.researcher_values import (
    USER_ASSERTED_LIMITATION,
    project_value,
)
from toxagent.domain import scientific_case as sc
from toxagent.domain.observation import ObservationKind, Producer
from toxagent.domain.scientific_case.model import Actor, InvalidCaseUpdate, ScientificCaseV1
from toxagent.domain.scientific_case.operations import verification_for
from toxagent.domain.scientific_case.values import (
    VerificationStatus,
    current_values,
    is_value_ref,
    iter_history,
    resolve_value_ref,
    value_versions,
)

NOW = datetime(2026, 10, 28, 9, 0, tzinfo=timezone.utc)
SESSION_ID = "ses_" + "1" * 32
CASE_ID = "scase_" + "2" * 32
RUN_ID = "run_" + "6" * 32


def _upd(op, actor=Actor.USER.value, **payload):
    return sc.CaseUpdate(op=op, payload=payload, actor=actor, at=NOW.isoformat(),
                         run_id=RUN_ID)


def _case() -> ScientificCaseV1:
    return sc.apply(None, _upd(
        "open", actor="server", case_id=CASE_ID, session_id=SESSION_ID,
        subject_key="analysis:ana_x", subject_refs=["analysis:ana_x"],
        question="Does the measurement contradict the prediction?",
    ))


def _state(case: ScientificCaseV1, key: str, value: str, *, unit: str | None = None,
           actor: str = Actor.USER.value, verification: str | None = None) -> ScientificCaseV1:
    payload = {"key": key, "value": value}
    if unit:
        payload["unit"] = unit
    if verification:
        payload["verification"] = verification
    return sc.apply(case, _upd("add_context", actor=actor, **payload))


# ── versions come from the log the case already keeps ──────────────────────

def test_a_corrected_value_becomes_a_second_version_and_the_first_remains():
    """§5.3: "Tạo observation version mới, giữ lịch sử.\""""
    case = _state(_case(), "free_cmax", "0.01", unit="µM")
    case = _state(case, "free_cmax", "0.02", unit="µM")

    history = list(iter_history(case, "free_cmax"))
    assert [row.version for row in history] == [1, 2]
    assert [row.value for row in history] == ["0.01", "0.02"]
    assert history[0].ref == "case_value:free_cmax@v1"
    assert history[1].ref == "case_value:free_cmax@v2"


def test_the_superseded_version_says_what_replaced_it_and_stays_citable():
    case = _state(_case(), "free_cmax", "0.01", unit="µM")
    case = _state(case, "free_cmax", "0.02", unit="µM")

    first, second = iter_history(case, "free_cmax")
    assert not first.current
    assert first.superseded_by == "case_value:free_cmax@v2"
    assert second.current and second.superseded_by is None
    # and the old version still resolves: a claim computed from it is not
    # retroactively about the new number
    assert resolve_value_ref(case, "case_value:free_cmax@v1").value == "0.01"


def test_versions_are_numbered_per_key():
    case = _state(_case(), "free_cmax", "0.01", unit="µM")
    case = _state(case, "ic50", "0.3", unit="µM")
    case = _state(case, "free_cmax", "0.02", unit="µM")

    assert [(r.key, r.version) for r in value_versions(case)] == [
        ("free_cmax", 1), ("ic50", 1), ("free_cmax", 2),
    ]
    assert {r.key: r.value for r in current_values(case)} == {
        "free_cmax": "0.02", "ic50": "0.3",
    }


def test_an_unresolvable_reference_is_a_finding_not_an_exception():
    """§5.3 asks for unresolved references to be detected, which needs them to
    come back as data."""
    assert resolve_value_ref(_case(), "case_value:free_cmax@v1") is None
    assert is_value_ref("case_value:free_cmax@v1")
    assert not is_value_ref("obs_123")


# ── standing: citable is not verified ─────────────────────────────────────

def test_a_value_with_no_stated_status_is_the_weakest_one():
    """Absent must not read as the strongest: that is how something typed
    becomes a verified measurement."""
    case = _state(_case(), "free_cmax", "0.01", unit="µM")

    assert current_values(case)[0].verification is VerificationStatus.USER_ASSERTED


def test_only_the_researcher_states_a_value_at_all():
    """``add_context`` is the researcher's operation, so a run cannot state a
    value in the case's name. This is the first of two locks."""
    with pytest.raises(InvalidCaseUpdate) as refusal:
        _state(_case(), "free_cmax", "0.01", unit="µM", actor=Actor.MODEL.value)

    assert "cannot be performed by the model" in str(refusal.value)


def test_the_second_lock_refuses_sme_verified_from_a_non_researcher_actor():
    """The allowlist above is about who may add context; this rule is about
    what standing a value may claim, and holds if that allowlist widens."""
    with pytest.raises(InvalidCaseUpdate) as refusal:
        verification_for(_upd("add_context", actor=Actor.SERVER.value, key="ic50",
                              value="0.3", verification="sme_verified"))

    assert "only be set by the researcher" in str(refusal.value)
    # the weaker two are fine from a non-researcher actor
    assert verification_for(_upd("add_context", actor=Actor.SERVER.value, key="ic50",
                                 value="0.3", verification="source_backed")) == "source_backed"


def test_the_researcher_may_state_sme_verified():
    case = _state(_case(), "ic50", "0.3", unit="µM", verification="sme_verified")

    assert current_values(case)[0].verification is VerificationStatus.SME_VERIFIED


def test_a_status_this_build_does_not_know_is_refused_at_the_door():
    with pytest.raises(InvalidCaseUpdate):
        _state(_case(), "ic50", "0.3", verification="peer_reviewed")


# ── the projection a claim cites ──────────────────────────────────────────

def _project(case: ScientificCaseV1, key: str = "free_cmax"):
    version = next(r for r in current_values(case) if r.key == key)
    return project_value(version, case_id=case.id, session_id=SESSION_ID,
                         run_id=RUN_ID, now=NOW)


def test_the_projection_is_a_researcher_observation_not_a_prediction():
    observation = _project(_state(_case(), "free_cmax", "0.01", unit="µM"))

    assert observation.producer is Producer.RESEARCHER
    assert observation.kind is ObservationKind.RESEARCHER_VALUE
    assert "probability" not in observation.canonical_payload


def test_the_projection_carries_the_stable_version_reference():
    """A claim records which exposure it was computed from, by a reference
    that resolves the same way in a later run -- an observation id would not."""
    observation = _project(_state(_case(), "free_cmax", "0.01", unit="µM"))

    assert observation.canonical_payload["ref"] == "case_value:free_cmax@v1"
    assert observation.provenance["value_ref"] == "case_value:free_cmax@v1"
    assert observation.provenance["context_item_id"] == "c1"


def test_an_unverified_value_requires_the_limitation_that_says_so():
    observation = _project(_state(_case(), "free_cmax", "0.01", unit="µM"))

    assert observation.required_limitations == (USER_ASSERTED_LIMITATION,)
    assert "not verified against any source" in observation.model_projection["reading"]


def test_an_sme_verified_value_does_not_carry_the_unverified_limitation():
    case = _state(_case(), "ic50", "0.3", unit="µM", verification="sme_verified")

    observation = _project(case, key="ic50")

    assert observation.required_limitations == ()


def test_the_value_is_canonicalised_with_its_conversion_recorded():
    """#59's rule reaches the researcher's own number: a µM exposure and a nM
    record have to meet somewhere, and the step must be checkable."""
    observation = _project(_state(_case(), "free_cmax", "0.01", unit="µM"))

    constraint = observation.canonical_payload["constraint"]
    assert constraint["canonical"]["low"] == "10"
    assert constraint["canonical"]["unit"] == "nM"
    assert constraint["conversion"]["factor"] == "1000"
    # the raw form the answer quotes is untouched
    assert observation.canonical_payload["value"] == "0.01"
    assert observation.canonical_payload["unit"] == "µM"


def test_a_value_in_a_unit_this_domain_cannot_place_is_reported_as_such():
    observation = _project(_state(_case(), "free_cmax", "0.01", unit="mg/mL"))

    assert observation.canonical_payload["constraint"]["kind"] == "unknown_unit"
    assert observation.canonical_payload["constraint"]["conversion"] is None


def test_the_same_version_projects_to_the_same_content_hash():
    """What lets a replayed case produce comparable claims across runs."""
    case = _state(_case(), "free_cmax", "0.01", unit="µM")

    first, second = _project(case), _project(case)

    assert first.content_sha256 == second.content_sha256
    assert first.id != second.id  # the id is minted per run; the content is not


def test_a_superseded_version_says_so_to_the_model():
    case = _state(_case(), "free_cmax", "0.01", unit="µM")
    case = _state(case, "free_cmax", "0.02", unit="µM")
    old = next(r for r in iter_history(case, "free_cmax") if r.version == 1)

    observation = project_value(old, case_id=case.id, session_id=SESSION_ID,
                                run_id=RUN_ID, now=NOW)

    assert observation.canonical_payload["superseded_by"] == "case_value:free_cmax@v2"
    assert not observation.model_projection["current"]
    assert "A later version of this value exists" in observation.model_projection["reading"]

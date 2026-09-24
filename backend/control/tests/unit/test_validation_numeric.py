"""Numeric claim validation (plan section 9.1)."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from toxagent.domain.ids import new_id
from toxagent.domain.observation import Observation, ObservationKind, Producer
from toxagent.validation.numeric import (
    parse_rendered_number,
    round_tolerance,
    validate_derived_numeric,
    validate_field_backed_numeric,
)
from toxagent.validation.wire import ClaimCandidate

NOW = datetime(2026, 9, 4, tzinfo=timezone.utc)


def an_observation(payload=None) -> Observation:
    return Observation.create(
        session_id=new_id("ses"), run_id=new_id("run"), producer=Producer.PREDICTOR,
        kind=ObservationKind.PREDICTION, schema_version="v1",
        canonical_payload=payload or {"predictions": {"herg": {"probability_blocker": 0.73064}}},
        model_projection={}, provenance={}, now=NOW,
    )


def claim(**overrides) -> ClaimCandidate:
    defaults = dict(
        claim_id=new_id("clm"), kind="numeric", text="x",
        observation_id=None, field_path="predictions.herg.probability_blocker",
        source_value=0.73064, rendered_value="0.731", transform="round:3",
    )
    defaults.update(overrides)
    return ClaimCandidate(**defaults)


# --- parsing -----------------------------------------------------------

@pytest.mark.parametrize(
    "text,expected",
    [
        ("0.731", 0.731), ("0,731", 0.731), ("73.1%", 73.1), ("73,1%", 73.1),
        ("-2.5", -2.5), ("42", 42.0), ("0", 0.0),
    ],
)
def test_parse_rendered_number_accepts_a_single_dot_or_vietnamese_comma_number(text, expected):
    assert parse_rendered_number(text) == pytest.approx(expected)


def test_parse_rendered_number_rejects_garbage():
    with pytest.raises(ValueError):
        parse_rendered_number("not a number")


@pytest.mark.parametrize(
    "text",
    [
        "0,0315 (3,15%)",   # the compound Vietnamese render from the Phase 3 live run
        "0.0315 (3.15%)",
        "0.731 probability",
        "73.1 %",           # a space before the percent sign is not canonical
        "~0.73",
        "1.2e-3",
    ],
)
def test_parse_rendered_number_rejects_a_compound_or_annotated_render(text):
    """ADR 0005: rendered_value is one number the transform produced, not a
    display phrase. The rejection names the fix instead of only 'unparseable'."""
    with pytest.raises(ValueError) as excinfo:
        parse_rendered_number(text)
    message = str(excinfo.value)
    assert "single number" in message
    assert "text" in message  # tells the model where the phrasing belongs


def test_a_compound_rendered_value_is_a_correctable_violation_not_a_crash():
    observation = an_observation()
    result = validate_field_backed_numeric(
        claim(observation_id=observation.id, rendered_value="0,0315 (3,15%)"), observation
    )
    assert [v.code for v in result] == ["claim_rendered_value_unparseable"]
    assert "single number" in result[0].message


def test_round_tolerance_matches_the_plan_formula():
    assert round_tolerance(3) == pytest.approx(0.5 * 10**-3 + 1e-12)


# --- field-backed numeric ------------------------------------------------

def test_a_correct_rounded_claim_has_no_violations():
    observation = an_observation()
    result = validate_field_backed_numeric(
        claim(observation_id=observation.id), observation
    )
    assert result == []


def test_a_missing_observation_is_one_violation():
    result = validate_field_backed_numeric(claim(observation_id=new_id("obs")), None)
    assert [v.code for v in result] == ["claim_observation_not_found"]


def test_an_unresolvable_field_path_is_reported():
    observation = an_observation()
    result = validate_field_backed_numeric(
        claim(observation_id=observation.id, field_path="predictions.clintox.probability_clinical_toxicity"),
        observation,
    )
    assert [v.code for v in result] == ["claim_field_path_unresolvable"]


def test_a_wrong_source_value_is_a_mismatch():
    observation = an_observation()
    result = validate_field_backed_numeric(
        claim(observation_id=observation.id, source_value=0.5), observation
    )
    assert "claim_source_value_mismatch" in [v.code for v in result]


def test_a_rendered_value_outside_tolerance_is_rejected():
    observation = an_observation()
    result = validate_field_backed_numeric(
        claim(observation_id=observation.id, rendered_value="0.800"), observation
    )
    assert "claim_rendered_value_mismatch" in [v.code for v in result]


def test_a_rendered_value_at_the_tolerance_boundary_passes():
    observation = an_observation({"predictions": {"herg": {"probability_blocker": 0.7305}}})
    result = validate_field_backed_numeric(
        claim(observation_id=observation.id, source_value=0.7305, rendered_value="0.730"), observation
    )
    assert result == []


def test_percent_transform_multiplies_by_exactly_100():
    observation = an_observation()
    result = validate_field_backed_numeric(
        claim(
            observation_id=observation.id, rendered_value="73.064", transform="percent:3",
        ),
        observation,
    )
    assert result == []


def test_percent_transform_rejects_a_value_that_forgot_to_multiply():
    observation = an_observation()
    result = validate_field_backed_numeric(
        claim(observation_id=observation.id, rendered_value="0.731", transform="percent:3"),
        observation,
    )
    assert "claim_rendered_value_mismatch" in [v.code for v in result]


def test_identity_requires_an_exact_render():
    observation = an_observation()
    result = validate_field_backed_numeric(
        claim(
            observation_id=observation.id, transform="identity",
            source_value=0.73064, rendered_value="0.73064",
        ),
        observation,
    )
    assert result == []


def test_a_numeric_claim_over_a_non_numeric_field_is_refused():
    observation = an_observation({"predictions": {"herg": {"label": "blocker"}}})
    result = validate_field_backed_numeric(
        claim(observation_id=observation.id, field_path="predictions.herg.label"), observation
    )
    assert [v.code for v in result] == ["claim_field_not_numeric"]


# --- derived (difference/ratio) ------------------------------------------

def test_a_correct_difference_claim_passes():
    a = claim(claim_id="clm_" + "a" * 32, source_value=0.8, rendered_value="0.8")
    b = claim(claim_id="clm_" + "b" * 32, source_value=0.3, rendered_value="0.3")
    d = claim(
        claim_id="clm_" + "c" * 32, kind="comparison", transform="difference",
        observation_id=None, field_path=None, source_value=0.5, rendered_value="0.5",
        input_claim_ids=[a.claim_id, b.claim_id],
    )
    by_id = {a.claim_id: a, b.claim_id: b}
    assert validate_derived_numeric(d, by_id) == []


def test_a_ratio_by_zero_is_refused_not_computed():
    a = claim(claim_id="clm_" + "a" * 32, source_value=0.8)
    zero = claim(claim_id="clm_" + "b" * 32, source_value=0.0)
    ratio = claim(
        claim_id="clm_" + "c" * 32, kind="comparison", transform="ratio",
        observation_id=None, field_path=None, source_value=999, rendered_value="999",
        input_claim_ids=[a.claim_id, zero.claim_id],
    )
    result = validate_derived_numeric(ratio, {a.claim_id: a, zero.claim_id: zero})
    assert [v.code for v in result] == ["claim_derived_division_by_zero"]


def test_a_derived_claim_with_a_missing_input_is_refused():
    a = claim(claim_id="clm_" + "a" * 32, source_value=0.8)
    d = claim(
        claim_id="clm_" + "c" * 32, kind="comparison", transform="difference",
        observation_id=None, field_path=None, source_value=0.5,
        input_claim_ids=[a.claim_id, "clm_" + "z" * 32],
    )
    result = validate_derived_numeric(d, {a.claim_id: a})
    assert [v.code for v in result] == ["claim_derived_input_missing"]


def test_a_derived_claim_with_the_wrong_arity_is_refused():
    a = claim(claim_id="clm_" + "a" * 32, source_value=0.8)
    d = claim(
        claim_id="clm_" + "c" * 32, kind="comparison", transform="difference",
        observation_id=None, field_path=None, source_value=0.5,
        input_claim_ids=[a.claim_id],
    )
    result = validate_derived_numeric(d, {a.claim_id: a})
    assert [v.code for v in result] == ["claim_derived_inputs_invalid"]

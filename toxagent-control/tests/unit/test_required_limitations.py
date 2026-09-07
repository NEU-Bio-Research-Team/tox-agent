"""Deriving required limitations from what a candidate actually claims.

Plan section 9.4's trigger table is per-claim, not per-observation: a claim
that only cites a probability field must not be forced to disclose an
applicability caveat it never touched, even though the observation it cites
also happens to carry applicability data.
"""
from __future__ import annotations

from toxagent.domain.ids import new_id
from toxagent.validation.limitations import required_for_answer
from toxagent.validation.wire import ClaimCandidate

OBS = new_id("obs")


def claim(field_path: str, **overrides) -> ClaimCandidate:
    defaults = dict(
        claim_id=new_id("clm"), kind="numeric", text="x", observation_id=OBS,
        field_path=field_path, source_value=0.1, rendered_value="0.1", transform="identity",
    )
    defaults.update(overrides)
    return ClaimCandidate(**defaults)


def test_a_probability_claim_requires_the_uncalibrated_disclosure():
    required = required_for_answer([claim("predictions.herg.probability_blocker")])
    assert required == {"uncalibrated_probability"}


def test_a_probability_claim_does_not_pull_in_an_unrelated_applicability_caveat():
    """The regression this guards: the observation carries
    applicability_is_rule_based in its own required_limitations (because the
    analysis has an applicability section at all), but a claim that never
    mentions applicability must not be forced to declare it."""
    required = required_for_answer(
        [claim("predictions.herg.probability_blocker")],
        observation_limitations={OBS: ("uncalibrated_probability", "applicability_is_rule_based")},
    )
    assert "applicability_is_rule_based" not in required


def test_a_claim_that_actually_names_applicability_does_require_it():
    required = required_for_answer([claim("applicability.status", kind="classification")])
    assert required == {"applicability_is_rule_based"}


def test_endpoint_unavailable_is_observation_wide_not_field_specific():
    """Unlike applicability, an unavailable endpoint is a property of the whole
    analysis: any claim built from that observation should disclose it."""
    required = required_for_answer(
        [claim("predictions.herg.probability_blocker")],
        observation_limitations={OBS: ("endpoint_unavailable",)},
    )
    assert "endpoint_unavailable" in required


def test_attribution_not_causality_is_observation_wide_too():
    """The regression this guards (live sweep 2026-09-06, progress log section
    14.4): a scientific claim citing an attribution observation by
    observation_id alone, with no field_path — legal, since SCIENTIFIC is not
    a FIELD_BACKED_KIND — used to escape the requirement entirely, because the
    only path to it was a `"attribution" in path or "tokens" in path` substring
    check against a field_path that need not exist. An attribution observation
    is entirely about attribution, unlike an analysis observation mixing
    several fields, so this is observation-wide exactly like
    endpoint_unavailable, not field-triggered like applicability."""
    required = required_for_answer(
        [claim(None, kind="scientific")],
        observation_limitations={OBS: ("attribution_not_causality",)},
    )
    assert "attribution_not_causality" in required


def test_a_classification_claim_over_a_label_needs_no_probability_caveat():
    required = required_for_answer([claim("predictions.herg.label", kind="classification")])
    assert required == set()


def test_citing_evidence_requires_the_scope_limitation():
    required = required_for_answer([], cited_evidence=True)
    assert required == {"evidence_scope_limited"}


def test_a_recommendation_requires_the_screening_disclaimer():
    required = required_for_answer([], has_recommendation=True)
    assert required == {"screening_not_safety_assessment"}

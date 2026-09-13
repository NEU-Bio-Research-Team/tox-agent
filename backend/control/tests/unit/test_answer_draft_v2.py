"""P1-4/P1-5: the model stops minting ids and stops restating numbers.

The audit found a model asked to generate 32 random hex characters, to copy a
canonical value it could read from an observation, and to render that value
under a locale convention. All three are server work. A live run had already
died on the first one — a good answer refused for id length, the one correction
attempt spent re-minting ids, the length wrong again, no answer at all.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from toxagent.validation.claim_resolver import (
    derived_value,
    render_number,
    resolve_draft,
)
from toxagent.validation.wire import CLAIM_ID_PATTERN
from toxagent.validation.wire_v2 import GroundedAnswerDraftV2

OBS = "obs_" + "a" * 32


class _Observation:
    """Just enough of the domain type for the resolver: a field lookup that
    raises on an unknown path, exactly as the real one does."""

    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def value_at(self, path: str):
        node = self._payload
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                raise KeyError(f"{path!r} is not a field of this observation")
            node = node[part]
        return node


OBSERVATIONS = {
    OBS: _Observation(
        {
            "predictions": {
                "herg": {
                    "probability_blocker": 0.7312,
                    "classification": "blocker",
                    "model_id": "herg-chemberta-v3",
                },
                "clintox": {"probability_toxic": 0.2},
            }
        }
    )
}


def _draft(**overrides) -> GroundedAnswerDraftV2:
    payload = {
        "answer_markdown": "The hERG blocker probability is 0.731.",
        "claims": [
            {
                "local_ref": "herg_p",
                "kind": "numeric",
                "text": "hERG blocker probability is 0.731",
                "observation_id": OBS,
                "field_path": "predictions.herg.probability_blocker",
                "transform": "round:3",
            }
        ],
    }
    payload.update(overrides)
    return GroundedAnswerDraftV2(**payload)


# --- the server issues identity ---------------------------------------------


def test_the_server_issues_the_claim_id() -> None:
    resolved = resolve_draft(_draft(), observations_by_id=OBSERVATIONS)
    assert resolved.ok
    claim = resolved.candidate.claims[0]
    assert CLAIM_ID_PATTERN.match(claim.claim_id)
    assert resolved.issued_ids["herg_p"] == claim.claim_id


def test_the_same_local_ref_twice_gets_two_different_global_ids() -> None:
    """Two runs may both call their claim 'claim_1'. That must not collide."""
    first = resolve_draft(_draft(), observations_by_id=OBSERVATIONS)
    second = resolve_draft(_draft(), observations_by_id=OBSERVATIONS)
    assert first.candidate.claims[0].claim_id != second.candidate.claims[0].claim_id
    for resolved in (first, second):
        assert CLAIM_ID_PATTERN.match(resolved.candidate.claims[0].claim_id)


def test_a_draft_cannot_carry_a_claim_id_at_all() -> None:
    with pytest.raises(ValidationError):
        GroundedAnswerDraftV2(
            answer_markdown="x",
            claims=[
                {
                    "local_ref": "a",
                    "claim_id": "clm_" + "0" * 32,
                    "kind": "scientific",
                    "text": "x",
                }
            ],
        )


@pytest.mark.parametrize("bad", ["", "Claim_1", "1claim", "a" * 33, "claim-1", "claim 1"])
def test_a_local_ref_has_to_be_a_label(bad: str) -> None:
    with pytest.raises(ValidationError):
        GroundedAnswerDraftV2(
            answer_markdown="x",
            claims=[{"local_ref": bad, "kind": "scientific", "text": "x"}],
        )


def test_duplicate_local_refs_are_refused_before_anything_resolves() -> None:
    with pytest.raises(ValidationError, match="repeated"):
        GroundedAnswerDraftV2(
            answer_markdown="x",
            claims=[
                {"local_ref": "a", "kind": "scientific", "text": "one"},
                {"local_ref": "a", "kind": "scientific", "text": "two"},
            ],
        )


# --- the server reads the value ---------------------------------------------


def test_a_draft_cannot_carry_a_source_value() -> None:
    with pytest.raises(ValidationError):
        GroundedAnswerDraftV2(
            answer_markdown="x",
            claims=[
                {
                    "local_ref": "a",
                    "kind": "numeric",
                    "text": "x",
                    "observation_id": OBS,
                    "field_path": "predictions.herg.probability_blocker",
                    "source_value": 0.999,
                }
            ],
        )


def test_the_value_comes_from_the_observation() -> None:
    resolved = resolve_draft(_draft(), observations_by_id=OBSERVATIONS)
    claim = resolved.candidate.claims[0]
    assert claim.source_value == 0.7312
    assert claim.rendered_value == "0.731"


def test_a_classification_value_is_read_not_stated() -> None:
    draft = _draft(
        claims=[
            {
                "local_ref": "cls",
                "kind": "classification",
                "text": "classified as a blocker",
                "observation_id": OBS,
                "field_path": "predictions.herg.classification",
            }
        ]
    )
    claim = resolve_draft(draft, observations_by_id=OBSERVATIONS).candidate.claims[0]
    assert claim.source_value == "blocker"
    assert claim.rendered_value == "blocker"


def test_a_field_backed_claim_must_name_a_field() -> None:
    with pytest.raises(ValidationError, match="field_path"):
        GroundedAnswerDraftV2(
            answer_markdown="x",
            claims=[
                {"local_ref": "a", "kind": "numeric", "text": "x", "observation_id": OBS}
            ],
        )


def test_an_unresolvable_field_path_is_a_correctable_violation() -> None:
    draft = _draft(
        claims=[
            {
                "local_ref": "a",
                "kind": "numeric",
                "text": "x",
                "observation_id": OBS,
                "field_path": "predictions.herg.no_such_field",
            }
        ]
    )
    resolved = resolve_draft(draft, observations_by_id=OBSERVATIONS)
    assert not resolved.ok
    assert [v.code for v in resolved.violations] == ["claim_field_path_unresolvable"]
    assert resolved.violations[0].path == "claims[0].field_path"


def test_an_observation_the_run_never_read_is_a_violation() -> None:
    draft = _draft(
        claims=[
            {
                "local_ref": "a",
                "kind": "numeric",
                "text": "x",
                "observation_id": "obs_" + "b" * 32,
                "field_path": "predictions.herg.probability_blocker",
            }
        ]
    )
    resolved = resolve_draft(draft, observations_by_id=OBSERVATIONS)
    assert [v.code for v in resolved.violations] == ["claim_observation_not_found"]


def test_a_non_numeric_field_on_a_numeric_claim_is_a_violation() -> None:
    draft = _draft(
        claims=[
            {
                "local_ref": "a",
                "kind": "numeric",
                "text": "x",
                "observation_id": OBS,
                "field_path": "predictions.herg.model_id",
            }
        ]
    )
    resolved = resolve_draft(draft, observations_by_id=OBSERVATIONS)
    assert [v.code for v in resolved.violations] == ["claim_field_not_numeric"]


# --- the server renders ------------------------------------------------------


@pytest.mark.parametrize(
    "value,transform,language,expected",
    [
        (0.7312, "round:3", "en", "0.731"),
        (0.7312, "round:3", "vi", "0,731"),
        (0.0315, "percent:2", "en", "3.15%"),
        (0.0315, "percent:2", "vi", "3,15%"),
        (0.7312, "identity", "en", "0.7312"),
        (0.7312, "identity", "vi", "0,7312"),
        (2.0, "identity", "en", "2"),
        (0.5, "round:0", "en", "0"),
    ],
)
def test_rendering_is_a_server_rule_not_a_prompt_instruction(
    value, transform, language, expected
) -> None:
    assert render_number(value, transform, language=language) == expected


def test_a_rendered_value_never_carries_display_phrasing() -> None:
    """ADR 0005: rendered_value is the number; the prose carries the phrasing."""
    rendered = render_number(0.0315, "percent:2", language="vi")
    assert "(" not in rendered and " " not in rendered


# --- comparisons are computed ------------------------------------------------


def test_a_comparison_is_computed_from_its_inputs() -> None:
    draft = _draft(
        answer_markdown="hERG is 0.5312 higher than ClinTox.",
        claims=[
            {
                "local_ref": "herg_p",
                "kind": "numeric",
                "text": "hERG",
                "observation_id": OBS,
                "field_path": "predictions.herg.probability_blocker",
            },
            {
                "local_ref": "clintox_p",
                "kind": "numeric",
                "text": "ClinTox",
                "observation_id": OBS,
                "field_path": "predictions.clintox.probability_toxic",
            },
            {
                "local_ref": "gap",
                "kind": "comparison",
                "text": "the difference",
                "transform": "difference",
                "input_local_refs": ["herg_p", "clintox_p"],
            },
        ],
    )
    resolved = resolve_draft(draft, observations_by_id=OBSERVATIONS)
    assert resolved.ok
    comparison = resolved.candidate.claims[2]
    assert comparison.source_value == pytest.approx(0.5312)
    assert comparison.rendered_value == "0.5312"
    # And it points at the ids the server issued, not at labels.
    assert comparison.input_claim_ids == [
        resolved.issued_ids["herg_p"],
        resolved.issued_ids["clintox_p"],
    ]


def test_a_comparison_may_name_inputs_that_appear_after_it() -> None:
    draft = _draft(
        answer_markdown="x",
        claims=[
            {
                "local_ref": "gap",
                "kind": "comparison",
                "text": "the ratio",
                "transform": "ratio",
                "input_local_refs": ["herg_p", "clintox_p"],
            },
            {
                "local_ref": "herg_p",
                "kind": "numeric",
                "text": "hERG",
                "observation_id": OBS,
                "field_path": "predictions.herg.probability_blocker",
            },
            {
                "local_ref": "clintox_p",
                "kind": "numeric",
                "text": "ClinTox",
                "observation_id": OBS,
                "field_path": "predictions.clintox.probability_toxic",
            },
        ],
    )
    resolved = resolve_draft(draft, observations_by_id=OBSERVATIONS)
    assert resolved.ok
    assert resolved.candidate.claims[0].source_value == pytest.approx(0.7312 / 0.2)


def test_a_comparison_naming_an_unknown_ref_is_refused_at_the_schema() -> None:
    with pytest.raises(ValidationError, match="input_local_refs"):
        GroundedAnswerDraftV2(
            answer_markdown="x",
            claims=[
                {
                    "local_ref": "gap",
                    "kind": "comparison",
                    "text": "x",
                    "transform": "difference",
                    "input_local_refs": ["a", "b"],
                }
            ],
        )


def test_a_claim_cannot_compare_itself() -> None:
    with pytest.raises(ValidationError, match="its own input"):
        GroundedAnswerDraftV2(
            answer_markdown="x",
            claims=[
                {
                    "local_ref": "a",
                    "kind": "comparison",
                    "text": "x",
                    "transform": "difference",
                    "input_local_refs": ["a"],
                }
            ],
        )


def test_a_ratio_by_zero_is_a_violation_not_an_infinity() -> None:
    assert derived_value("ratio", 1.0, 0.0) is None
    assert derived_value("difference", 1.0, 0.25) == pytest.approx(0.75)
    assert derived_value("difference", "x", 1.0) is None


# --- recommendations ---------------------------------------------------------


def test_a_recommendation_basis_is_rewritten_to_issued_ids() -> None:
    draft = _draft(
        recommended_next_steps=[
            {"text": "Confirm in vitro.", "basis_local_refs": ["herg_p"]}
        ]
    )
    resolved = resolve_draft(draft, observations_by_id=OBSERVATIONS)
    step = resolved.candidate.recommended_next_steps[0]
    assert step.basis_claim_ids == [resolved.issued_ids["herg_p"]]


def test_a_recommendation_naming_an_unknown_ref_is_refused() -> None:
    with pytest.raises(ValidationError, match="basis_local_refs"):
        GroundedAnswerDraftV2(
            answer_markdown="x",
            claims=[{"local_ref": "a", "kind": "scientific", "text": "x"}],
            recommended_next_steps=[{"text": "y", "basis_local_refs": ["nope"]}],
        )


# --- the tool surface --------------------------------------------------------


def test_the_v2_tool_description_carries_no_example_identifier() -> None:
    """Examples in a prompt are instructions to imitate. The old description
    carried a full, valid-looking claim id, and it was imitated."""
    from toxagent.tools.definitions.answer import _V2_DESCRIPTION

    assert "clm_" not in _V2_DESCRIPTION
    assert "32" not in _V2_DESCRIPTION
    assert "hex" not in _V2_DESCRIPTION
    assert "local_ref" in _V2_DESCRIPTION


def test_the_v2_tool_does_not_ask_the_model_for_a_number() -> None:
    from toxagent.tools.definitions.answer import _V2_DESCRIPTION

    assert "do not send the number" in _V2_DESCRIPTION
    assert "rendered_value" not in _V2_DESCRIPTION

"""P1-6: say how much of the attribution never reached the molecule.

The audit's CCO/hERG explanation put 35.84% of its importance on ``[CLS]`` and
``[SEP]`` and was reported as having no unmapped mass. These tests run the
captured payload and pin the accounting, the wording, and the one thing that
must never happen: renormalizing the special tokens away.
"""
from __future__ import annotations

import pytest

from tests.support.audit_fixtures import CCO_ATTRIBUTION, load
from toxagent.application.explanation import extract_highlights
from toxagent.application.xai_coverage import (
    COVERAGE_POLICY_VERSION,
    ExplanationCoverage,
    classify_coverage,
    compute_coverage,
    is_special_token,
    mapped_only_relative_importance,
    rank_atoms_by_absolute_total,
)

FIXTURE = load(CCO_ATTRIBUTION)
PAYLOAD = FIXTURE["payload"]
EXPECTED = FIXTURE["expected_coverage"]


# --- the finding ------------------------------------------------------------


def test_the_audit_payload_reports_its_special_token_mass() -> None:
    coverage = compute_coverage(PAYLOAD, negative_contributor_count=3)
    assert coverage.special_token_importance_fraction == pytest.approx(
        EXPECTED["special_token_importance_fraction"], abs=1e-9
    )
    assert coverage.mapped_importance_fraction == pytest.approx(
        EXPECTED["mapped_importance_fraction"], abs=1e-9
    )
    assert coverage.other_unmapped_importance_fraction == pytest.approx(0.0, abs=1e-9)
    assert coverage.coverage_status == EXPECTED["coverage_status"]


def test_the_three_fractions_account_for_the_whole_mass() -> None:
    coverage = compute_coverage(PAYLOAD)
    total = (
        coverage.mapped_importance_fraction
        + coverage.special_token_importance_fraction
        + coverage.other_unmapped_importance_fraction
    )
    assert total == pytest.approx(1.0, abs=1e-9)


def test_the_summary_sentence_names_the_markers_and_the_contributors() -> None:
    highlights = extract_highlights(PAYLOAD)
    coverage = ExplanationCoverage(**{
        k: v for k, v in (
            ("mapped_importance_fraction", highlights.coverage["mapped_importance_fraction"]),
            ("special_token_importance_fraction",
             highlights.coverage["special_token_importance_fraction"]),
            ("other_unmapped_importance_fraction",
             highlights.coverage["other_unmapped_importance_fraction"]),
            ("unmapped_importance", highlights.coverage["unmapped_importance"]),
            ("coverage_status", highlights.coverage["coverage_status"]),
            ("positive_contributor_count", highlights.coverage["positive_contributor_count"]),
            ("negative_contributor_count", highlights.coverage["negative_contributor_count"]),
        )
    })
    sentence = coverage.summary_sentence()
    assert "35.8%" in sentence
    assert "sequence markers" in sentence
    assert "3 contributor(s)" in sentence
    assert "0 positive, 3 negative" in sentence


def test_extract_highlights_carries_the_accounting() -> None:
    highlights = extract_highlights(PAYLOAD)
    assert highlights.contributor_count == 3
    assert len(highlights.negative_contributors) == 3
    assert highlights.positive_contributors == ()
    assert highlights.coverage_status == "limited"
    assert highlights.coverage["coverage_policy_version"] == COVERAGE_POLICY_VERSION
    assert highlights.unmapped_importance == pytest.approx(
        EXPECTED["unmapped_importance"], abs=1e-12
    )


def test_nothing_renormalizes_the_special_tokens_away() -> None:
    """The provenance number stays raw. The drawing view is a separate,
    explicitly named field."""
    coverage = compute_coverage(PAYLOAD)
    assert coverage.mapped_importance_fraction < 1.0
    drawn = mapped_only_relative_importance(PAYLOAD["atoms"])
    assert sum(atom["mapped_only_relative_importance"] for atom in drawn) == pytest.approx(1.0)
    # And the two views are not the same number.
    assert drawn[0]["mapped_only_relative_importance"] != pytest.approx(
        drawn[0]["relative_importance"]
    )


# --- edges ------------------------------------------------------------------


def test_an_unreported_unmapped_mass_is_unknown_not_zero() -> None:
    coverage = compute_coverage({"atoms": [], "tokens": []})
    assert coverage.unmapped_importance is None
    assert coverage.mapped_importance_fraction is None
    assert coverage.coverage_status == "unknown"
    assert not coverage.is_known
    assert "unknown" in coverage.summary_sentence()


def test_a_fully_mapped_attribution_is_high_coverage() -> None:
    coverage = compute_coverage(
        {"unmapped_importance": 0.0, "tokens": [{"token": "C", "importance": 1.0}]}
    )
    assert coverage.unmapped_importance == 0.0
    assert coverage.coverage_status == "high"
    assert coverage.special_token_importance_fraction == 0.0


def test_unmapped_mass_with_no_token_list_says_neither() -> None:
    """Attributing it all to markers, or none of it, are both guesses."""
    coverage = compute_coverage({"unmapped_importance": 0.4})
    assert coverage.unmapped_importance == 0.4
    assert coverage.special_token_importance_fraction is None
    assert coverage.other_unmapped_importance_fraction is None
    assert coverage.coverage_status == "limited"


def test_structural_characters_are_not_called_sequence_markers() -> None:
    payload = {
        "unmapped_importance": 0.4,
        "tokens": [
            {"token": "[CLS]", "importance": 0.1},
            {"token": "1", "importance": 0.3},
            {"token": "C", "importance": 0.6},
        ],
    }
    coverage = compute_coverage(payload)
    assert coverage.special_token_importance_fraction == pytest.approx(0.1)
    assert coverage.other_unmapped_importance_fraction == pytest.approx(0.3)


@pytest.mark.parametrize(
    "token,expected",
    [
        ("[CLS]", True),
        ("[sep]", True),
        ("<s>", True),
        ("[C@H]", False),
        ("[nH]", False),
        ("C", False),
        (None, False),
        (12, False),
    ],
)
def test_an_atom_token_in_brackets_is_not_a_marker(token, expected) -> None:
    assert is_special_token(token) is expected


@pytest.mark.parametrize(
    "fraction,status",
    [(None, "unknown"), (0.0, "low"), (0.49, "low"), (0.5, "limited"),
     (0.79, "limited"), (0.8, "high"), (1.0, "high")],
)
def test_coverage_bands(fraction, status) -> None:
    assert classify_coverage(fraction) == status


def test_ranking_uses_magnitude_so_a_strong_negative_is_not_last() -> None:
    atoms = [
        {"atom_index": 0, "importance": 0.1, "signed_contribution": 0.1},
        {"atom_index": 1, "importance": 0.9, "signed_contribution": -0.9},
        {"atom_index": 2, "importance": 0.5, "signed_contribution": 0.5},
    ]
    assert [atom["atom_index"] for atom in rank_atoms_by_absolute_total(atoms)] == [1, 2, 0]


def test_a_zero_mass_structure_does_not_divide_by_zero() -> None:
    drawn = mapped_only_relative_importance(
        [{"atom_index": 0, "importance": 0.0}, {"atom_index": 1, "importance": 0.0}]
    )
    assert [atom["mapped_only_relative_importance"] for atom in drawn] == [0.0, 0.0]

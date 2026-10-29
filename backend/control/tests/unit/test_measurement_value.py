"""The representation and numeric-constraint layers of §5.2 (#59).

Each test here is one of the ways the old single-string comparison got a
scientific question wrong: a unit difference read as two measurements, a
censored bound read as a point, or two numbers read as a contradiction.
"""
from decimal import Decimal

import pytest

from toxagent.domain.measurement_value import (
    CONCENTRATION,
    Agreement,
    ConstraintKind,
    NumericConstraint,
    Verdict,
    canonicalise,
    same_dimension,
    unit_dimension,
    units_equivalent,
)

# ── representation: one value, two spellings ──────────────────────────────


def test_micromolar_and_nanomolar_are_the_same_value():
    """0,3 µM and 300 nM are one measurement written twice (§13 H1)."""
    micro = NumericConstraint.parse(0.3, "µM")
    nano = NumericConstraint.parse(300, "nM")

    assert micro.equivalent_to(nano)
    assert micro.low == nano.low == Decimal("300")


def test_the_conversion_used_is_recorded_not_just_the_result():
    """A converted number whose conversion is not recorded cannot be checked."""
    constraint = NumericConstraint.parse(0.3, "µM")

    assert constraint.conversion is not None
    assert constraint.conversion.from_unit == "µM"
    assert constraint.conversion.to_unit == "nM"
    assert constraint.conversion.factor == "1000"
    assert constraint.conversion.dimension == CONCENTRATION


def test_the_raw_value_survives_canonicalisation():
    """An answer quotes what the source wrote; only comparison uses nM."""
    data = NumericConstraint.parse(0.3, "µM").to_dict()

    assert data["raw"] == {"value": "0.3", "unit": "µM", "comparator": "="}
    assert data["canonical"]["low"] == "300"
    assert data["canonical"]["unit"] == "nM"


def test_u_is_accepted_for_keyboards_without_mu():
    assert NumericConstraint.parse(1, "uM").equivalent_to(NumericConstraint.parse(1, "µM"))


def test_an_unknown_unit_is_unknown_not_a_wildcard():
    """§5.2: missing or unplaceable metadata is unknown, never a match."""
    constraint = NumericConstraint.parse(5, "fold-change")

    assert constraint.kind is ConstraintKind.UNKNOWN_UNIT
    assert not constraint.placed
    assert constraint.conversion is None
    assert constraint.raw_value == "5"  # the record is kept; the comparison is refused


def test_two_identical_unknown_unit_strings_are_still_not_convertible():
    """Equal spelling is not the question; "may these be converted" is."""
    assert not same_dimension("fold-change", "fold-change")
    assert units_equivalent("fold-change", "fold-change")  # same scale as written
    assert unit_dimension("fold-change") is None


def test_units_of_different_dimensions_are_never_equivalent():
    assert not units_equivalent("nM", "%")
    assert not same_dimension("nM", "pIC50")


def test_a_pic50_is_not_turned_into_a_concentration():
    """ADR 0013 keeps them apart, and so does this module: the conversion needs
    a convention that belongs with the caller who knows the assay."""
    pic50 = NumericConstraint.parse(6.5, "pIC50")
    nanomolar = NumericConstraint.parse(316, "nM")

    assert not pic50.equivalent_to(nanomolar)
    agreement, reason = pic50.compare(nanomolar)
    assert agreement is Agreement.UNDETERMINED
    assert "dimension" in reason


def test_canonicalise_refuses_rather_than_guesses():
    assert canonicalise(1, "fold-change") == (None, None)
    assert canonicalise(None, "nM") == (None, None)
    value, conversion = canonicalise(1, "µM")
    assert (value, conversion.factor) == (Decimal("1000"), "1000")


# ── numeric constraint: what the number permits ───────────────────────────


def test_a_point_value_below_the_threshold_is_a_determinate_yes():
    exact = NumericConstraint.parse(0.3, "µM")

    assert exact.kind is ConstraintKind.EXACT
    assert exact.permits_threshold(10, "µM", direction="below") is Verdict.YES
    assert exact.permits_threshold(10, "µM", direction="above") is Verdict.NO


def test_a_lower_bound_can_settle_a_threshold_question():
    """"> 30 µM" against "below 10 µM" is a determinate no: everything the
    source permits is above the cut."""
    censored = NumericConstraint.parse(30, "µM", comparator=">")

    assert censored.kind is ConstraintKind.CENSORED
    assert censored.high is None
    assert censored.permits_threshold(10, "µM", direction="below") is Verdict.NO
    assert censored.permits_threshold(10, "µM", direction="above") is Verdict.YES


def test_an_upper_bound_leaves_the_same_question_open():
    """"< 30 µM" permits values on both sides of 10 µM, so the honest answer
    is undetermined -- not the midpoint, and not the bound read as a value."""
    censored = NumericConstraint.parse(30, "µM", comparator="<")

    assert censored.permits_threshold(10, "µM", direction="below") is Verdict.UNDETERMINED
    assert censored.permits_threshold(10, "µM", direction="above") is Verdict.UNDETERMINED


def test_a_threshold_in_another_unit_of_the_same_dimension_still_applies():
    """A task rule stated in µM applies to a record in nM without the caller
    converting first -- converting at each call site is how they drift."""
    record = NumericConstraint.parse(300, "nM")

    assert record.permits_threshold(10, "µM", direction="below") is Verdict.YES


def test_a_threshold_in_another_dimension_does_not_apply():
    record = NumericConstraint.parse(300, "nM")

    assert record.permits_threshold(10, "%", direction="below") is Verdict.UNDETERMINED


def test_a_range_keeps_both_ends():
    """1-10 µM is a range; a midpoint would be a measurement nobody made."""
    span = NumericConstraint.parse(1, "µM", high=10)

    assert span.kind is ConstraintKind.RANGE
    assert (span.low, span.high) == (Decimal("1000"), Decimal("10000"))
    assert span.permits_threshold(10, "µM", direction="below") is Verdict.UNDETERMINED
    assert span.permits_threshold(0.5, "µM", direction="above") is Verdict.YES


def test_an_inclusive_bound_includes_its_endpoint():
    assert NumericConstraint.parse(10, "µM", comparator="<=").high_inclusive
    assert not NumericConstraint.parse(10, "µM", comparator="<").high_inclusive
    assert (NumericConstraint.parse(10, "µM", comparator="≤")
            .permits_threshold(10, "µM", direction="below") is Verdict.UNDETERMINED)


def test_a_record_with_no_value_permits_nothing_and_says_so():
    nothing = NumericConstraint.parse(None, "nM")

    assert nothing.kind is ConstraintKind.NOT_ASSESSED
    assert not nothing.placed
    assert nothing.permits_threshold(10, "µM", direction="below") is Verdict.UNDETERMINED
    assert NumericConstraint.not_assessed().kind is ConstraintKind.NOT_ASSESSED


def test_direction_must_be_one_of_the_two():
    with pytest.raises(ValueError):
        NumericConstraint.parse(1, "nM").permits_threshold(1, "nM", direction="sideways")


# ── comparison: a difference is not yet a contradiction ───────────────────


def test_two_disjoint_values_are_undetermined_without_a_declared_tolerance():
    """§5.2: two different numbers are not a contradiction by themselves. The
    terfenadine case -- 60 nM binding against a 56 000 nM patch clamp -- is
    where this matters, and the module refuses to settle it by `!=`."""
    binding = NumericConstraint.parse(60, "nM")
    patch_clamp = NumericConstraint.parse(56000, "nM")

    agreement, reason = binding.compare(patch_clamp)
    assert agreement is Agreement.UNDETERMINED
    assert "no measurement tolerance was declared" in reason


def test_a_declared_tolerance_turns_a_large_gap_into_a_disagreement():
    binding = NumericConstraint.parse(60, "nM")
    patch_clamp = NumericConstraint.parse(56000, "nM")

    agreement, reason = binding.compare(patch_clamp, tolerance_fold=10)
    assert agreement is Agreement.DISAGREE
    assert "beyond the declared" in reason


def test_a_gap_within_the_declared_tolerance_is_not_a_disagreement():
    low = NumericConstraint.parse(50, "nM")
    high = NumericConstraint.parse(320, "nM")

    agreement, reason = low.compare(high, tolerance_fold=10)
    assert agreement is Agreement.UNDETERMINED
    assert "within the declared" in reason


def test_overlapping_ranges_agree_across_units():
    exact = NumericConstraint.parse(300, "nM")
    censored = NumericConstraint.parse(1, "µM", comparator="<")

    agreement, reason = exact.compare(censored)
    assert agreement is Agreement.AGREE
    assert "overlap" in reason


def test_an_unbounded_gap_has_no_fold_difference():
    """Two lower bounds cannot be set apart by a ratio, however large the
    tolerance: neither side's upper end is stated."""
    one = NumericConstraint.parse(30, "µM", comparator=">")
    other = NumericConstraint.parse(60, "µM", comparator=">")

    agreement, _ = one.compare(other, tolerance_fold=2)
    assert agreement is Agreement.AGREE  # both permit everything above 60 µM

    below = NumericConstraint.parse(1, "nM", comparator="<")
    above = NumericConstraint.parse(1000, "nM", comparator=">")
    agreement, reason = below.compare(above, tolerance_fold=2)
    assert agreement is Agreement.DISAGREE


def test_a_zero_floor_leaves_the_fold_difference_undefined():
    """A ratio against zero is not a fold difference; the answer stays open."""
    at_zero = NumericConstraint.parse(0, "nM")
    above = NumericConstraint.parse(10, "nM")

    agreement, reason = at_zero.compare(above, tolerance_fold=2)
    assert agreement is Agreement.UNDETERMINED
    assert "not defined" in reason


def test_comparison_with_an_unplaceable_record_is_undetermined():
    placed = NumericConstraint.parse(300, "nM")
    unplaceable = NumericConstraint.parse(5, "fold-change")

    agreement, reason = placed.compare(unplaceable)
    assert agreement is Agreement.UNDETERMINED
    assert "no usable value" in reason

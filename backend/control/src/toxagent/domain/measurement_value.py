"""Measurement values: how a number is written, and what it constrains (#59).

§5.2 of the 2026-10-28 architecture proposal splits what was one string
comparison into three layers. Layer 1, identity and context, is
:class:`~toxagent.domain.evidence_relation.EvidenceScope`'s: same ligand, same
target, same assay. This module is the other two.

* **Representation.** The same quantity written two ways. ``0.3 µM`` and
  ``300 nM`` are one value in two units and must not read as two.
  :func:`canonicalise` returns the value in its dimension's canonical unit
  *and the conversion it used*: a converted number whose conversion is not
  recorded cannot be checked by the researcher the answer is for.
* **Numeric constraint.** What the number permits. ``> 30 µM`` is a bound,
  ``0.3 µM`` is a point, ``1–10 µM`` is a range, and the three license
  different conclusions under the same task rule.
  :meth:`NumericConstraint.permits_threshold` answers that question, and
  ``UNDETERMINED`` is one of its three answers.

Three rules carried from §5.2, each of them a way this module refuses to help:

* **A unit it does not know is unknown, never a wildcard.** Nothing is
  converted by guessing; an unconvertible pair is reported as unconvertible.
* **Two different numbers are not a contradiction by themselves.**
  :meth:`NumericConstraint.compare` answers ``undetermined`` for disjoint
  values unless the caller declares the tolerance beyond which a difference is
  real. There is no default tolerance here: how far apart two measurements of
  one quantity may legitimately sit is a property of the assays, not of this
  module, and inventing a number would put a scientific constant in a utility.
* **No dimension bridging.** pIC50 and a molar IC50 are inter-convertible in
  principle, and this module still will not do it, for the reason
  ``exposure_margin`` will not convert a mass concentration: the conversion
  needs a convention (which log, of which unit) that belongs with the caller
  who knows the assay, not with the units table.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Any

#: Dimensions. Two values are only ever comparable within one of them.
CONCENTRATION = "concentration"
LOG_MOLAR = "log_molar"
PERCENT = "percent"

#: Unit -> (dimension, factor to the dimension's canonical unit). The
#: concentration factors are the ones ``domain/exposure_margin.py`` has used
#: since W9-13, which imports them from here so there is one table rather than
#: two that can drift. ``uM`` is accepted for keyboards without µ.
_UNITS: dict[str, tuple[str, Decimal]] = {
    "pM": (CONCENTRATION, Decimal("0.001")),
    "nM": (CONCENTRATION, Decimal(1)),
    "µM": (CONCENTRATION, Decimal(1000)),
    "uM": (CONCENTRATION, Decimal(1000)),
    "mM": (CONCENTRATION, Decimal(1_000_000)),
    "M": (CONCENTRATION, Decimal(1_000_000_000)),
    # Log-molar potency keeps its own dimension on purpose: a pIC50 is not a
    # concentration, and nothing here turns one into the other (ADR 0013).
    "pIC50": (LOG_MOLAR, Decimal(1)),
    "pKi": (LOG_MOLAR, Decimal(1)),
    "%": (PERCENT, Decimal(1)),
}

#: The unit each dimension is canonicalised to.
CANONICAL_UNIT: dict[str, str] = {
    CONCENTRATION: "nM",
    LOG_MOLAR: "pIC50",
    PERCENT: "%",
}

#: Comparators a provider may state a value against, mapped to whether the
#: bound they describe includes the value itself.
_LOWER_BOUNDS = {">": False, ">=": True, "≥": True}
_UPPER_BOUNDS = {"<": False, "<=": True, "≤": True}
EXACT_COMPARATORS = ("=", "==", "~")


def concentration_factors() -> dict[str, Decimal]:
    """The molar units and their factors to nM, for a caller that accepts only
    those -- ``exposure_margin`` will not divide a mass by a molar amount, so
    it asks for this subset rather than the whole table."""
    return {unit: factor for unit, (dimension, factor) in _UNITS.items()
            if dimension == CONCENTRATION}


def unit_dimension(unit: str | None) -> str | None:
    """The dimension of ``unit``, or ``None`` when the unit is not known.

    Not knowing is a state of its own: the caller must not read ``None`` as
    "no constraint on comparison".
    """
    if unit is None:
        return None
    known = _UNITS.get(unit.strip())
    return known[0] if known else None


def same_dimension(left: str | None, right: str | None) -> bool:
    """Whether two units measure the same kind of thing.

    Two unknown units are *not* the same dimension, even when the strings are
    equal: this answers "may these be converted into each other", and an
    unknown unit may not.
    """
    mine, theirs = unit_dimension(left), unit_dimension(right)
    return mine is not None and mine == theirs


def units_equivalent(left: str | None, right: str | None) -> bool:
    """Whether two unit strings are two spellings of one comparable scale.

    ``nM``/``µM`` are equivalent as *units* -- a value in one converts into
    the other -- so a record pair differing only in unit is not a pair of
    different measurements (§5.2). The magnitudes they imply are the numeric
    layer's business, not this function's.
    """
    if left is None or right is None:
        return False
    if left.strip() == right.strip():
        return True
    return same_dimension(left, right)


@dataclass(frozen=True, slots=True)
class Conversion:
    """How a raw value became a canonical one. Recorded so it can be checked."""

    from_unit: str
    to_unit: str
    factor: str
    dimension: str

    def to_dict(self) -> dict[str, Any]:
        return {"from_unit": self.from_unit, "to_unit": self.to_unit,
                "factor": self.factor, "dimension": self.dimension}


class ConstraintKind(str, Enum):
    """What shape of number a record states.

    ``NOT_ASSESSED`` is for a record that states no value; ``UNKNOWN_UNIT``
    for one that states a value in a unit this domain cannot place. Both are
    reportable outcomes, not errors, and neither is a licence to compare.
    """

    EXACT = "exact"
    RANGE = "range"
    CENSORED = "censored"
    NOT_ASSESSED = "not_assessed"
    UNKNOWN_UNIT = "unknown_unit"


class Verdict(str, Enum):
    """Three answers. ``UNDETERMINED`` is the honest one more often than not."""

    YES = "yes"
    NO = "no"
    UNDETERMINED = "undetermined"


class Agreement(str, Enum):
    AGREE = "agree"
    DISAGREE = "disagree"
    UNDETERMINED = "undetermined"


def _plain(value: Decimal | None) -> str | None:
    """A decimal as a human writes it: no exponent, no trailing zeros.

    ``Decimal("0.3") * 1000`` is ``300.0``, and ``normalize()`` alone would
    make that ``3E+2``. Both are the same number and neither is what a
    researcher reads in an answer.
    """
    if value is None:
        return None
    return format(value.normalize(), "f")


def _decimal(value: Any) -> Decimal | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return Decimal(repr(float(value))) if isinstance(value, float) else Decimal(str(value).strip())
    except (InvalidOperation, ValueError, TypeError):
        return None


@dataclass(frozen=True, slots=True)
class NumericConstraint:
    """A stated value as the set of values it permits.

    The interval is in the dimension's canonical unit, so two constraints from
    different units are comparable without either side re-deriving the
    conversion. ``low``/``high`` of ``None`` mean unbounded on that side, which
    is what a censored value is: ``> 30 µM`` permits everything above 30 000 nM
    and says nothing about how far above.

    The raw form is kept beside the canonical one, always. An answer quotes
    what the source wrote; only the comparison uses the converted number.
    """

    kind: ConstraintKind
    #: As written by the source.
    raw_value: str | None = None
    raw_unit: str | None = None
    comparator: str = "="
    #: Canonical interval. Both ``None`` when nothing numeric is established.
    low: Decimal | None = None
    high: Decimal | None = None
    #: Whether the bounds include their endpoints.
    low_inclusive: bool = True
    high_inclusive: bool = True
    canonical_unit: str | None = None
    dimension: str | None = None
    conversion: Conversion | None = None

    # ── construction ──────────────────────────────────────────────────────

    @classmethod
    def not_assessed(cls) -> "NumericConstraint":
        return cls(kind=ConstraintKind.NOT_ASSESSED)

    @classmethod
    def parse(
        cls, value: Any, unit: str | None, *, comparator: str | None = None,
        high: Any = None,
    ) -> "NumericConstraint":
        """Build a constraint from what a provider stated.

        ``high`` makes it a range: a record giving 1–10 µM states a range, not
        a point, and flattening it to a midpoint would invent a measurement.
        """
        raw = _decimal(value)
        raw_text = None if value is None else str(value).strip()
        comparator = (comparator or "=").strip() or "="
        if raw is None:
            return cls(kind=ConstraintKind.NOT_ASSESSED, raw_value=raw_text,
                       raw_unit=unit, comparator=comparator)

        known = _UNITS.get((unit or "").strip())
        if known is None:
            # A value in an unplaceable unit is still the value the source
            # wrote; it is the comparison that is refused, not the record.
            return cls(kind=ConstraintKind.UNKNOWN_UNIT, raw_value=raw_text,
                       raw_unit=unit, comparator=comparator)

        dimension, factor = known
        canonical = CANONICAL_UNIT[dimension]
        conversion = Conversion(from_unit=(unit or "").strip(), to_unit=canonical,
                                factor=str(factor), dimension=dimension)
        low_value = raw * factor

        upper = _decimal(high)
        if upper is not None:
            top, bottom = max(low_value, upper * factor), min(low_value, upper * factor)
            return cls(kind=ConstraintKind.RANGE, raw_value=raw_text, raw_unit=unit,
                       comparator=comparator, low=bottom, high=top,
                       canonical_unit=canonical, dimension=dimension, conversion=conversion)

        if comparator in _LOWER_BOUNDS:
            return cls(kind=ConstraintKind.CENSORED, raw_value=raw_text, raw_unit=unit,
                       comparator=comparator, low=low_value, high=None,
                       low_inclusive=_LOWER_BOUNDS[comparator],
                       canonical_unit=canonical, dimension=dimension, conversion=conversion)
        if comparator in _UPPER_BOUNDS:
            return cls(kind=ConstraintKind.CENSORED, raw_value=raw_text, raw_unit=unit,
                       comparator=comparator, low=None, high=low_value,
                       high_inclusive=_UPPER_BOUNDS[comparator],
                       canonical_unit=canonical, dimension=dimension, conversion=conversion)
        return cls(kind=ConstraintKind.EXACT, raw_value=raw_text, raw_unit=unit,
                   comparator=comparator, low=low_value, high=low_value,
                   canonical_unit=canonical, dimension=dimension, conversion=conversion)

    # ── what it permits ───────────────────────────────────────────────────

    @property
    def placed(self) -> bool:
        """Whether this constraint says anything a comparison can use."""
        return self.kind in (ConstraintKind.EXACT, ConstraintKind.RANGE,
                             ConstraintKind.CENSORED)

    def permits_threshold(
        self, threshold: Any, unit: str, *, direction: str = "below"
    ) -> Verdict:
        """Whether the constrained value is ``direction`` the threshold.

        ``direction`` is ``"below"`` or ``"above"``, and the threshold carries
        its own unit so a task rule stated in µM can be applied to a record in
        nM without the caller converting first.

        This is where a censored value earns its keeping: ``> 30 µM`` against
        "below 10 µM" is a determinate **no**, while ``< 30 µM`` against the
        same rule is **undetermined** -- the source permits values on both
        sides. Collapsing either to a point estimate would answer a question
        the source did not answer.
        """
        if direction not in ("below", "above"):
            raise ValueError("direction must be 'below' or 'above'")
        limit = _decimal(threshold)
        if limit is None or not self.placed or self.dimension is None:
            return Verdict.UNDETERMINED
        known = _UNITS.get(unit.strip())
        if known is None or known[0] != self.dimension:
            return Verdict.UNDETERMINED
        cut = limit * known[1]

        if direction == "below":
            # Entirely below the cut?
            if self.high is not None and (self.high < cut or
                                          (self.high == cut and not self.high_inclusive)):
                return Verdict.YES
            # Entirely at or above it?
            if self.low is not None and self.low >= cut:
                return Verdict.NO
            return Verdict.UNDETERMINED

        if self.low is not None and (self.low > cut or
                                     (self.low == cut and not self.low_inclusive)):
            return Verdict.YES
        if self.high is not None and self.high <= cut:
            return Verdict.NO
        return Verdict.UNDETERMINED

    def equivalent_to(self, other: "NumericConstraint") -> bool:
        """Whether both state the same value, allowing for how it was written.

        ``0.3 µM`` and ``300 nM`` are equivalent; this is the representation
        layer's whole job (§5.2).
        """
        if not (self.placed and other.placed):
            return False
        if self.dimension != other.dimension or self.kind is not other.kind:
            return False
        return (self.low, self.high, self.low_inclusive, self.high_inclusive) == (
            other.low, other.high, other.low_inclusive, other.high_inclusive)

    def compare(
        self, other: "NumericConstraint", *, tolerance_fold: Any = None
    ) -> tuple[Agreement, str]:
        """Whether these two numbers disagree, and the reason for the answer.

        Overlapping intervals agree. Disjoint intervals are **not** a
        contradiction on their own: two measurements of one quantity differ,
        and how far apart they may legitimately sit is a property of the
        assays. So a disjoint pair is ``undetermined`` unless the caller
        declares ``tolerance_fold`` -- the ratio beyond which the gap is more
        than measurement spread -- and the gap exceeds it. §5.2: "Hai số khác
        nhau cũng không tự thành scientific contradiction."
        """
        if not (self.placed and other.placed):
            return Agreement.UNDETERMINED, "one side states no usable value"
        if self.dimension != other.dimension:
            return Agreement.UNDETERMINED, "different dimensions; not convertible"

        if self._overlaps(other):
            return Agreement.AGREE, "the stated ranges overlap"

        fold = _decimal(tolerance_fold)
        if fold is None or fold <= 0:
            return (Agreement.UNDETERMINED,
                    "the stated values are disjoint, but no measurement tolerance "
                    "was declared, so the gap may be assay spread")
        gap = self._fold_gap(other)
        if gap is None:
            return (Agreement.UNDETERMINED,
                    "the gap is unbounded on one side; a fold difference is not defined")
        if gap > fold:
            return Agreement.DISAGREE, f"the values differ by {gap:f}-fold, beyond the declared {fold:f}"
        return Agreement.UNDETERMINED, f"the values differ by {gap:f}-fold, within the declared {fold:f}"

    # ── interval arithmetic ───────────────────────────────────────────────

    def _overlaps(self, other: "NumericConstraint") -> bool:
        if self.high is not None and other.low is not None:
            if self.high < other.low:
                return False
            if self.high == other.low and not (self.high_inclusive and other.low_inclusive):
                return False
        if other.high is not None and self.low is not None:
            if other.high < self.low:
                return False
            if other.high == self.low and not (other.high_inclusive and self.low_inclusive):
                return False
        return True

    def _fold_gap(self, other: "NumericConstraint") -> Decimal | None:
        """The smallest ratio between the two permitted sets, or ``None`` when
        one side is unbounded towards the other and no finite gap exists."""
        if self.high is not None and other.low is not None and self.high < other.low:
            lower, upper = self.high, other.low
        elif other.high is not None and self.low is not None and other.high < self.low:
            lower, upper = other.high, self.low
        else:
            return None
        if lower <= 0:
            return None
        return upper / lower

    def to_dict(self) -> dict[str, Any]:
        """Raw and canonical side by side; the answer quotes the raw one."""
        return {
            "kind": self.kind.value,
            "raw": {"value": self.raw_value, "unit": self.raw_unit,
                    "comparator": self.comparator},
            "canonical": {
                "low": _plain(self.low),
                "high": _plain(self.high),
                "low_inclusive": self.low_inclusive,
                "high_inclusive": self.high_inclusive,
                "unit": self.canonical_unit,
                "dimension": self.dimension,
            },
            "conversion": None if self.conversion is None else self.conversion.to_dict(),
        }


def canonicalise(value: Any, unit: str | None) -> tuple[Decimal | None, Conversion | None]:
    """One value in its dimension's canonical unit, with the conversion used.

    Returns ``(None, None)`` when the value or unit cannot be placed -- the
    caller reports that, rather than substituting a guess.
    """
    constraint = NumericConstraint.parse(value, unit)
    if not constraint.placed:
        return None, None
    return constraint.low, constraint.conversion

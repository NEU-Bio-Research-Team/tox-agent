"""A server-computed summary of measured activities for one structure and target.

Live e2e, 2026-09-26: ``get_chembl_activities`` returned ten hERG records for
terfenadine — nine between 50 and 320 nM and one patch-clamp IC50 of 56 000 nM.
The model read three of them, called the data "inconsistent", described
204 nM as *more* potent than 56 nM, and recommended a patch-clamp study the
records already contained. The spread of a set of measurements is arithmetic,
not judgement, so the server does it: per measurement type, how many exact
values, their range and median, and which records sit far from the rest.

A record flagged as far from the median is not declared wrong. A value 1000×
away from its siblings is often a unit or curation error, and sometimes a
real assay difference; the summary says which records to check, not which to
drop.
"""
from __future__ import annotations

import math
from statistics import median
from typing import Any, Iterable, Mapping

METHOD_VERSION = "activity-summary-1"

#: 1.5 log units, about 30-fold, from the median of the same measurement type.
OUTLIER_LOG_UNITS = 1.5

_NANOMOLAR = {"nM": 1.0, "uM": 1e3, "µM": 1e3, "μM": 1e3, "pM": 1e-3, "mM": 1e6}


def _to_nanomolar(value: Any, units: Any) -> float | None:
    factor = _NANOMOLAR.get(str(units or "").strip())
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if factor is None or not math.isfinite(number) or number <= 0:
        return None
    return number * factor


def _round(value: float) -> float:
    """Three significant figures: enough to compare potencies, not a false precision."""
    return float(f"{value:.3g}")


def summarize(records: Iterable[tuple[str, Mapping[str, Any]]]) -> dict[str, Any]:
    """``records`` are ``(evidence_id, normalized_facts)`` pairs.

    Only exact (``=``) values in a concentration unit enter the statistics;
    qualified ones (``>``, ``<``) are counted apart, because "> 30 µM" is a
    bound, not a measurement.
    """
    groups: dict[str, list[tuple[str, float]]] = {}
    qualified: dict[str, int] = {}
    for evidence_id, facts in records:
        kind = str(facts.get("standard_type") or "").strip() or "unknown"
        if (facts.get("standard_relation") or "=") != "=":
            qualified[kind] = qualified.get(kind, 0) + 1
            continue
        value = _to_nanomolar(facts.get("standard_value"), facts.get("standard_units"))
        if value is None:
            qualified[kind] = qualified.get(kind, 0) + 1
            continue
        groups.setdefault(kind, []).append((evidence_id, value))

    by_type: dict[str, Any] = {}
    for kind, members in sorted(groups.items()):
        values = [value for _, value in members]
        mid = median(values)
        far = [
            {"evidence_id": evidence_id, "value_nM": _round(value),
             "fold_from_median": _round(max(value, mid) / min(value, mid))}
            for evidence_id, value in members
            if abs(math.log10(value) - math.log10(mid)) > OUTLIER_LOG_UNITS
        ]
        within = len(values) - len(far)
        finding = (
            f"{within} of {len(values)} {kind} values lie within 30-fold of the median "
            f"{_round(mid):g} nM (range {_round(min(values)):g}-{_round(max(values)):g} nM)."
        )
        if far:
            finding += (f" {len(far)} sit far from the rest; check their units, relation and "
                        "assay before weighing them.")
        by_type[kind] = {
            "finding": finding,
            "n": len(values),
            "min_nM": _round(min(values)),
            "median_nM": _round(mid),
            "max_nM": _round(max(values)),
            # pIC50 / pKi of the median: -log10 of the molar concentration.
            "median_p": round(9 - math.log10(mid), 2),
            "within_30_fold_of_median": within,
            "far_from_median": far,
        }
    for kind, count in qualified.items():
        by_type.setdefault(kind, {"n": 0})["qualified_or_unitless"] = count
    return {
        "method_version": METHOD_VERSION,
        "by_type": by_type,
        "reading": (
            "Lower IC50/Ki/Kd means more potent: 50 nM blocks more strongly than 200 nM. "
            "Weigh the set, not one record. A record far from the median (over ~30-fold) is "
            "worth checking for units, relation and assay before it is weighed; it is not, "
            "on its own, evidence that the measurements disagree."
        ),
    }

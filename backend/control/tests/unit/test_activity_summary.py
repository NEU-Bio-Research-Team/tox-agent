"""The server-computed spread of measured activities (live e2e, 2026-09-26)."""
from __future__ import annotations

from toxagent.domain.activity_summary import summarize

#: The ten hERG records ChEMBL returned for terfenadine on the live stack.
TERFENADINE = [
    ("evd_a", "IC50", "204.0"), ("evd_b", "IC50", "56.0"), ("evd_c", "IC50", "56.0"),
    ("evd_d", "IC50", "199.53"), ("evd_e", "Ki", "320.0"), ("evd_f", "IC50", "128.82"),
    ("evd_g", "IC50", "50.0"), ("evd_h", "IC50", "199.53"), ("evd_i", "IC50", "56000.0"),
    ("evd_j", "IC50", "213.0"),
]


def _facts(kind: str, value: str, relation: str = "=", units: str = "nM") -> dict:
    return {"standard_type": kind, "standard_value": value, "standard_relation": relation,
            "standard_units": units}


def test_the_terfenadine_set_reads_as_consistent_with_one_record_to_check():
    summary = summarize((eid, _facts(kind, value)) for eid, kind, value in TERFENADINE)
    ic50 = summary["by_type"]["IC50"]
    assert ic50["n"] == 9
    assert ic50["min_nM"] == 50.0 and ic50["max_nM"] == 56000.0
    assert ic50["median_nM"] == 200.0  # 199.53 at three significant figures
    assert ic50["median_p"] == 6.7
    assert ic50["within_30_fold_of_median"] == 8
    assert [far["evidence_id"] for far in ic50["far_from_median"]] == ["evd_i"]
    assert summary["by_type"]["Ki"]["n"] == 1
    assert "more potent" in summary["reading"]
    assert ic50["finding"].startswith("8 of 9 IC50 values lie within 30-fold of the median 200 nM")


def test_bounds_and_unknown_units_stay_out_of_the_statistics():
    summary = summarize([
        ("evd_a", _facts("IC50", "30", relation=">", units="uM")),
        ("evd_b", _facts("IC50", "12", units="%")),
        ("evd_c", _facts("IC50", "1.5", units="uM")),
    ])
    ic50 = summary["by_type"]["IC50"]
    assert ic50["n"] == 1 and ic50["median_nM"] == 1500.0
    assert ic50["qualified_or_unitless"] == 2

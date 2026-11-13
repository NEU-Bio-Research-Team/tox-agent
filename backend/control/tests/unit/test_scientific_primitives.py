"""W9-13: scientific primitives a skill cannot supply (RETHINK §4.7, §4.10)."""
from __future__ import annotations


import pytest

from toxagent.domain import exposure_margin as em

pytestmark = pytest.mark.anyio


def test_the_margin_is_ic50_over_free_cmax_in_the_same_unit():
    result = em.compute(em.Concentration(30, "µM", "context:c1"),
                        em.Concentration(50, "nM", "context:c2"))
    assert result["margin"] == 600.0
    assert result["ic50_nM"] == 30_000.0 and result["free_cmax_nM"] == 50.0


def test_a_total_cmax_is_corrected_by_the_fraction_unbound():
    result = em.compute(em.Concentration(1, "uM", "evidence:e"),
                        em.Concentration(2, "µM", "context:c2"),
                        fraction_unbound=0.1, fu_source_ref="context:c3")
    assert result["free_cmax_nM"] == 200.0
    assert result["margin"] == 5.0
    assert result["inputs"]["cmax"]["kind"] == "total, corrected by fraction unbound"


@pytest.mark.parametrize(("kwargs", "message"), [
    ({"ic50": em.Concentration(1, "mg/L", "c")}, "unit must be one of"),
    ({"ic50": em.Concentration(0, "nM", "c")}, "greater than zero"),
    ({"fraction_unbound": 1.5}, r"\(0, 1\]"),
])
def test_bad_inputs_are_refused(kwargs, message):
    args = {"ic50": em.Concentration(1, "nM", "c"), "cmax": em.Concentration(1, "nM", "c")}
    fu = kwargs.pop("fraction_unbound", None)
    args.update(kwargs)
    with pytest.raises(em.InvalidMarginInput, match=message):
        em.compute(args["ic50"], args["cmax"], fraction_unbound=fu)


def test_a_value_must_be_written_in_its_source():
    assert em.transcription_check(30, "patch_clamp_ic50 30 µM, in-house")
    assert em.transcription_check(0.05, "free Cmax 0,05 µM")
    assert em.transcription_check(13.5, "IC50 = 13.5 nM")
    assert not em.transcription_check(3, "IC50 = 30 µM")
    assert not em.transcription_check(0.5, "free Cmax 0.05 µM")

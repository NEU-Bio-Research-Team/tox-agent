"""W9-13: scientific primitives a skill cannot supply (RETHINK §4.7, §4.10)."""
from __future__ import annotations


import httpx
import pytest

from toxagent.config import ChemblSettings
from toxagent.domain import exposure_margin as em
from toxagent.research.providers.chembl import ChemblActivityProvider

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


def _chembl(handler) -> ChemblActivityProvider:
    return ChemblActivityProvider(ChemblSettings(retry_attempts=1),
                                  transport=httpx.MockTransport(handler))


async def test_chembl_matches_the_structure_then_reads_activities_against_the_target():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path.endswith("/molecule.json"):
            return httpx.Response(200, json={"molecules": [
                {"molecule_chembl_id": "CHEMBL17157", "pref_name": "TERFENADINE"}]})
        return httpx.Response(200, json={"activities": [
            {"activity_id": 101, "standard_type": "IC50", "standard_relation": "=",
             "standard_value": "56.0", "standard_units": "nM", "pchembl_value": "7.25",
             "assay_chembl_id": "CHEMBL999", "assay_type": "B",
             "assay_description": "Inhibition of hERG in HEK293 cells by patch clamp",
             "document_chembl_id": "CHEMBL1", "document_year": 2003,
             "target_pref_name": "HERG", "target_organism": "Homo sapiens"},
            {"activity_id": 102, "standard_type": "IC50", "standard_value": None},
        ]})

    lookup = await _chembl(handler).activities(canonical_smiles="CC(C)(C)c1ccc", target="herg",
                                               limit=10)
    assert lookup.molecule_chembl_id == "CHEMBL17157"
    assert seen[0].url.params["molecule_structures__canonical_smiles__flexmatch"] == "CC(C)(C)c1ccc"
    assert seen[1].url.params["target_chembl_id"] == "CHEMBL240"
    [hit] = lookup.hits  # a row without a value is not a measurement
    assert hit.provider_record_id == "activity:101"
    assert "IC50 = 56.0 nM" in hit.title
    assert hit.normalized_facts["assay_chembl_id"] == "CHEMBL999"
    assert hit.canonical_url.startswith("https://www.ebi.ac.uk/")


async def test_a_structure_chembl_does_not_hold_is_a_gap_not_an_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"molecules": []})

    lookup = await _chembl(handler).activities(canonical_smiles="C", target="herg", limit=5)
    assert lookup.molecule_chembl_id is None and lookup.hits == ()


def test_a_misconfigured_host_is_refused_at_start_up():
    with pytest.raises(ValueError, match="allowed_hosts"):
        ChemblActivityProvider(ChemblSettings(base_url="https://evil.example/chembl"))

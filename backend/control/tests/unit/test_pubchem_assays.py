"""Structure-bound PubChem assays retain their provenance and target scope."""
from __future__ import annotations

import httpx
import pytest

from toxagent.domain.errors import EvidenceUnavailable
from toxagent.platform.config import CompoundSettings
from toxagent.research.providers.pubchem_assays import PubChemAssayProvider

pytestmark = pytest.mark.anyio

COLUMNS = ["AID", "SID", "CID", "Activity Outcome", "Target Accession", "Target GeneID",
           "Activity Value [uM]", "Activity Name", "Assay Name", "Assay Type", "PubMed ID"]
ROWS = [
    ["82355", "103181781", "5405", "Active", "Q12809", "3757", "0.204", "IC50",
     "HERG inhibition in human cells", "Confirmatory", "12190308"],
    ["1511", "855647", "5405", "Inactive", "AAA62473", "3757", "", "",
     "Protects hERG from block", "Screening", ""],
    ["42", "33", "5405", "Active", "Q11111", "100", "5", "IC50", "Unrelated", "", ""],
    ["44", "55", "9999", "Active", "Q12809", "3757", "1", "IC50", "Wrong CID", "", ""],
]


def provider(handler):
    return PubChemAssayProvider(CompoundSettings(), transport=httpx.MockTransport(handler))


async def test_real_assay_shape_keeps_target_identity_outcome_and_source_separate():
    requests = []
    def handler(request):
        requests.append(request)
        if request.method == "POST":
            return httpx.Response(200, json={"IdentifierList": {"CID": [5405]}})
        return httpx.Response(200, json={"Table": {"Columns": {"Column": COLUMNS},
                                                    "Row": [{"Cell": r} for r in ROWS]}})
    service = provider(handler)
    result = await service.assays(canonical_smiles="CCC", target="herg", limit=1)
    assert result.cid == 5405 and result.total_matches == 2 and len(result.hits) == 1
    assert requests[0].method == "POST" and "CCC" not in str(requests[0].url)
    [hit] = result.hits
    assert hit.canonical_url == "https://pubchem.ncbi.nlm.nih.gov/bioassay/82355"
    assert hit.normalized_facts["activity_outcome"] == "Active"
    assert hit.normalized_facts["activity_value_uM"] == "0.204"
    assert hit.normalized_facts["pubmed_id"] == "12190308"
    assert hit.identifier.pmid is None
    await service.aclose()


@pytest.mark.parametrize("body", [{}, {"IdentifierList": {"CID": [1, 2]}}, {"IdentifierList": {"CID": []}}])
async def test_ambiguous_or_malformed_identity_is_not_reported_as_no_evidence(body):
    service = provider(lambda request: httpx.Response(200, json=body))
    with pytest.raises(EvidenceUnavailable):
        await service.assays(canonical_smiles="C", target="herg", limit=10)
    await service.aclose()


async def test_no_compound_is_a_gap_but_bad_assay_schema_is_a_provider_failure():
    service = provider(lambda request: httpx.Response(404))
    assert (await service.assays(canonical_smiles="C", target="herg", limit=10)).cid is None
    await service.aclose()
    service = provider(lambda request: httpx.Response(200, json={"IdentifierList": {"CID": [1]}}
                                                      if request.method == "POST" else {"Table": {}}))
    with pytest.raises(EvidenceUnavailable):
        await service.assays(canonical_smiles="C", target="herg", limit=10)
    await service.aclose()

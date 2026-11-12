"""PubChem BioAssay summaries, resolved by structure and identified target.

PubChem's assay summary is a screening record, not a safety verdict. It can
include the same underlying assay as other databases, and does not necessarily
include the experimental protocol or a numerical qualifier.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, replace
from typing import Any

import httpx

from ...domain.errors import EvidenceUnavailable
from ...domain.evidence import SourceIdentifier, SourceType
from ...platform.config import CompoundSettings
from ..interfaces import SearchHit
from .pubchem import PubChemCompoundProvider

# A distinct body limit is needed: a common compound can have thousands of
# assays (caffeine's summary was 1.75 MB). Still bounded against large inputs.
MAX_ASSAY_BYTES = 12 * 1024 * 1024


@dataclass(frozen=True)
class AssayLookup:
    cid: int | None
    hits: tuple[SearchHit, ...]
    total_matches: int


class PubChemAssayProvider(PubChemCompoundProvider):
    name = "pubchem_bioassay"

    def __init__(self, settings: CompoundSettings, *, transport: httpx.AsyncBaseTransport | None = None):
        # At most one attempt per request: retries should not consume the whole
        # agent turn and PubChem enforces shared request rate limits.
        super().__init__(replace(settings, max_response_bytes=MAX_ASSAY_BYTES, retry_attempts=1),
                         transport=transport)
        # PubChem asks all clients to stay under five requests per second.
        # Space this provider's requests within one control-plane process; a
        # multi-worker installation still needs a shared rate limiter.
        self._rate_lock = asyncio.Lock()
        self._next_request_at = 0.0

    async def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        async with self._rate_lock:
            delay = self._next_request_at - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            self._next_request_at = time.monotonic() + 0.5
        return await super()._request(method, path, **kwargs)

    @property
    def allowed_hosts(self) -> tuple[str, ...]:
        return self._settings.allowed_hosts

    async def assays(self, *, canonical_smiles: str, target: str, limit: int) -> AssayLookup:
        if target != "herg":
            raise ValueError("PubChem assay target is not supported")
        # POST avoids recording a potentially private structure in access logs.
        response = await self._request("POST", "/compound/smiles/cids/JSON",
                                       data={"smiles": canonical_smiles})
        if response.status_code == 404:
            return AssayLookup(None, (), 0)
        if response.status_code != 200:
            raise EvidenceUnavailable(f"PubChem identity lookup answered {response.status_code}")
        body = self._parse_json(response)
        identifiers = body.get("IdentifierList") if isinstance(body, dict) else None
        ids = identifiers.get("CID") if isinstance(identifiers, dict) else None
        if not isinstance(ids, list) or len(ids) != 1 or not isinstance(ids[0], int) or ids[0] <= 0:
            raise EvidenceUnavailable("PubChem did not resolve the structure to one compound")
        cid = ids[0]
        response = await self._request("GET", f"/compound/cid/{cid}/assaysummary/JSON")
        if response.status_code == 404:
            return AssayLookup(cid, (), 0)
        if response.status_code != 200:
            raise EvidenceUnavailable(f"PubChem assay lookup answered {response.status_code}")
        body = self._parse_json(response)
        table = body.get("Table") if isinstance(body, dict) else None
        if not isinstance(table, dict) or not isinstance(table.get("Columns"), dict):
            raise EvidenceUnavailable("PubChem returned an invalid assay summary")
        columns = table["Columns"].get("Column")
        rows = table.get("Row")
        required = {"AID", "SID", "CID", "Target Accession", "Target GeneID", "Assay Name"}
        if (not isinstance(columns, list) or not all(isinstance(c, str) for c in columns)
                or len(set(columns)) != len(columns) or not required.issubset(columns)
                or not isinstance(rows, list)):
            raise EvidenceUnavailable("PubChem returned an incomplete assay summary")
        matches: list[SearchHit] = []
        for row in rows:
            cells = row.get("Cell") if isinstance(row, dict) else None
            if not isinstance(cells, list) or len(cells) != len(columns):
                continue
            record = dict(zip(columns, cells))
            aid, sid = str(record.get("AID") or ""), str(record.get("SID") or "")
            if (str(record.get("CID")) != str(cid) or not aid.isdigit() or not sid.isdigit()
                    or not (str(record.get("Target GeneID") or "") == "3757"
                            or str(record.get("Target Accession") or "").upper() == "Q12809")):
                continue
            matches.append(self._hit(record, cid=cid, aid=aid, sid=sid))
        return AssayLookup(cid, tuple(matches[:limit]), len(matches))

    @staticmethod
    def _hit(row: dict[str, Any], *, cid: int, aid: str, sid: str) -> SearchHit:
        name = str(row.get("Assay Name") or "Untitled assay")[:1000]
        outcome = str(row.get("Activity Outcome") or "Unspecified")[:100]
        assay_type = str(row.get("Assay Type") or "")[:100]
        activity_name = str(row.get("Activity Name") or "")[:100]
        value = str(row.get("Activity Value [uM]") or "")[:100]
        pmid = str(row.get("PubMed ID") or "")[:30]
        facts = {"database": "PubChem BioAssay", "pubchem_cid": cid, "assay_id": aid,
                 "substance_id": sid, "target_gene_id": row.get("Target GeneID"),
                 "target_accession": row.get("Target Accession"), "activity_outcome": outcome,
                 "activity_name": activity_name, "activity_value_uM": value or None,
                 "assay_type": assay_type, "pubmed_id": pmid or None}
        # Distinct assays from one publication must not collapse under PMID.
        return SearchHit(
            provider_record_id=f"aid:{aid}:sid:{sid}", source_type=SourceType.DATABASE,
            title=f"PubChem BioAssay {aid}, CID {cid}: {name}",
            canonical_url=f"https://pubchem.ncbi.nlm.nih.gov/bioassay/{aid}",
            identifier=SourceIdentifier(other=f"pubchem_aid:{aid}:sid:{sid}"),
            abstract_or_excerpt=(f"{name}. Outcome: {outcome}; assay type: {assay_type or 'not stated'}. "
                                 f"Activity: {activity_name or 'not stated'} {value or 'not reported'} "
                                 "µM. Consult the assay protocol and original source; this summary "
                                 "does not establish safety or a numeric qualifier."),
            normalized_facts=facts, raw=row,
        )

"""ChEMBL bioactivity lookup (RETHINK §4.7/§4.10, W9-13).

"Access to ChEMBL/assay databases" is on the RETHINK list of things a skill
cannot add: it needs an adapter, provenance and permission. This is the
adapter. It finds the molecule *by structure* — the canonical SMILES the
analysis snapshot recorded, through ChEMBL's ``flexmatch`` (same connectivity;
it can also match stereoisomers or other forms, so the number of matches is
reported) — so the activities belong to the structure that was predicted on,
never to a molecule that merely shares a name, and then reads its measured
activities against one declared target. Query shapes checked against the live
service on 2026-09-26 (terfenadine → CHEMBL17157, hERG = CHEMBL240).

Transport discipline is PubChem's: host allowlist checked at start-up, no
redirects, a bounded body, a content-type check, retries for transport
failures only, and a circuit breaker.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any
from urllib.parse import urlparse

import httpx

from ...config import ChemblSettings
from ...domain.errors import EvidenceUnavailable, ProviderRateLimited
from ...domain.evidence import SourceIdentifier, SourceType
from ..circuit_breaker import CircuitBreaker, CircuitOpen
from ..interfaces import SearchHit
from ..transport import rebuild_decoded_response, retry_transient

#: Product endpoints with a ChEMBL target. A closed map: Tox21 assays are
#: cell-based readouts, not single targets, and are not guessed at.
TARGETS = {"herg": ("CHEMBL240", "Potassium voltage-gated channel subfamily H member 2 (hERG)")}

#: Measurement types read. Potency and affinity only; a percent inhibition at
#: one concentration is not comparable across assays without its dose.
STANDARD_TYPES = ("IC50", "Ki", "Kd", "EC50")

_ACTIVITY_FIELDS = (
    "activity_id,standard_type,standard_relation,standard_value,standard_units,pchembl_value,"
    "assay_chembl_id,assay_description,assay_type,document_chembl_id,document_year,"
    "target_pref_name,target_organism"
)


@dataclass(frozen=True)
class ActivityLookup:
    """What ChEMBL holds for one structure and one target."""

    molecule_chembl_id: str | None
    target_chembl_id: str
    hits: tuple[SearchHit, ...]
    #: How many ChEMBL molecules the structure matched; the first is used.
    structure_matches: int = 0


class ChemblActivityProvider:
    name = "chembl"

    def __init__(self, settings: ChemblSettings, *,
                 transport: httpx.AsyncBaseTransport | None = None) -> None:
        host = urlparse(settings.base_url).hostname or ""
        if host not in settings.allowed_hosts:
            raise ValueError(
                f"ChEMBL base_url host {host!r} is not in its own allowed_hosts "
                f"{settings.allowed_hosts!r} — this deployment is misconfigured"
            )
        self._settings = settings
        self._client = httpx.AsyncClient(
            base_url=settings.base_url,
            timeout=httpx.Timeout(settings.hard_timeout_s, connect=settings.timeout_s),
            transport=transport, headers={"accept": "application/json"},
            follow_redirects=False,
        )
        self._circuit = CircuitBreaker(
            failure_threshold=settings.circuit_failure_threshold,
            reset_after_s=settings.circuit_reset_after_s,
        )

    @property
    def allowed_hosts(self) -> tuple[str, ...]:
        return self._settings.allowed_hosts

    async def aclose(self) -> None:
        await self._client.aclose()

    async def activities(self, *, canonical_smiles: str, target: str, limit: int) -> ActivityLookup:
        target_id, target_name = TARGETS[target]
        molecules = await self._get("/molecule.json", params={
            "molecule_structures__canonical_smiles__flexmatch": canonical_smiles,
            "only": "molecule_chembl_id,pref_name", "limit": 1,
        })
        found = (molecules.get("molecules") or [])
        matches = int(((molecules.get("page_meta") or {}).get("total_count")) or len(found))
        if not found:
            return ActivityLookup(molecule_chembl_id=None, target_chembl_id=target_id, hits=())
        molecule_id = str(found[0]["molecule_chembl_id"])
        name = found[0].get("pref_name") or molecule_id
        body = await self._get("/activity.json", params={
            "molecule_chembl_id": molecule_id, "target_chembl_id": target_id,
            "standard_type__in": ",".join(STANDARD_TYPES), "only": _ACTIVITY_FIELDS,
            "limit": max(1, min(int(limit), 50)),
        })
        hits = tuple(
            self._hit(row, molecule_id=molecule_id, molecule_name=name,
                      target_id=target_id, target_name=target_name)
            for row in body.get("activities") or ()
            if row.get("activity_id") is not None and row.get("standard_value") is not None
        )
        return ActivityLookup(molecule_chembl_id=molecule_id, target_chembl_id=target_id, hits=hits,
                              structure_matches=matches)

    def _hit(self, row: dict[str, Any], *, molecule_id: str, molecule_name: str,
             target_id: str, target_name: str) -> SearchHit:
        relation = row.get("standard_relation") or "="
        measurement = (f"{row.get('standard_type')} {relation} {row.get('standard_value')} "
                       f"{row.get('standard_units') or ''}").strip()
        year = row.get("document_year")
        return SearchHit(
            provider_record_id=f"activity:{row['activity_id']}",
            source_type=SourceType.DATABASE,
            title=f"{molecule_name} ({molecule_id}) against {target_name}: {measurement}",
            published_at=date(int(year), 1, 1) if year else None,
            canonical_url=f"https://www.ebi.ac.uk/chembl/compound_report_card/{molecule_id}/",
            identifier=SourceIdentifier(other=f"chembl_activity:{row['activity_id']}"),
            abstract_or_excerpt=(
                f"Measured {measurement} for {molecule_name} ({molecule_id}) against "
                f"{target_name} ({target_id}) in assay {row.get('assay_chembl_id')} "
                f"[{row.get('assay_type') or 'type not stated'}]: "
                f"{row.get('assay_description') or 'no assay description'}. "
                f"Source document {row.get('document_chembl_id') or 'not stated'}."
            ),
            normalized_facts={
                "database": "ChEMBL", "molecule_chembl_id": molecule_id,
                "target_chembl_id": target_id, "target_organism": row.get("target_organism"),
                "standard_type": row.get("standard_type"), "standard_relation": relation,
                "standard_value": row.get("standard_value"),
                "standard_units": row.get("standard_units"),
                "pchembl_value": row.get("pchembl_value"),
                "assay_chembl_id": row.get("assay_chembl_id"),
                "assay_type": row.get("assay_type"),
                "document_chembl_id": row.get("document_chembl_id"),
            },
            raw=row,
        )

    # --- transport, mirroring the PubChem provider --------------------------

    async def _get(self, path: str, *, params: dict[str, Any]) -> dict[str, Any]:
        try:
            self._circuit.before_call()
        except CircuitOpen as exc:
            raise EvidenceUnavailable(str(exc)) from exc
        try:
            response = await retry_transient(
                lambda: self._read_bounded(path, params),
                attempts=max(1, self._settings.retry_attempts),
                backoff_s=self._settings.retry_backoff_s,
                retry_on=(httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout),
                should_retry=lambda r: r.status_code >= 500,
            )
        except EvidenceUnavailable:
            self._circuit.record_failure()
            raise
        except httpx.HTTPError as exc:
            self._circuit.record_failure()
            raise EvidenceUnavailable(f"ChEMBL transport failure: {type(exc).__name__}") from exc
        if response.status_code == 429:
            self._circuit.record_failure()
            raise ProviderRateLimited("ChEMBL rate-limited this request")
        if response.status_code >= 500:
            self._circuit.record_failure()
            raise EvidenceUnavailable(f"ChEMBL answered {response.status_code}")
        self._circuit.record_success()
        if response.status_code == 404:
            return {}
        if response.status_code != 200:
            raise EvidenceUnavailable(f"ChEMBL answered {response.status_code}")
        media_type = (response.headers.get("content-type") or "").split(";")[0].strip().lower()
        if media_type and media_type not in self._settings.allowed_content_types:
            raise EvidenceUnavailable(f"ChEMBL answered {media_type!r}, which is not JSON")
        try:
            return response.json()
        except ValueError as exc:
            raise EvidenceUnavailable("ChEMBL returned a non-JSON response") from exc

    async def _read_bounded(self, path: str, params: dict[str, Any]) -> httpx.Response:
        limit = self._settings.max_response_bytes
        request = self._client.build_request("GET", path, params=params)
        response = await self._client.send(request, stream=True)
        try:
            chunks: list[bytes] = []
            total = 0
            async for chunk in response.aiter_bytes():
                total += len(chunk)
                if total > limit:
                    raise EvidenceUnavailable(f"ChEMBL sent more than {limit} bytes")
                chunks.append(chunk)
        finally:
            await response.aclose()
        return rebuild_decoded_response(response=response, body=b"".join(chunks), request=request)


def build_chembl_provider(settings: ChemblSettings) -> ChemblActivityProvider | None:
    if not settings.provider:
        return None
    if settings.provider == "chembl":
        return ChemblActivityProvider(settings)
    raise ValueError(f"unknown ChEMBL provider: {settings.provider!r}")
